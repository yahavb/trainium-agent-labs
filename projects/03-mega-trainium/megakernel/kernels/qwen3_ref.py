"""Reference implementation and layout helpers for the Qwen3 batch=1 decode megakernel.

Everything here is plain torch so it serves three purposes:
  1. a float32 CPU reference the NKI kernel is checked against,
  2. the per-op baseline when run in bf16 under torch.compile(backend="neuron_libtorch"),
  3. the single place that defines the tensor LAYOUTS the kernel consumes.

Layouts (L = num_layers, B = batch, H = hidden, d = head_dim, q = q_heads, kv = kv_heads, I = intermediate):
  X        [B, 1, H]                      hidden states for the one new token
  W_qkv    [L, H, (q + 2*kv) * d]         Q heads, then K heads, then V heads (HF weight.T)
  W_out    [L, q*d, H]
  W_gate   [L, H, I]   W_up [L, H, I]   W_down [L, I, H]
  g_attn   [L, 1, H]   g_mlp [L, 1, H]    RMSNorm gammas (input_layernorm, post_attention_layernorm)
  g_q      [L, 1, d]   g_k   [L, 1, d]    Qwen3 per-head QK-norm gammas
  K_cache  [L, B, kv, d, S_max]           transposed flat cache (K_cache_transposed=True)
  V_cache  [L, B, kv, S_max, d]
  cos, sin [d//2, B, 1]                   RoPE tables for the new token (first half of the HF table)
  mask     [S_ctx, B, q, 1] uint8         1 = attend; must be 0 at and after the write position
  pos      [B, 1] uint32                  cache write position of the new token
"""
from __future__ import annotations

import math
import torch


def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)) * g.float()


def rope_rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF rotate_half convention. x [B, S, heads, d]; cos/sin [d//2, B, S]."""
    d = x.shape[-1]
    c = cos.permute(1, 2, 0).unsqueeze(2).float()  # [B, S, 1, d//2]
    s = sin.permute(1, 2, 0).unsqueeze(2).float()
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def rope_tables(positions: torch.Tensor, d: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [d//2, B, 1] for integer positions [B]."""
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))  # [d//2]
    ang = positions.float()[:, None] * inv[None, :]                             # [B, d//2]
    return ang.cos().T.unsqueeze(-1), ang.sin().T.unsqueeze(-1)


def decode_mask(pos: torch.Tensor, S_ctx: int, q_heads: int) -> torch.Tensor:
    """[S_ctx, B, q, 1] uint8 in the attention_tkg pregenerated-mask convention.

    Rows [0, S_ctx-1) gate the prior cache slots (1 = attend, so 1 iff slot < write position).
    The LAST row is the ACTIVE token's own column: attention_tkg places the new token's score in
    the last column of the context tile (gen_mask_tkg._load_active_mask: "bottom-right chunk"),
    so it must be 1 or the current token is silently ignored. Cache slot S_ctx-1 is therefore
    never read as prior; keep the write position below S_ctx-1 or grow S_ctx.
    """
    B = pos.shape[0]
    s = torch.arange(S_ctx)[:, None]                      # [S_ctx, 1]
    m = (s < pos.view(1, B).long()).to(torch.uint8)       # [S_ctx, B]
    m[S_ctx - 1] = 1                                      # active-token column
    return m[:, :, None, None].expand(S_ctx, B, q_heads, 1).contiguous()


def pack_layer_weights(layers, dtype=torch.bfloat16):
    """Stack HF Qwen3DecoderLayer parameters into the kernel layouts. `layers` is an iterable of
    dicts with HF names (self_attn.q_proj.weight, ...). Returns a dict of stacked tensors."""
    cols = {k: [] for k in ("W_qkv", "W_out", "W_gate", "W_up", "W_down", "g_attn", "g_mlp", "g_q", "g_k")}
    for p in layers:
        cols["W_qkv"].append(torch.cat([p["self_attn.q_proj.weight"].T, p["self_attn.k_proj.weight"].T,
                                        p["self_attn.v_proj.weight"].T], dim=1))
        cols["W_out"].append(p["self_attn.o_proj.weight"].T)
        cols["W_gate"].append(p["mlp.gate_proj.weight"].T)
        cols["W_up"].append(p["mlp.up_proj.weight"].T)
        cols["W_down"].append(p["mlp.down_proj.weight"].T)
        cols["g_attn"].append(p["input_layernorm.weight"].view(1, -1))
        cols["g_mlp"].append(p["post_attention_layernorm.weight"].view(1, -1))
        cols["g_q"].append(p["self_attn.q_norm.weight"].view(1, -1))
        cols["g_k"].append(p["self_attn.k_norm.weight"].view(1, -1))
    return {k: torch.stack(v).contiguous().to(dtype) for k, v in cols.items()}


def decode_layers_ref(X, W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k,
                      K_cache, V_cache, cos, sin, mask, pos, num_layers: int, eps: float,
                      q_heads: int, kv_heads: int, d: int, compute_dtype=torch.float32, active_excluded: bool = False):
    """Functional reference: returns (hidden [B,1,H] in compute_dtype, K_cache', V_cache').

    Math in `compute_dtype` (float32 on CPU for the reference; bfloat16 on device for the baseline).
    active_excluded=True is a deliberately WRONG variant (the new token does not attend to itself), used
    by the checker to recognise the mask-convention bug from ATTEMPTS.md attempt 1.
    """
    B, S, H = X.shape
    assert S == 1, "reference handles one new token per sequence"
    cd = compute_dtype
    scale = 1.0 / math.sqrt(d)
    g = q_heads // kv_heads
    S_ctx = mask.shape[0]
    m = mask.view(S_ctx, B, kv_heads, g, 1).permute(1, 4, 2, 3, 0).bool().clone()  # [B,1,kv,g,S_ctx]
    m[..., S_ctx - 1] = False   # last mask row is the active token's column (see decode_mask); not a prior slot
    cur = X.to(cd)
    for l in range(num_layers):
        h = rmsnorm(cur, g_attn[l, 0], eps).to(cd)
        qkv = h @ W_qkv[l].to(cd)                                   # [B,1,(q+2kv)d]
        qp = qkv[..., : q_heads * d].reshape(B, 1, q_heads, d)
        kp = qkv[..., q_heads * d : (q_heads + kv_heads) * d].reshape(B, 1, kv_heads, d)
        v = qkv[..., (q_heads + kv_heads) * d :].reshape(B, 1, kv_heads, d).to(cd)
        qn = rmsnorm(qp, g_q[l, 0], eps)
        kn = rmsnorm(kp, g_k[l, 0], eps)
        qr = rope_rotate_half(qn, cos, sin).to(cd)                    # [B,1,q,d]
        kr = rope_rotate_half(kn, cos, sin).to(cd)                    # [B,1,kv,d]
        # cache write, expressed as a one-hot blend so the same code traces under torch.compile
        # (the neuron_libtorch FX passes reject mixed int/slice/tensor index_put)
        onehot = (torch.arange(K_cache.shape[-1], device=X.device).view(1, 1, 1, -1) == pos.view(B, 1, 1, 1).long())
        Kl = torch.where(onehot, kr[:, 0].unsqueeze(-1).to(K_cache.dtype), K_cache[l])                  # [B,kv,d,S_max]
        Vl = torch.where(onehot.transpose(-1, -2), v[:, 0].unsqueeze(-2).to(V_cache.dtype), V_cache[l])   # [B,kv,S_max,d]
        K_cache = torch.cat([K_cache[:l], Kl.unsqueeze(0), K_cache[l + 1:]], dim=0)
        V_cache = torch.cat([V_cache[:l], Vl.unsqueeze(0), V_cache[l + 1:]], dim=0)
        Kp = Kl[:, :, :, :S_ctx].to(cd)                                # [B,kv,d,S_ctx]
        Vp = Vl[:, :, :S_ctx, :].to(cd)                                # [B,kv,S_ctx,d]
        qg = qr.reshape(B, 1, kv_heads, g, d)
        sp = torch.einsum("bskgd,bkdc->bskgc", qg, Kp) * scale        # [B,1,kv,g,S_ctx]
        sp = torch.where(m, sp, torch.full_like(sp, float("-inf")))
        sa = (qg * kr.reshape(B, 1, kv_heads, 1, d)).sum(-1, keepdim=True) * scale  # [B,1,kv,g,1]
        if active_excluded:
            sa = torch.full_like(sa, float("-inf"))
        pr = torch.softmax(torch.cat([sp, sa], dim=-1).float(), dim=-1).to(cd)
        out = torch.einsum("bskgc,bkcd->bskgd", pr[..., :S_ctx], Vp) + pr[..., S_ctx:] * v.reshape(B, 1, kv_heads, 1, d)
        attn = out.reshape(B, 1, q_heads * d) @ W_out[l].to(cd)
        cur = cur + attn
        h2 = rmsnorm(cur, g_mlp[l, 0], eps).to(cd)
        mlp = (torch.nn.functional.silu(h2 @ W_gate[l].to(cd)) * (h2 @ W_up[l].to(cd))) @ W_down[l].to(cd)
        cur = cur + mlp
    return cur, K_cache, V_cache
