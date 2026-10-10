"""Neuron-friendly rewrite of ONE causal Wan2.1-1.3B block from StreamDiffusionV2.

Original: repos/StreamDiffusionV2/models/wan/causal_model.py, CausalWanAttentionBlock (+ CausalWanSelfAttention,
WanT2VCrossAttention). See tier2/SHAPES.md for the op-by-op list of what was replaced.

Graph contract (all tensors, all shapes fixed at trace time):

    y, new_k_cache, new_v_cache = block(x, e, context, rope_cos, rope_sin, k_cache, v_cache, attn_bias)

    x          [1, L_chunk, dim]                 tokens of the chunk being denoised
    e          [1, F, 6, dim]                    time modulation (time_projection output), F = latent frames per chunk
    context    [1, L_text, dim]                  text tokens AFTER model.text_embedding
    rope_cos   [1, L_chunk, 1, head_dim // 2]    host-precomputed for the chunk's (frame, h, w) positions
    rope_sin   [1, L_chunk, 1, head_dim // 2]
    k_cache    [1, L_cache, heads, head_dim]     keys stored AFTER RoPE (same as the repo)
    v_cache    [1, L_cache, heads, head_dim]
    attn_bias  [1, 1, 1, L_cache]                additive, 0 = attend, MASK_NEG = empty slot; indexes the UPDATED cache

Cache update, no pointers in the graph:

    new_cache = cat(cache[:, :S], cache[:, S + L_chunk:], new_kv)          S = sink_tokens

The chunk attends over new_cache, i.e. over itself plus everything that survived the shift, exactly like the repo,
which writes the chunk's K/V into the cache first and then attends over the whole cache.

With S = 0 this is the plain FIFO `cat(cache[:, L_chunk:], new_kv)`. With S = num_sink_tokens * frame_tokens (the
default, matching the repo config) the first S tokens are passed through untouched and only the rest is a FIFO.

How this relates to the repo's ring buffer (causal_model.py:304-398):
  * The repo keeps the cache in place and overwrites the slot named by `evict_idx[0]`, then rotates that list, so
    frames sit in arbitrary slot order. Attention has no mask and RoPE is baked into the keys, so slot order does
    not change the result: the SET of cached frames is what matters. The shift here keeps the same set.
  * The repo never evicts the first `sink_size` frames (slots 0..2); its ring only cycles the remaining slots.
    That is the S-token passthrough above.
  * While the cache is filling, the repo attends over the filled prefix only (`cache_seqlens`). Here the cache is
    always full length, so the host passes attn_bias to hide empty slots, and for the first `num_sink` chunks copies
    the new K/V from the tail into its sink slot (CacheState below). After that the host does nothing.
  * NOT in the graph (host-side, see SHAPES.md): the adaptive sink refresh (adapt_sink_threshold, data dependent),
    the t_refresh RoPE re-alignment of cached keys, and the batch-of-denoising-steps layout (one cache per step).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ART_DIR = Path("/workspace/livevid/artifacts/tier2")
CKPT_GLOB = "/root/.cache/huggingface/hub/models--jerryfeng--StreamDiffusionV2/snapshots/*/wan_causal_dmd_v2v/model.pt"
MASK_NEG = -1.0e4  # finite so bf16 softmax stays NaN-free; exp(-1e4) is exactly 0 in fp32 and bf16


@dataclass(frozen=True)
class BlockCfg:
    """Shapes for one block call. Defaults are the StreamDiffusionV2 demo v2v config (see SHAPES.md)."""
    height: int = 512
    width: int = 512
    frames_per_chunk: int = 1   # num_frame_per_block (latent frames per chunk)
    num_kv_cache: int = 6       # cache window, in latent frames
    num_sink: int = 3           # num_sink_tokens (it counts frames, despite the name)
    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    text_len: int = 512
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6

    @property
    def grid_h(self): return self.height // 16

    @property
    def grid_w(self): return self.width // 16

    @property
    def frame_tokens(self): return self.grid_h * self.grid_w

    @property
    def l_chunk(self): return self.frames_per_chunk * self.frame_tokens

    @property
    def l_cache(self): return self.num_kv_cache * self.frame_tokens

    @property
    def sink_tokens(self): return self.num_sink * self.frame_tokens

    @property
    def head_dim(self): return self.dim // self.num_heads

    @property
    def tag(self):
        return (f"{self.height}x{self.width}_L{self.l_chunk}_cache{self.l_cache}_sink{self.sink_tokens}"
                f"_text{self.text_len}")


class RMSNorm(nn.Module):
    """WanRMSNorm: normalise in fp32 like the original, scale in the working dtype."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class _Attn(nn.Module):
    """Parameter container with the same state-dict names as WanSelfAttention."""

    def __init__(self, dim, eps):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)


def apply_rope(x, cos, sin):
    """Real-valued RoPE. x [B, L, N, D] holds D/2 (re, im) pairs; cos/sin [1, L, 1, D/2].

    Same as view_as_real(view_as_complex(x) * (cos + i sin)): (a + ib)(c + is) = (ac - bs) + i(as + bc).
    """
    b, l, n, d = x.shape
    xp = x.reshape(b, l, n, d // 2, 2)
    a, bb = xp[..., 0], xp[..., 1]
    return torch.stack([a * cos - bb * sin, a * sin + bb * cos], dim=-1).reshape(b, l, n, d)


class CausalBlockPort(nn.Module):

    def __init__(self, cfg: BlockCfg = BlockCfg(), manual_attn: bool = False):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.frames = cfg.frames_per_chunk
        self.frame_tokens = cfg.frame_tokens
        self.l_chunk = cfg.l_chunk
        self.sink_tokens = cfg.sink_tokens
        self.manual_attn = manual_attn  # fallback if the compiler rejects aten::scaled_dot_product_attention

        dim, eps = cfg.dim, cfg.eps
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = _Attn(dim, eps)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.cross_attn = _Attn(dim, eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, cfg.ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(cfg.ffn_dim, dim))
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def _attend(self, q, k, v, bias=None):
        """q [1, Lq, N, D], k/v [1, Lk, N, D] -> [1, Lq, N * D]."""
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.manual_attn:
            w = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
            if bias is not None:
                w = w + bias
            out = torch.matmul(torch.softmax(w, dim=-1), v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return out.transpose(1, 2).flatten(2)

    def _per_frame(self, x, scale, shift=None):
        """Apply per-frame modulation ([1, F, 1, dim]) to tokens [1, F * frame_tokens, dim]."""
        x = x.reshape(1, self.frames, self.frame_tokens, -1) * scale
        if shift is not None:
            x = x + shift
        return x.reshape(1, self.l_chunk, -1)

    def forward(self, x, e, context, rope_cos, rope_sin, k_cache, v_cache, attn_bias):
        n, d, lc, s = self.num_heads, self.head_dim, self.l_chunk, self.sink_tokens
        e0, e1, e2, e3, e4, e5 = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)  # each [1, F, 1, dim]

        # self-attention over the shifted cache + this chunk
        h = self._per_frame(self.norm1(x), 1 + e1, e0)
        sa = self.self_attn
        q = apply_rope(sa.norm_q(sa.q(h)).view(1, lc, n, d), rope_cos, rope_sin)
        k = apply_rope(sa.norm_k(sa.k(h)).view(1, lc, n, d), rope_cos, rope_sin)
        v = sa.v(h).view(1, lc, n, d)
        new_k_cache = torch.cat([k_cache[:, :s], k_cache[:, s + lc:], k], dim=1)
        new_v_cache = torch.cat([v_cache[:, :s], v_cache[:, s + lc:], v], dim=1)
        y = sa.o(self._attend(q, new_k_cache, new_v_cache, attn_bias))
        x = x + self._per_frame(y, e2)

        # cross-attention over the text tokens (no mask: the repo passes context_lens=None)
        ca = self.cross_attn
        hq = ca.norm_q(ca.q(self.norm3(x))).view(1, lc, n, d)
        ck = ca.norm_k(ca.k(context)).view(1, -1, n, d)
        cv = ca.v(context).view(1, -1, n, d)
        x = x + ca.o(self._attend(hq, ck, cv))

        # feed-forward
        y = self.ffn(self._per_frame(self.norm2(x), 1 + e4, e3))
        x = x + self._per_frame(y, e5)
        return x, new_k_cache, new_v_cache


# --------------------------------------------------------------------------------------------------------------
# Host-side helpers (never traced)
# --------------------------------------------------------------------------------------------------------------

def rope_cos_sin(cfg: BlockCfg, start_frame: int, dtype=torch.float32):
    """cos/sin [1, L_chunk, 1, head_dim/2] for latent frames start_frame .. start_frame + F - 1.

    Mirrors rope_params (wan_base/modules/model.py:29) and _get_causal_rope_freqs (causal_model.py:57): the head
    dim is split into (temporal, height, width) bands and token order is frame-major, then row, then column.
    """
    d = cfg.head_dim
    f, h, w = cfg.frames_per_chunk, cfg.grid_h, cfg.grid_w

    def angles(pos, dim):
        inv = 1.0 / torch.pow(10000, torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        return torch.outer(pos.to(torch.float64), inv)

    t = angles(torch.arange(start_frame, start_frame + f), d - 4 * (d // 6)).repeat_interleave(h * w, dim=0)
    hh = angles(torch.arange(h), 2 * (d // 6)).repeat_interleave(w, dim=0).repeat(f, 1)
    ww = angles(torch.arange(w), 2 * (d // 6)).repeat(h, 1).repeat(f, 1)
    ang = torch.cat([t, hh, ww], dim=-1).view(1, f * h * w, 1, d // 2)
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


class CacheState:
    """Host bookkeeping that makes the fixed-size graph reproduce the repo's cold start (no graph pointers).

    Layout: [num_sink sink slots | (num_kv_cache - num_sink) FIFO slots], one slot = one chunk.
    Usage per chunk:  bias = st.bias();  y, k, v = block(..., st.k, st.v, bias);  st.commit(k, v)
    """

    def __init__(self, cfg: BlockCfg, dtype=torch.float32):
        assert cfg.frames_per_chunk == 1, "slot bookkeeping assumes one latent frame per chunk"
        self.cfg = cfg
        shape = (1, cfg.l_cache, cfg.num_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype)
        self.v = torch.zeros(shape, dtype=dtype)
        self.frames = [None] * cfg.num_kv_cache  # frame id held by each slot, None = empty
        self.seen = 0

    def _shifted(self):
        s = self.cfg.num_sink
        return self.frames[:s] + self.frames[s + 1:] + [self.seen]

    def bias(self, dtype=None):
        valid = torch.tensor([f is not None for f in self._shifted()])
        b = torch.where(valid, 0.0, MASK_NEG).repeat_interleave(self.cfg.frame_tokens)
        return b.view(1, 1, 1, -1).to(dtype or self.k.dtype)

    def commit(self, new_k, new_v):
        cfg, l = self.cfg, self.cfg.frame_tokens
        self.frames = self._shifted()
        self.k, self.v = new_k, new_v
        if self.seen < cfg.num_sink:  # first frames become sinks: move them from the FIFO tail to a sink slot
            self.k, self.v = new_k.clone(), new_v.clone()
            dst = slice(self.seen * l, (self.seen + 1) * l)
            self.k[:, dst] = new_k[:, -l:]
            self.v[:, dst] = new_v[:, -l:]
            self.frames[self.seen] = self.seen
            self.frames[-1] = None
        self.seen += 1


def extract_weights(layer: int = 0, force: bool = False):
    """Pull one block's weights (plus the tiny embedding layers used to build realistic inputs) out of model.pt."""
    out = ART_DIR / f"block{layer}_fp32.pt"
    if out.exists() and not force:
        return torch.load(out, map_location="cpu")
    paths = glob.glob(CKPT_GLOB)
    if not paths:
        raise FileNotFoundError(f"no checkpoint matches {CKPT_GLOB}")
    sd = torch.load(paths[0], map_location="cpu", weights_only=False)["generator"]
    prefix = f"model.blocks.{layer}."
    block = {k[len(prefix):]: v.float().clone() for k, v in sd.items() if k.startswith(prefix)}
    embed_keys = ("patch_embedding", "text_embedding", "time_embedding", "time_projection")
    embed = {k[len("model."):]: v.float().clone() for k, v in sd.items() if k.split(".")[1] in embed_keys}
    data = {"layer": layer, "ckpt": paths[0], "block": block, "embed": embed}
    ART_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save(data, tmp)
    os.replace(tmp, out)
    return data


def extract_all_layers(num_layers: int = 30):
    """One pass over model.pt that writes block<i>_fp32.pt for every layer still missing."""
    todo = [i for i in range(num_layers) if not (ART_DIR / f"block{i}_fp32.pt").exists()]
    if not todo:
        return
    sd = torch.load(glob.glob(CKPT_GLOB)[0], map_location="cpu", weights_only=False)["generator"]
    ART_DIR.mkdir(parents=True, exist_ok=True)
    for i in todo:
        prefix = f"model.blocks.{i}."
        block = {k[len(prefix):]: v.float().clone() for k, v in sd.items() if k.startswith(prefix)}
        tmp = ART_DIR / f"block{i}_fp32.tmp"
        torch.save({"layer": i, "block": block, "embed": {}}, tmp)
        os.replace(tmp, ART_DIR / f"block{i}_fp32.pt")


def load_port(cfg: BlockCfg = BlockCfg(), layer: int = 0, manual_attn: bool = False):
    data = extract_weights(layer)
    model = CausalBlockPort(cfg, manual_attn=manual_attn)
    model.load_state_dict(data["block"], strict=True)
    return model.eval().requires_grad_(False), data


def make_inputs(cfg: BlockCfg, embed: dict, seed: int, timestep: float = 700.0, text_tokens: int = 48):
    """Random latent / text pushed through the REAL embedding layers, so x, e and context have real statistics.

    Returns fp32 (x [1, L_chunk, dim], e [1, F, 6, dim], context [1, L_text, dim]).
    """
    g = torch.Generator().manual_seed(seed)
    f = cfg.frames_per_chunk
    latent = torch.randn(1, 16, f, cfg.height // 8, cfg.width // 8, generator=g)
    x = F.conv3d(latent, embed["patch_embedding.weight"], embed["patch_embedding.bias"], stride=(1, 2, 2))
    x = x.flatten(2).transpose(1, 2)

    half = cfg.freq_dim // 2
    t = torch.full((f,), timestep, dtype=torch.float64)
    sinus = torch.outer(t, torch.pow(10000, -torch.arange(half, dtype=torch.float64) / half))
    emb = torch.cat([torch.cos(sinus), torch.sin(sinus)], dim=1).float()
    emb = F.linear(F.silu(F.linear(emb, embed["time_embedding.0.weight"], embed["time_embedding.0.bias"])),
                   embed["time_embedding.2.weight"], embed["time_embedding.2.bias"])
    e = F.linear(F.silu(emb), embed["time_projection.1.weight"], embed["time_projection.1.bias"])
    e = e.view(1, f, 6, cfg.dim)

    text = torch.zeros(1, cfg.text_len, cfg.text_dim)  # the repo zero-pads prompts to text_len
    text[:, :text_tokens] = torch.randn(1, text_tokens, cfg.text_dim, generator=g) * 0.1
    context = F.linear(F.gelu(F.linear(text, embed["text_embedding.0.weight"], embed["text_embedding.0.bias"]),
                              approximate="tanh"),
                       embed["text_embedding.2.weight"], embed["text_embedding.2.bias"])
    return x, e, context


def cosine(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def max_abs(a, b):
    return (a.double() - b.double()).abs().max().item()
