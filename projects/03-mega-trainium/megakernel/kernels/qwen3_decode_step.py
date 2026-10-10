"""Qwen3 full batch=1 decode step in ONE NKI launch: token id -> next token id.

  token id --(indirect DMA row gather)--> embedding [1, 1, H]
           --> 36 decoder layers (qwen3_megakernel.qwen3_decode_layers, KV cache updated in place)
           --> final RMSNorm (nkilib rmsnorm_tkg, result stays in SBUF as [128, 1, H//128])
           --> LM head (nkilib output_projection_tkg, vocab sharded over the two physical cores)
           --> greedy argmax (nkilib cascaded_max_core) --> next token id [1, 1] uint32

LM-head weight layout. rmsnorm_tkg leaves the normed hidden in SBUF as out[p, 0, f] with
    h(p, f) = (f // F) * (128 * F) + p * F + f % F,   F = H // 128 // lnc
(see its docstring pseudocode), and output_projection_tkg contracts attention[d, b, n, s] against
weight row n * 128 + d. Feeding the SBUF tile in directly as [D=128, B=1, N=H//128, S=1] therefore
needs W_lm_packed[n * 128 + d, :] = lm_head.weight.T[h(d, n), :]; pack_lm_head() does that on the
host, so the kernel needs no transpose.
"""
import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.max.cascaded_max import CascadedMaxConfig, cascaded_max_core
from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg
from nkilib.core.utils.allocator import BufferManager
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.logging import get_logger

from qwen3_megakernel import _SBM_SIZE_BYTES, qwen3_decode_layers


def embed_lookup(token_ids, embed, prg_id: int, n_prgs: int):
    """token_ids [1, 1] uint32 @ HBM, embed [V, H] @ HBM -> X [1, 1, H] @ shared HBM (written by core 0).

    Also used for any one-row table lookup (RoPE cos/sin rows by position)."""
    V, H = embed.shape
    tok_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tok_sb, src=token_ids[0:1, 0:1])
    row_sb = nl.ndarray((1, H), dtype=embed.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=row_sb, src=embed.ap(pattern=[[1, 1], [1, H]], offset=0, vector_offset=tok_sb, indirect_dim=0))
    X = nl.ndarray((1, 1, H), dtype=embed.dtype, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=X.reshape((1, H)), src=row_sb)
    if n_prgs > 1:
        nisa.core_barrier(data=X, cores=(0, 1))
    return X


def build_decode_mask(pos_ids, S_ctx: int, q_heads: int, prg_id: int, n_prgs: int):
    """[S_ctx, 1, q, 1] uint8 decode mask from pos on the device; same values as qwen3_ref.decode_mask:
    1 for cache slots s < pos, and 1 in the last row (attention_tkg's active-token column).

    Built once per step and shared by every layer. (attention_block_tkg can build the prior mask itself
    from pos_ids, but that path names an op "rope_pos_ids_load" without a per-call prefix, so it compiles
    only once per kernel, not once per layer.) SBUF layout: slot s = p * F + j on partition p, F = S_ctx/128,
    with the q head copies innermost, which is exactly the HBM byte order of [S_ctx, 1, q, 1].
    """
    P = nl.tile_size.pmax
    F = S_ctx // P
    slot_i = nl.ndarray((P, F * q_heads), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(slot_i, [[1, F], [0, q_heads]], offset=0, channel_multiplier=F)
    slot = nl.ndarray((P, F * q_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slot, src=slot_i)
    pos_u = nl.ndarray((P, 1), dtype=pos_ids.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_u, src=pos_ids.ap(pattern=[[0, P], [1, 1]], offset=0))   # broadcast to all partitions
    pos_f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos_f, src=pos_u)
    prior = nl.ndarray((P, F * q_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=prior, data=slot, op0=nl.less, operand0=pos_f)
    last = nl.ndarray((P, F * q_heads), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=last, data=slot, op0=nl.greater_equal, operand0=float(S_ctx - 1))
    m_sb = nl.ndarray((P, F * q_heads), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=m_sb, data1=prior, data2=last, op=nl.maximum)
    mask = nl.ndarray((S_ctx, 1, q_heads, 1), dtype=nl.uint8, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=mask.reshape((P, F * q_heads)), src=m_sb)
    if n_prgs > 1:
        nisa.core_barrier(data=mask, cores=(0, 1))
    return mask


def mask_test(pos_ids, S_ctx: int, q_heads: int):
    _, n_prgs, prg_id = get_verified_program_sharding_info("mask_test", (0, 1), 2)
    return build_decode_mask(pos_ids, S_ctx, q_heads, prg_id, n_prgs)


def lm_head_argmax(hidden, g_final, W_lm, lm_bias, prg_id: int, n_prgs: int, eps: float):
    """hidden [1, 1, H] @ HBM -> (logits [1, Vp] @ HBM, next token [1, 1] uint32 @ HBM)."""
    _, _, H = hidden.shape
    H1 = H // nl.tile_size.pmax
    norm_sb = nl.ndarray((nl.tile_size.pmax, 1, H1), dtype=hidden.dtype, buffer=nl.sbuf)
    rmsnorm_tkg(input=hidden, gamma=g_final, output=norm_sb, eps=eps)
    # Manual placement in the same SBUF region the layer stack's MLP uses (free again by now). With
    # output_projection_tkg's default auto-allocating manager the compiler fails to place its tiles
    # next to the layer stack's manually placed buffers (NCC_EGCA111), although each compiles alone.
    sbm = BufferManager(0, _SBM_SIZE_BYTES, get_logger("lm_head"))
    sbm.set_auto_alloc(False)
    logits = output_projection_tkg(attention=norm_sb.reshape((nl.tile_size.pmax, 1, H1, 1)), weight=W_lm,
                                   bias=lm_bias, sbm=sbm)
    if n_prgs > 1:
        nisa.core_barrier(data=logits, cores=(0, 1))
    cfg = CascadedMaxConfig(logits.shape, inp_dtype=logits.dtype)
    _, idx_sb = cascaded_max_core(logits.reshape(cfg.inp_shape), cfg)
    token = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=token, src=idx_sb[:, 0:1])
    return logits, token


def qwen3_head_parts(token_ids, embed, hidden, g_final, W_lm, lm_bias, eps: float = 1e-6):
    """Test harness: embedding lookup and LM head + argmax, without the layer stack."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_head_parts", (0, 1), 2)
    X = embed_lookup(token_ids, embed, prg_id, n_prgs)
    logits, token = lm_head_argmax(hidden, g_final, W_lm, lm_bias, prg_id, n_prgs, eps)
    return X, logits, token


def qwen3_decode_step(
    token_ids,  # [1, 1] uint32
    embed,      # [V, H]
    W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache, cos, sin, mask, pos_ids,
    g_final,    # [1, H]
    W_lm,       # [H, Vp] packed by pack_lm_head
    lm_bias,    # [1, Vp] 0 for real vocab entries, -inf-like for padding (or None when Vp == V)
    num_layers: int,
    eps: float = 1e-6,
):
    """One full greedy decode step. Returns (next token [1, 1] uint32, logits [1, Vp], hidden [1, 1, H])."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_decode_step", (0, 1), 2)
    X = embed_lookup(token_ids, embed, prg_id, n_prgs)
    hidden = qwen3_decode_layers(X, W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache,
                                 cos, sin, mask, pos_ids, num_layers=num_layers, eps=eps)
    logits, token = lm_head_argmax(hidden, g_final, W_lm, lm_bias, prg_id, n_prgs, eps)
    return token, logits, hidden


qwen3_decode_step_jit = nki.jit(qwen3_decode_step)


def qwen3_decode_step_resident(
    token_ids,    # [1, 1] uint32: the token to process
    pos_ids,      # [1, 1] uint32: its position (= KV-cache write slot)
    embed,
    W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache,
    cos_table,    # [S_ctx, d//2] RoPE cos for every position (resident)
    sin_table,    # [S_ctx, d//2]
    g_final, W_lm, lm_bias,
    num_layers: int,
    S_ctx: int,
    eps: float = 1e-6,
):
    """qwen3_decode_step with every per-step input device-resident.

    The host uploads nothing per token: RoPE rows are gathered from resident tables at pos, the decode
    mask is built on the device from pos (build_decode_mask), and the kernel returns
    (next token, pos + 1) so the caller feeds the outputs of step t straight back in as the inputs of
    step t + 1. Returns (next_token [1, 1], next_pos [1, 1], logits [1, Vp]).
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_decode_step_resident", (0, 1), 2)
    half_d = cos_table.shape[1]
    q_heads = W_out.shape[1] // (2 * half_d)
    X = embed_lookup(token_ids, embed, prg_id, n_prgs)
    mask = build_decode_mask(pos_ids, S_ctx, q_heads, prg_id, n_prgs)
    cos = embed_lookup(pos_ids, cos_table, prg_id, n_prgs).reshape((half_d, 1, 1))
    sin = embed_lookup(pos_ids, sin_table, prg_id, n_prgs).reshape((half_d, 1, 1))
    hidden = qwen3_decode_layers(X, W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache,
                                 cos, sin, mask, pos_ids, num_layers=num_layers, eps=eps)
    logits, token = lm_head_argmax(hidden, g_final, W_lm, lm_bias, prg_id, n_prgs, eps)

    pos_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_sb, src=pos_ids[0:1, 0:1])
    next_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=next_sb, data=pos_sb, op0=nl.add, operand0=1)
    next_pos = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=next_pos, src=next_sb)
    return token, next_pos, logits


def qwen3_decode_loop(
    token_ids, pos_ids, embed,
    W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache,
    cos_table, sin_table, g_final, W_lm, lm_bias,
    num_layers: int,
    S_ctx: int,
    num_steps: int,
    eps: float = 1e-6,
):
    """Persistent decode: num_steps greedy tokens in ONE launch, each step's token fed to the next on chip.

    STATUS (2026-10-10): DOES NOT COMPILE. neuronx-cc 2.27 fails with an internal error
    (NCC_IBVF059 "tensor size must be more than 0", then walrus "empty MemoryLocationSet") on core 1,
    after ~15 min even at 2 layers. Kept because the loop mechanics were checked on a toy kernel
    (dynamic_range traced once, counter-indexed scatter correct) and because the measurement that
    motivates it stands: see NOTE.md.

    An on-device loop around the resident step: nl.dynamic_range runs on the device and is traced once
    (nl.sequential_range with a constant trip count unrolls in NKI 0.6, multiplying compile time). The
    current token and position live in two HBM words that each iteration reads and rewrites, the KV
    cache fills in place, and token k is scattered to out_tokens[0, k] through a step counter (the loop
    register itself cannot index a DMA). The host launches once per num_steps tokens and reads them all
    at the end. Returns (out_tokens [1, num_steps], next_pos [1, 1]).
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_decode_loop", (0, 1), 2)
    half_d = cos_table.shape[1]
    q_heads = W_out.shape[1] // (2 * half_d)
    tok_buf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    pos_buf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    out_tokens = nl.ndarray((1, num_steps), dtype=nl.uint32, buffer=nl.shared_hbm)
    init_sb = nl.ndarray((1, 2), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=init_sb[:, 0:1], src=token_ids[0:1, 0:1])
    nisa.dma_copy(dst=init_sb[:, 1:2], src=pos_ids[0:1, 0:1])
    if prg_id == 0:
        nisa.dma_copy(dst=tok_buf, src=init_sb[:, 0:1])
        nisa.dma_copy(dst=pos_buf, src=init_sb[:, 1:2])
    if n_prgs > 1:
        nisa.core_barrier(data=pos_buf, cores=(0, 1))
    step_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(step_sb, [[0, 1]], offset=0)

    for _ in nl.dynamic_range(num_steps):
        X = embed_lookup(tok_buf, embed, prg_id, n_prgs)
        cos = embed_lookup(pos_buf, cos_table, prg_id, n_prgs).reshape((half_d, 1, 1))
        sin = embed_lookup(pos_buf, sin_table, prg_id, n_prgs).reshape((half_d, 1, 1))
        mask = build_decode_mask(pos_buf, S_ctx, q_heads, prg_id, n_prgs)
        hidden = qwen3_decode_layers(X, W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache,
                                     cos, sin, mask, pos_buf, num_layers=num_layers, eps=eps)
        _, token = lm_head_argmax(hidden, g_final, W_lm, lm_bias, prg_id, n_prgs, eps)
        if n_prgs > 1:
            nisa.core_barrier(data=token, cores=(0, 1))
        st_sb = nl.ndarray((1, 2), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st_sb[:, 0:1], src=token[0:1, 0:1])
        nisa.dma_copy(dst=st_sb[:, 1:2], src=pos_buf[0:1, 0:1])
        nisa.tensor_scalar(dst=st_sb[:, 1:2], data=st_sb[:, 1:2], op0=nl.add, operand0=1)
        if prg_id == 0:
            nisa.dma_copy(dst=tok_buf, src=st_sb[:, 0:1])
            nisa.dma_copy(dst=pos_buf, src=st_sb[:, 1:2])
            nisa.dma_copy(dst=out_tokens.reshape((num_steps, 1)).ap(pattern=[[1, 1], [1, 1]], offset=0,
                                                                   vector_offset=step_sb, indirect_dim=0),
                          src=st_sb[:, 0:1])
        nisa.tensor_scalar(dst=step_sb, data=step_sb, op0=nl.add, operand0=1)
        if n_prgs > 1:
            nisa.core_barrier(data=pos_buf, cores=(0, 1))
    return out_tokens, pos_buf


def rope_tables_all(S_ctx: int, d: int, theta: float):
    """(cos, sin) [S_ctx, d//2] float32 for positions 0..S_ctx-1, HF convention."""
    import torch
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = torch.arange(S_ctx, dtype=torch.float32)[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def lm_head_row_perm(H: int, lnc: int):
    """perm[n * 128 + d] = h(d, n): which hidden index feeds packed LM-head row n * 128 + d."""
    import numpy as np
    P = 128
    F = H // P // lnc
    d = np.arange(P)[None, :]
    n = np.arange(H // P)[:, None]
    return ((n // F) * (P * F) + d * F + n % F).reshape(-1)


def pack_lm_head(lm_head_weight, lnc: int, pad_to: int = 1):
    """lm_head.weight [V, H] (torch) -> (W_lm [H, Vp], lm_bias [1, Vp] or None, V)."""
    import torch
    V, H = lm_head_weight.shape
    Vp = -(-V // pad_to) * pad_to
    W = lm_head_weight.t()[torch.as_tensor(lm_head_row_perm(H, lnc))]
    bias = None
    if Vp != V:
        W = torch.cat([W, torch.zeros(H, Vp - V, dtype=W.dtype)], dim=1)
        bias = torch.zeros(1, Vp, dtype=W.dtype)
        bias[0, V:] = -1e30
    return W.contiguous(), bias, V


# ---------------------------------------------------------------------------------------------
# Opt-in: hardware-DGE weight loads in nkilib's output_projection_tkg.
#
# Profile `head` showed the LM head streaming W_lm through software DGE: one DMA_DIRECT2D per weight
# row block issued serially on GpSimd, ~230 GB/s per physical core against ~435 GB/s attainable.
# _load_weight_h_block below is nkilib's helper with dge_mode=hwdge on its DMAs, alternating the
# Sync and Activation queues so two descriptor streams run in parallel. It replaces the module-level
# helper, so it also applies to the o_proj inside every attention block. Call before tracing.
# ---------------------------------------------------------------------------------------------
def _load_weight_h_block_hwdge(w_sbuf_slot, w_shard_hbm, h_block_size: int, h_block_offset: int, cfg):
    w_h_sliced = w_shard_hbm.slice(dim=2, start=h_block_offset, end=h_block_offset + h_block_size)
    if not cfg.use_double_row:
        for head_idx in nl.affine_range(0, cfg.n_size, 2):
            nisa.dma_copy(src=w_h_sliced.select(dim=0, index=head_idx), dst=w_sbuf_slot[head_idx][:, :h_block_size],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        for head_idx in nl.affine_range(1, cfg.n_size, 2):
            nisa.dma_copy(src=w_h_sliced.select(dim=0, index=head_idx), dst=w_sbuf_slot[head_idx][:, :h_block_size],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    else:
        for head_idx in nl.affine_range(0, cfg.n_size, 2):
            pair_idx = head_idx // 2
            nisa.dma_copy(src=w_h_sliced.select(dim=0, index=head_idx), dst=w_sbuf_slot[pair_idx][:, 0, :h_block_size],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            nisa.dma_copy(src=w_h_sliced.select(dim=0, index=head_idx + 1), dst=w_sbuf_slot[pair_idx][:, 1, :h_block_size],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)


def use_hwdge_output_projection():
    import nkilib.core.output_projection.output_projection_tkg as opt
    opt._load_weight_h_block = _load_weight_h_block_hwdge


# ---------------------------------------------------------------------------------------------
# Our own LM head: weights stationary, host-pre-tiled so every weight DMA moves 32 KB contiguous per
# partition (nkilib's output_projection_tkg moves ~1.2 KB per partition per DMA at this shape, ~250 GB/s).
# ---------------------------------------------------------------------------------------------
LMH_NB = 512           # vocab columns per weight block (4 tiles of 128)
LMH_RING = 4           # weight blocks in flight


def pack_lm_head_tiled(lm_head_weight, lnc: int, H1: int = 32, nb: int = LMH_NB):
    """lm_head.weight [V, H] -> (W_tiled [lnc, n_blocks, 128, H1 * nb], V, Vp).

    Row order matches rmsnorm_tkg's SBUF layout (lm_head_row_perm); vocab padded to lnc * n_blocks * nb.
    W_tiled[c, b, p, f * nb + j] = W[h(p, f), c * n_blocks * nb + b * nb + j]: one block is one DMA of
    H1 * nb contiguous elements per partition.
    """
    import torch
    V, H = lm_head_weight.shape
    per = lnc * nb
    Vp = -(-V // per) * per
    n_blocks = Vp // per
    Wt = lm_head_weight.t()[torch.as_tensor(lm_head_row_perm(H, lnc))]          # [H, V], row f*128 + p
    if Vp != V:
        Wt = torch.cat([Wt, torch.zeros(H, Vp - V, dtype=Wt.dtype)], dim=1)
    Wt = Wt.reshape(H1, 128, lnc, n_blocks, nb)                                   # [f, p, c, b, j]
    return Wt.permute(2, 3, 1, 0, 4).reshape(lnc, n_blocks, 128, H1 * nb).contiguous(), V, Vp


def lm_head_tiled_argmax(hidden, g_final, W_tiled, V: int, prg_id: int, n_prgs: int, eps: float):
    """Same contract as lm_head_argmax, with the LM head computed by our own weights-stationary matmul."""
    _, _, H = hidden.shape
    P = nl.tile_size.pmax
    H1 = H // P
    lnc_w, n_blocks, _, blk = W_tiled.shape
    nb = blk // H1
    nt = nb // P
    Vp = lnc_w * n_blocks * nb
    norm_sb = nl.ndarray((P, 1, H1), dtype=hidden.dtype, buffer=nl.sbuf)
    rmsnorm_tkg(input=hidden, gamma=g_final, output=norm_sb, eps=eps)
    x = norm_sb.reshape((P, H1))

    n_tiles = n_blocks * nt
    # Manual placement in the region the layer stack's manually placed buffers use (as for nkilib's head
    # in lm_head_argmax): auto-allocated, the ring's first weight DMA (no input dependency) can be hoisted
    # over the layer stack and overwrite manually placed buffers the compiler cannot see, which showed up
    # as an out-of-bounds KV-cache scatter (nrta 1006) on the second generated token.
    sbm = BufferManager(0, _SBM_SIZE_BYTES, get_logger("lm_head_tiled"))
    sbm.set_auto_alloc(False)
    sbm.open_scope(name="lm_head_tiled")
    ring = sbm.alloc((P, LMH_RING, blk), dtype=W_tiled.dtype, buffer=nl.sbuf, name="lmh_ring")
    logit_sb = sbm.alloc((P, n_tiles), dtype=nl.float32, buffer=nl.sbuf, name="lmh_logits_f32")
    for b in range(n_blocks):
        w_sb = ring[:, b % LMH_RING, :]
        nisa.dma_copy(dst=w_sb, src=W_tiled[prg_id, b], dge_mode=nisa.dge_mode.hwdge,
                      engine=nisa.engine.sync if b % 2 == 0 else nisa.engine.scalar)
        for t in range(nt):
            # one PSUM tile per vocab tile: with four accumulation groups sharing one [P, 4] PSUM tile,
            # tiles 0..2 of every block lost one of their 32 k-contributions (16% error; tile 3 exact)
            ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
            for f in range(H1):
                nisa.nc_matmul(dst=ps, stationary=w_sb[:, f * nb + t * P:f * nb + (t + 1) * P],
                               moving=x[:, f:f + 1], accumulate=(f > 0))
            nisa.tensor_copy(dst=logit_sb[:, b * nt + t:b * nt + t + 1], src=ps)

    col0 = prg_id * n_blocks * nb
    pad_from = V - col0                      # first padded vocab column in this core's shard (if any)
    if pad_from < n_blocks * nb:
        nisa.memset(dst=logit_sb[:, pad_from // P:n_tiles], value=-30000.0)

    logits = nl.ndarray((1, Vp), dtype=hidden.dtype, buffer=nl.shared_hbm)
    out_sb = sbm.alloc((P, n_tiles), dtype=hidden.dtype, buffer=nl.sbuf, name="lmh_logits")
    nisa.tensor_copy(dst=out_sb, src=logit_sb)
    # SBUF [p, tile] holds vocab col0 + tile * 128 + p
    nisa.dma_copy(dst=logits.ap(pattern=[[1, P], [P, n_tiles]], offset=col0), src=out_sb)
    sbm.close_scope()
    if n_prgs > 1:
        nisa.core_barrier(data=logits, cores=(0, 1))
    cfg = CascadedMaxConfig(logits.shape, inp_dtype=logits.dtype)
    _, idx_sb = cascaded_max_core(logits.reshape(cfg.inp_shape), cfg)
    token = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=token, src=idx_sb[:, 0:1])
    return logits, token


def qwen3_head_parts_tiled(token_ids, embed, hidden, g_final, W_tiled, V: int, eps: float = 1e-6):
    """Test harness for lm_head_tiled_argmax (with the embedding gather, like qwen3_head_parts)."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_head_parts_tiled", (0, 1), 2)
    X = embed_lookup(token_ids, embed, prg_id, n_prgs)
    logits, token = lm_head_tiled_argmax(hidden, g_final, W_tiled, V, prg_id, n_prgs, eps)
    return X, logits, token


# STATUS (2026-10-10): runs at a fixed position (2 layers: 3.71 vs 4.28 ms with nkilib's head) but hits an
# out-of-bounds indirect DMA (nrta 1006) on the second token of kernels/generate.py --tiled-head. Unresolved,
# so NOT the final kernel; see ATTEMPTS.md #18.
def qwen3_decode_step_tiled(
    token_ids, pos_ids, embed,
    W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache,
    cos_table, sin_table, g_final,
    W_tiled,      # [lnc, n_blocks, 128, H1 * LMH_NB] from pack_lm_head_tiled
    num_layers: int,
    S_ctx: int,
    V: int,
    eps: float = 1e-6,
):
    """qwen3_decode_step_resident with our own LM head (lm_head_tiled_argmax). Logits are [1, Vp] with the
    padded vocab entries at -30000. Returns (next_token [1, 1], next_pos [1, 1], logits [1, Vp])."""
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_decode_step_tiled", (0, 1), 2)
    half_d = cos_table.shape[1]
    q_heads = W_out.shape[1] // (2 * half_d)
    X = embed_lookup(token_ids, embed, prg_id, n_prgs)
    mask = build_decode_mask(pos_ids, S_ctx, q_heads, prg_id, n_prgs)
    cos = embed_lookup(pos_ids, cos_table, prg_id, n_prgs).reshape((half_d, 1, 1))
    sin = embed_lookup(pos_ids, sin_table, prg_id, n_prgs).reshape((half_d, 1, 1))
    hidden = qwen3_decode_layers(X, W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache,
                                 cos, sin, mask, pos_ids, num_layers=num_layers, eps=eps)
    logits, token = lm_head_tiled_argmax(hidden, g_final, W_tiled, V, prg_id, n_prgs, eps)

    pos_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_sb, src=pos_ids[0:1, 0:1])
    next_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=next_sb, data=pos_sb, op0=nl.add, operand0=1)
    next_pos = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=next_pos, src=next_sb)
    return token, next_pos, logits
