"""Qwen3 batch=1 decode megakernel: N decoder layers in ONE NKI kernel launch.

Built on AWS's experimental `nkilib.experimental.transformer.transformer_tkg`, which already
fuses RMSNorm + QKV + RoPE + attention + o_proj (attention_block_tkg) and RMSNorm + gate/up +
SiLU + down (mlp) per layer. Two things it does not do that Qwen3 needs, and this file adds:

  1. Qwen3's per-head QK-norm before RoPE (rmsnorm_QK_pre_rope_* wired to q_norm / k_norm gammas).
  2. Stacked per-layer weights with a leading L axis instead of Python lists. In NKI's standalone
     NumPy path a Python list argument means "one tensor per rank" (multi-core execution), so the
     list-of-tensors signature of transformer_tkg cannot be called from the host at all.

Layouts are documented in qwen3_ref.py. The kernel runs on one logical NeuronCore with LNC=2
(`kernel[2](...)`): the two physical cores shard the hidden dimension inside each block, exactly as
transformer_tkg does. Tensor parallelism across logical cores (collectives) is deliberately out of
scope for this first version.
"""
import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.mlp.mlp import mlp
from nkilib.core.utils.allocator import BufferManager
from nkilib.core.utils.common_types import ActFnType, NormType, QuantizationType
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.logging import get_logger
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
from nkilib.experimental.transformer.transformer_tkg import _load_input_to_sbuf, _store_output_to_hbm

_SBM_SIZE_BYTES = 200 * 1024  # same SBUF budget transformer_tkg uses


def qwen3_decode_layers(
    X,          # [B, S_tkg, H] bf16 @ HBM
    W_qkv,      # [L, H, (q+2kv)*d]
    W_out,      # [L, q*d, H]
    W_gate,     # [L, H, I]
    W_up,       # [L, H, I]
    W_down,     # [L, I, H]
    g_attn,     # [L, 1, H]
    g_mlp,      # [L, 1, H]
    g_q,        # [L, 1, d]
    g_k,        # [L, 1, d]
    K_cache,    # [L, B, kv, d, S_max]   (transposed flat cache)
    V_cache,    # [L, B, kv, S_max, d]
    cos,        # [d//2, B, S_tkg]
    sin,        # [d//2, B, S_tkg]
    mask,       # [S_ctx, B, q, S_tkg] uint8
    pos_ids,    # [B, 1] uint32 cache write position
    num_layers: int,
    eps: float = 1e-6,
    mlp_gate_up_column_tiling: bool = False,  # weights-stationary gate/up: 6% faster per layer than nkilib default (profiled)
    mlp_down_column_tiling: bool = False,
):
    B, S_tkg, H = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_decode_layers", (0, 1), 2)

    sbm = BufferManager(0, _SBM_SIZE_BYTES, get_logger("qwen3_decode_layers"))
    sbm.set_auto_alloc(False)

    current = X
    for l in range(num_layers):
        # ---- attention block: RMSNorm -> QKV -> QK-norm -> RoPE -> attention(+cache update) -> o_proj
        sbm.set_name_prefix(f"L{l}_attn_")
        sbm.set_auto_alloc(True)
        attn_result = attention_block_tkg(
            X=current,
            X_hidden_dim_actual=H,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=eps,
            rmsnorm_X_gamma=g_attn[l],
            W_qkv=W_qkv[l],
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,          # Qwen3 QK-norm
            rmsnorm_QK_pre_rope_eps=eps,
            rmsnorm_QK_pre_rope_W_Q=g_q[l],
            rmsnorm_QK_pre_rope_W_K=g_k[l],
            cos=cos,
            sin=sin,
            rope_contiguous_layout=True,               # HF rotate_half halves
            rmsnorm_QK_post_rope_enabled=False,
            rmsnorm_QK_post_rope_eps=eps,
            rmsnorm_QK_post_rope_W_Q=None,
            rmsnorm_QK_post_rope_W_K=None,
            K_cache_transposed=True,
            active_blocks_table=None,
            K_cache=K_cache[l],
            V_cache=V_cache[l],
            attention_mask=mask,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=pos_ids,
            W_out=W_out[l],
            bias_out=None,
            quantization_type_out=QuantizationType.NONE,
            weight_dequant_scale_out=None,
            input_dequant_scale_out=None,
            transposed_out=False,
            out_in_sb=False,
            sbm=sbm,
        )
        attn_out = attn_result[0]
        while sbm.heap:
            sbm.pop_heap()
        if n_prgs > 1:
            nisa.core_barrier(data=attn_out, cores=(0, 1))

        # ---- residual 1 (HBM round trip, as in transformer_tkg's HBM path)
        attn_residual = sbm.alloc(current.shape, dtype=dtype, buffer=nl.shared_hbm, name="attn_residual")
        cur_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
        attn_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
        res_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cur_sb, src=current.reshape((BxS, H)))
        nisa.dma_copy(dst=attn_sb, src=attn_out.reshape((BxS, H)))
        nisa.tensor_tensor(dst=res_sb, data1=cur_sb, data2=attn_sb, op=nl.add)
        nisa.dma_copy(dst=attn_residual.reshape((BxS, H)), src=res_sb)

        # ---- MLP block: RMSNorm -> gate/up -> SiLU * up -> down
        sbm.set_name_prefix(f"L{l}_mlp_")
        sbm.set_auto_alloc(False)
        mlp_outputs = mlp(
            hidden_tensor=attn_residual,
            gate_proj_weights_tensor=W_gate[l],
            up_proj_weights_tensor=W_up[l],
            down_proj_weights_tensor=W_down[l],
            normalization_weights_tensor=g_mlp[l],
            normalization_type=NormType.RMS_NORM,
            activation_fn=ActFnType.SiLU,
            eps=eps,
            quantization_type=QuantizationType.NONE,
            use_tkg_gate_up_proj_column_tiling=mlp_gate_up_column_tiling,
            use_tkg_down_proj_column_tiling=mlp_down_column_tiling,
            sbm=sbm,
        )
        mlp_out = mlp_outputs[0]
        if n_prgs > 1:
            nisa.core_barrier(data=mlp_out, cores=(0, 1))

        # ---- residual 2
        layer_output = sbm.alloc(attn_residual.shape, dtype=dtype, buffer=nl.shared_hbm, name="layer_output")
        a_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
        m_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
        o_sb = nl.ndarray((BxS, H), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_sb, src=attn_residual.reshape((BxS, H)))
        nisa.dma_copy(dst=m_sb, src=mlp_out.reshape((BxS, H)))
        nisa.tensor_tensor(dst=o_sb, data1=a_sb, data2=m_sb, op=nl.add)
        nisa.dma_copy(dst=layer_output.reshape((BxS, H)), src=o_sb)

        current = layer_output

    return current


def _gather_shards_sb(sharded_sb, dtype, prg_id: int, n_prgs: int, H0: int, H1: int, H1_shard: int, BxS: int):
    """[H0, H1_shard*BxS] per-core shard -> [H0, BxS*H1] full hidden on both cores, SBUF to SBUF.

    transformer_tkg's _sb2sb_all_reduce_gather minus the all-reduce: with one logical core there are
    no tensor-parallel partial sums, only the two physical cores' H shards to exchange.
    """
    gathered_sb = nl.ndarray((H0, H1 * BxS), dtype=dtype, buffer=nl.sbuf)
    f_shard = nl.ds(start=prg_id * BxS * H1_shard, size=BxS * H1_shard)
    nisa.tensor_copy(dst=gathered_sb[:, f_shard], src=sharded_sb)
    if n_prgs > 1:
        other = 1 - prg_id
        f_other = nl.ds(start=other * BxS * H1_shard, size=BxS * H1_shard)
        nisa.sendrecv(src=sharded_sb, dst=gathered_sb[:, f_other], send_to_rank=other, recv_from_rank=other, pipe_id=0)
    out_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
    src_view = gathered_sb.rearrange(('h0', ('h1', 'bs')), ('h0', 'bs', 'h1'), {'h1': H1})
    nisa.tensor_copy(dst=out_sb.reshape((H0, BxS, H1)), src=src_view)
    return out_sb


def qwen3_decode_layers_sbuf(
    X, W_qkv, W_out, W_gate, W_up, W_down, g_attn, g_mlp, g_q, g_k, K_cache, V_cache, cos, sin, mask, pos_ids,
    num_layers: int,
    eps: float = 1e-6,
    mlp_gate_up_column_tiling: bool = False,
    mlp_down_column_tiling: bool = False,
):
    """Same contract as qwen3_decode_layers, with the residual stream kept in SBUF.

    transformer_tkg's sbuf_residual_and_cc path (attention with transposed_out + out_in_sb, MLP with
    store_output_in_sbuf, SBUF-to-SBUF shard exchange between the two physical cores), plus one change:
    the library still stores every layer's output to HBM and reloads it; here the hidden state stays in
    SBUF from layer to layer and only the last layer writes HBM.
    """
    B, S_tkg, H = X.shape
    dtype = X.dtype
    BxS = B * S_tkg
    _, n_prgs, prg_id = get_verified_program_sharding_info("qwen3_decode_layers_sbuf", (0, 1), 2)
    H0 = nl.tile_size.pmax
    H1 = H // H0
    H1_shard = H1 // n_prgs

    sbm = BufferManager(0, _SBM_SIZE_BYTES, get_logger("qwen3_decode_layers_sbuf"))
    sbm.set_auto_alloc(False)

    hidden_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
    _load_input_to_sbuf(hidden_sb, X, BxS, H0, H1, H1_shard, n_prgs)
    layer_output = None
    for l in range(num_layers):
        sbm.set_name_prefix(f"L{l}_attn_")
        residual_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=residual_sb, src=hidden_sb)
        sbm.set_auto_alloc(True)
        attn_result = attention_block_tkg(
            X=hidden_sb.reshape((H0, BxS, H1)),
            X_hidden_dim_actual=H,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=eps,
            rmsnorm_X_gamma=g_attn[l],
            W_qkv=W_qkv[l],
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None,
            input_dequant_scale_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,          # Qwen3 QK-norm
            rmsnorm_QK_pre_rope_eps=eps,
            rmsnorm_QK_pre_rope_W_Q=g_q[l],
            rmsnorm_QK_pre_rope_W_K=g_k[l],
            cos=cos,
            sin=sin,
            rope_contiguous_layout=True,
            rmsnorm_QK_post_rope_enabled=False,
            rmsnorm_QK_post_rope_eps=eps,
            rmsnorm_QK_post_rope_W_Q=None,
            rmsnorm_QK_post_rope_W_K=None,
            K_cache_transposed=True,
            active_blocks_table=None,
            K_cache=K_cache[l],
            V_cache=V_cache[l],
            attention_mask=mask,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=pos_ids,
            W_out=W_out[l],
            bias_out=None,
            quantization_type_out=QuantizationType.NONE,
            weight_dequant_scale_out=None,
            input_dequant_scale_out=None,
            transposed_out=True,
            out_in_sb=True,
            sbm=sbm,
        )
        while sbm.heap:
            sbm.pop_heap()
        attn_sb = _gather_shards_sb(attn_result[0].reshape((H0, H1_shard * BxS)), dtype, prg_id, n_prgs,
                                    H0, H1, H1_shard, BxS)

        mlp_in_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mlp_in_sb, data1=residual_sb, data2=attn_sb, op=nl.add)
        residual2_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=residual2_sb, src=mlp_in_sb)

        sbm.set_name_prefix(f"L{l}_mlp_")
        sbm.set_auto_alloc(False)
        mlp_outputs = mlp(
            hidden_tensor=mlp_in_sb.reshape((H0, BxS, H1)),
            gate_proj_weights_tensor=W_gate[l],
            up_proj_weights_tensor=W_up[l],
            down_proj_weights_tensor=W_down[l],
            normalization_weights_tensor=g_mlp[l],
            normalization_type=NormType.RMS_NORM,
            activation_fn=ActFnType.SiLU,
            eps=eps,
            quantization_type=QuantizationType.NONE,
            store_output_in_sbuf=True,
            use_tkg_gate_up_proj_column_tiling=mlp_gate_up_column_tiling,
            use_tkg_down_proj_column_tiling=mlp_down_column_tiling,
            sbm=sbm,
        )
        mlp_sb = _gather_shards_sb(mlp_outputs[0].reshape((H0, H1_shard * BxS)), dtype, prg_id, n_prgs,
                                   H0, H1, H1_shard, BxS)
        hidden_sb = nl.ndarray((H0, BxS * H1), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hidden_sb, data1=residual2_sb, data2=mlp_sb, op=nl.add)

    layer_output = sbm.alloc((B, S_tkg, H), dtype=dtype, buffer=nl.shared_hbm, name="layer_output")
    if prg_id == 0:
        _store_output_to_hbm(layer_output, hidden_sb, BxS, H0, H1, H1_shard, n_prgs)
    if n_prgs > 1:
        nisa.core_barrier(data=layer_output, cores=(0, 1))
    return layer_output


qwen3_decode_layers_jit = nki.jit(qwen3_decode_layers)
qwen3_decode_layers_sbuf_jit = nki.jit(qwen3_decode_layers_sbuf)
