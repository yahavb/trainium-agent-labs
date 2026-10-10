"""
Multi-head attention, baseline M6: the V6 single-head kernel (my_attention_v6_hwdge_bf16.py) run once
per head inside one kernel call. nkibench level 9. Measured on seat-68 as "M4 V4 per head" (results/
seat68/mha), from when V6 was called V4.

    out[:, h, :] = softmax(q[:, h, :] @ k[:, h, :].T / sqrt(d)) @ v[:, h, :]    q, k, v: [S, H, d]

MEASURED on seat-68: misses nkibench's tolerance at the two seq-128 16-head shapes (worst error 0.0245
of the output RMS against 0.02, from bf16 P @ V), and at 32 heads it is SLOWER than the V0 loop, 128.3
against 125.3 us: 128 small DMAs, each triggered from Sync or Scalar at ~0.6 us, leave Sync 82 us busy.
Hardware descriptors only pay when each DMA moves enough -- the reason mha_fast.py loads in chunks.

V6's two changes carry over unchanged -- hardware DMA descriptors (Q on Sync, K on Scalar, V and the
store on Sync) and bf16 for the P @ V half -- but the structure is still one head at a time on one
core: a small DMA per head per tensor, V cast to bf16 per head, every softmax step per head.

    python nkibench.py --level 9 --check mha_v6_loop.py
    python bench_suites.py mha
"""

import nki
import nki.isa as nisa
import nki.language as nl

HW = nisa.dge_mode.hwdge


@nki.jit
def nki_mha_(q, k, v):
    S, H, d = q.shape
    assert S <= 128 and d <= 128, f"single-tile heads: needs seq <= 128 and dim <= 128, got {S}, {d}"
    scale = 1.0 / (d ** 0.5)

    out = nl.ndarray((S, H, d), dtype=q.dtype, buffer=nl.shared_hbm)

    for h in nl.affine_range(H):
        q_sb = nl.ndarray((S, d), dtype=q.dtype, buffer=nl.sbuf)
        k_sb = nl.ndarray((S, d), dtype=k.dtype, buffer=nl.sbuf)
        v_sb = nl.ndarray((S, d), dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_sb, src=q.ap(pattern=[[H * d, S], [1, d]], offset=h * d),
                      dge_mode=HW, engine=nisa.engine.sync)
        nisa.dma_copy(dst=k_sb, src=k.ap(pattern=[[H * d, S], [1, d]], offset=h * d),
                      dge_mode=HW, engine=nisa.engine.scalar)
        nisa.dma_copy(dst=v_sb, src=v.ap(pattern=[[H * d, S], [1, d]], offset=h * d),
                      dge_mode=HW, engine=nisa.engine.sync)

        qT_ps = nl.ndarray((d, S), dtype=nl.float32, buffer=nl.psum)
        kT_ps = nl.ndarray((d, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qT_ps, data=q_sb)
        nisa.nc_transpose(dst=kT_ps, data=k_sb)
        qT = nl.ndarray((d, S), dtype=q.dtype, buffer=nl.sbuf)
        kT = nl.ndarray((d, S), dtype=k.dtype, buffer=nl.sbuf)
        nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)
        nisa.tensor_copy(dst=kT, src=kT_ps, engine=nisa.engine.vector)

        s_ps = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

        neg_max = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)

        v_bf = nl.ndarray((S, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_bf, src=v_sb, engine=nisa.engine.vector)

        p = nl.ndarray((S, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        row_sum = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
                               reduce_op=nl.add, reduce_res=row_sum)

        pT_ps = nl.ndarray((S, S), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=pT_ps, data=p)
        pT = nl.ndarray((S, S), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pT, src=pT_ps, engine=nisa.engine.vector)

        o_ps = nl.ndarray((S, d), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_bf)

        inv_sum = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sum, data=row_sum)
        o_sb = nl.ndarray((S, d), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

        nisa.dma_copy(dst=out.ap(pattern=[[H * d, S], [1, d]], offset=h * d), src=o_sb,
                      dge_mode=HW, engine=nisa.engine.sync)
    return out
