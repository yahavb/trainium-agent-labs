"""
Multi-head attention, baseline M0: the V0 single-head algorithm (my_attention_v0.py), run once per head
inside one kernel call. nkibench level 9.

    out[:, h, :] = softmax(q[:, h, :] @ k[:, h, :].T / sqrt(d)) @ v[:, h, :]    q, k, v: [S, H, d]

MEASURED on seat-68: correct on all four level-9 shapes; 74.5, 76.1, 38.4 and 125.3 us. Its profile at
32 heads shows GpSimd busy 86 us building descriptors in software for 128 small DMAs.

Everything the fast kernel (mha_fast.py) changes is left as a straightforward port would have it:
one core, one small DMA per head per tensor (S rows of d elements, strided across the heads), default
descriptor generation, float32 throughout, and every softmax step issued per head.

    python nkibench.py --level 9 --check mha_v0_loop.py
    python bench_suites.py mha
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_mha_(q, k, v):
    S, H, d = q.shape
    assert S <= 128 and d <= 128, f"single-tile heads: needs seq <= 128 and dim <= 128, got {S}, {d}"
    scale = 1.0 / (d ** 0.5)

    out = nl.ndarray((S, H, d), dtype=q.dtype, buffer=nl.shared_hbm)

    for h in nl.affine_range(H):
        # One head's rows: S partitions, d contiguous elements each, H*d apart in HBM.
        q_sb = nl.ndarray((S, d), dtype=q.dtype, buffer=nl.sbuf)
        k_sb = nl.ndarray((S, d), dtype=k.dtype, buffer=nl.sbuf)
        v_sb = nl.ndarray((S, d), dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_sb, src=q.ap(pattern=[[H * d, S], [1, d]], offset=h * d))
        nisa.dma_copy(dst=k_sb, src=k.ap(pattern=[[H * d, S], [1, d]], offset=h * d))
        nisa.dma_copy(dst=v_sb, src=v.ap(pattern=[[H * d, S], [1, d]], offset=h * d))

        qT_ps = nl.ndarray((d, S), dtype=nl.float32, buffer=nl.psum)
        kT_ps = nl.ndarray((d, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qT_ps, data=q_sb)
        nisa.nc_transpose(dst=kT_ps, data=k_sb)
        qT = nl.ndarray((d, S), dtype=q.dtype, buffer=nl.sbuf)
        kT = nl.ndarray((d, S), dtype=k.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qT, src=qT_ps, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=kT, src=kT_ps, engine=nisa.engine.vector)

        s_ps = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

        row_max = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=s_ps, axis=1)
        exp_bias = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=exp_bias, data=row_max, op0=nl.multiply, operand0=-scale)
        p = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=p, op=nl.exp, data=s_ps, scale=scale, bias=exp_bias)
        row_sum = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_sum, op=nl.add, data=p, axis=1)

        pT_ps = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=pT_ps, data=p)
        pT = nl.ndarray((S, S), dtype=v.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pT, src=pT_ps, engine=nisa.engine.vector)

        o_ps = nl.ndarray((S, d), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)

        inv_sum = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sum, data=row_sum)
        o_sb = nl.ndarray((S, d), dtype=out.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

        nisa.dma_copy(dst=out.ap(pattern=[[H * d, S], [1, d]], offset=h * d), src=o_sb)
    return out
