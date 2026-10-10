"""
Single-tile attention kernel for level 8 of the ladder in nkibench.py:

    out = softmax(Q @ K.T / sqrt(d)) @ V        q, k, v: [n, d], n <= 128, d <= 128

V0, THE BASELINE: the kernel the 17-instruction simulator trace came from, with one change. As traced,
it ended in tensor_scalar(op0=nl.divide). The simulator accepts that, but the device compiler rejects
it ("invalid kwarg 'op0': unsupported operator 'divide'", nki 0.6.0 on seat-65), so the traced kernel
never compiled for a NeuronCore. Here the last step takes the reciprocal of the row sums and
multiplies: one [n, 1] reciprocal more, issued where the divide was. Everything else is as traced.

The traced version passed verify_sdk.py in the simulator (8 cases, 1.00x the HBM byte floor), which
is how the divide went unnoticed. Not yet run on a NeuronCore.

Lines marked VERIFY use instructions none of the shipped reference kernels call; verify_sdk.py
confirmed each one exists with these keywords.

    python verify_sdk.py
    python nkibench.py --level 8 --check my_attention.py

Data path: Q, K, V come in from HBM once and O goes out once. The [n, n] score and weight
matrices never leave the chip, which is what keeps fused attention compute bound.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
    n, d = q.shape
    scale = 1.0 / (d ** 0.5)

    out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # 1. HBM -> SBUF: rows on the partition axis
    q_sb = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    k_sb = nl.ndarray((n, d), dtype=k.dtype, buffer=nl.sbuf)
    v_sb = nl.ndarray((n, d), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)
    nisa.dma_copy(dst=v_sb, src=v)

    # 2. Transpose Q and K so d is on the partition axis: nc_matmul contracts over partitions
    qT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
    kT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=qT_ps, data=q_sb)                                    # VERIFY
    nisa.nc_transpose(dst=kT_ps, data=k_sb)                                    # VERIFY
    qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
    kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qT, src=qT_ps)
    nisa.tensor_copy(dst=kT, src=kT_ps)

    # 3. S = Q @ K.T -> PSUM [n, n], float32
    s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

    # 4. Softmax along each row, in float32. Subtracting the row max keeps exp() from overflowing.
    row_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=s_ps, axis=1)          # VERIFY
    neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-scale)

    # p = exp(scale * S - scale * max): scale and max-subtraction folded into one instruction
    p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=p, op=nl.exp, data=s_ps, bias=neg_max, scale=scale)    # VERIFY

    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_sum, op=nl.add, data=p, axis=1)                 # VERIFY

    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys
    pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)                                       # VERIFY
    pT = nl.ndarray((n, n), dtype=v.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps)

    # 6. O = P @ V -> PSUM [n, d], float32, not yet normalised
    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)

    # 7. Normalise last: n*d multiplies instead of n*n divides. The device compiler has no divide in
    # tensor_scalar, so multiply by the reciprocal of the row sums.
    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)
    o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

    # 8. SBUF -> HBM
    nisa.dma_copy(dst=out, src=o_sb)
    return out
