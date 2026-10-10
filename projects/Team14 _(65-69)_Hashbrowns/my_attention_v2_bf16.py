"""
V2 of the level-8 attention kernel: the V1 schedule (my_attention.py) with the P @ V half in
bfloat16. Same interface, same float32 inputs and output.

NOT YET VERIFIED, in the simulator or on the device:

    python verify_sdk.py my_attention_v2_bf16.py
    python bench_device.py

Why: the inputs are float32, so in V1 the P transpose, its PSUM copy and P @ V all go through the
Tensor engine's float32 path, which runs below its bf16 rate. P is in [0, 1] and V is plain data, so
rounding both to bf16 costs about 3 significant digits there -- well inside nkibench's 2e-2 of the
output RMS. Q @ K.T stays float32: the large-value scores (~900) would shift by whole units in bf16,
and that error goes through exp().

Only a device run says whether this is faster. At this size the matmuls may be too short for their
rate to matter at all.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
    n, d = q.shape
    assert n <= 128 and d <= 128, f"single-tile kernel: needs seq <= 128 and dim <= 128, got {n}, {d}"
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
    # Scale folded into Q on its way out of PSUM, on the Scalar engine; kT on Vector in parallel.
    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)               # VERIFY nl.copy
    nisa.tensor_copy(dst=kT, src=kT_ps)

    # 3. S = (scale * Q) @ K.T -> PSUM [n, n], float32, already scaled
    s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

    # 4. Softmax along each row. negate=True returns -max, the exp bias as it stands.
    neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)  # VERIFY negate

    # V to bf16 for P @ V. Issued here, not after the load, so it does not sit ahead of the kT copy
    # in Vector's queue: Vector is idle at this point while exp runs on the Scalar engine.
    v_bf = nl.ndarray((n, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_bf, src=v_sb)

    # p = exp(S - max) written straight out as bf16; the row sum is still accumulated in float32.
    p = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,         # VERIFY
                           reduce_op=nl.add, reduce_res=row_sum)

    # 5. Transpose P in bf16: a Tensor-engine transpose keeps its input's dtype
    pT_ps = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)                                       # VERIFY bf16 PSUM
    pT = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps)

    # 1 / row_sum, issued after the pT copy so it does not delay P @ V on the Vector queue.
    # The device compiler has no divide in tensor_scalar, so step 7 multiplies by this.
    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)

    # 6. O = P @ V with bf16 operands -> PSUM [n, d], float32, not yet normalised
    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_bf)

    # 7. Normalise last: n*d multiplies instead of n*n
    o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

    # 8. SBUF -> HBM
    nisa.dma_copy(dst=out, src=o_sb)
    return out
