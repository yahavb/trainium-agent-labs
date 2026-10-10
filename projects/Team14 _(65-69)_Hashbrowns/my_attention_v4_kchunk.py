"""
V4 of the level-8 attention kernel: V1 (my_attention.py) with the back half -- transpose P, copy it
out of PSUM, multiply by V -- done in chunks of 32 keys, accumulating P @ V in PSUM.

NOT YET VERIFIED, in the simulator or on the device:

    python verify_sdk.py my_attention_v4_kchunk.py
    python bench_device.py

This is the "smaller chunks into SBUF" experiment, built to be measured rather than assumed. It is
exact: the row max and row sum are already complete before the chunks start, and P @ V is a sum over
keys, which PSUM accumulation computes chunk by chunk. No online-softmax rescaling is needed.

Why it is expected to LOSE, and what would prove that wrong:
  * each chunk's pT is [32 keys, n queries]: n values per lane on 32 lanes, so each copy streams as
    long as V1's single [n, n] copy did, and there are n/32 of them;
  * the loop adds a Tensor transpose, a copy and a matmul per chunk, each paying a fixed cost.
The case for it is overlap: the copy of chunk i on Vector running while the Tensor engine works on
chunk i+1. Instruction scheduling is off by default in nki 0.6.0, so each engine runs in program
order (T0, mm0, T1, mm1, ...) and mm0 waits for copy0 before T1 can start -- run this with +opt in
bench_device.py as well, which lets the scheduler reorder it into a pipeline.

Chunks are 32 keys when n is a multiple of 32 (all three level-8 shapes); otherwise one chunk of n,
which is V1's back half again.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
    n, d = q.shape
    assert n <= 128 and d <= 128, f"single-tile kernel: needs seq <= 128 and dim <= 128, got {n}, {d}"
    scale = 1.0 / (d ** 0.5)
    kc = 32 if n % 32 == 0 else n
    chunks = n // kc

    out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # 1. HBM -> SBUF: rows on the partition axis
    q_sb = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    k_sb = nl.ndarray((n, d), dtype=k.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)

    # 2. Transpose Q and K so d is on the partition axis; scale folded into Q on the Scalar engine
    qT_ps = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.psum)
    kT_ps = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=qT_ps, data=q_sb)
    nisa.nc_transpose(dst=kT_ps, data=k_sb)
    qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
    kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)
    nisa.tensor_copy(dst=kT, src=kT_ps)

    # 3. S = (scale * Q) @ K.T -> PSUM [n, n], float32, already scaled
    s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

    # 4. Softmax along each row, whole rows: the chunks below need the final max and sum
    neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
    p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
                           reduce_op=nl.add, reduce_res=row_sum)

    # 5. O = P @ V, one chunk of keys at a time, summed in PSUM. Every tile here starts at
    # partition 0: V is loaded per chunk rather than sliced, because slicing v_sb by keys would put
    # the matmul's moving operand at partition 32, 64, ...
    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    for c in nl.affine_range(chunks):
        v_c = nl.ndarray((kc, d), dtype=v.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_c, src=v[c * kc:(c + 1) * kc, :])
        pT_ps = nl.ndarray((kc, n), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=pT_ps, data=p[:, c * kc:(c + 1) * kc])
        pT = nl.ndarray((kc, n), dtype=v.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pT, src=pT_ps)
        nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_c)

    # 6. Normalise while copying out of PSUM, then SBUF -> HBM
    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)
    o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
    nisa.dma_copy(dst=out, src=o_sb)
    return out
