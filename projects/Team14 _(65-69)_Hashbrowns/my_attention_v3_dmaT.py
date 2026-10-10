"""
V3 of the level-8 attention kernel: V1 (my_attention.py) with Q and K transposed by the DMA engine on
their way in from HBM, instead of by the Tensor engine after they arrive.

NOT YET VERIFIED, in the simulator or on the device:

    python verify_sdk.py my_attention_v3_dmaT.py
    python bench_device.py

What it removes from the critical path: both Tensor-engine transposes of Q and K, and both PSUM -> SBUF
copies after them -- four instructions and two engine handoffs at the front of the chain. The load
goes straight into the [d, n] layout nc_matmul contracts over.

What it gives back: V1 folded the softmax scale into the qT copy, and that copy no longer exists. The
bias is -scale * max again, one [n, 1] multiply after the max reduction; the exp takes scale=scale.

The open question is the DMA itself. A transposing load reads 4-byte elements with a stride, which
is far less efficient per byte than a contiguous load. Whether that costs more than the four
instructions it replaces is what the benchmark measures. dma_transpose may also be limited to 2-byte
dtypes on this hardware, and these inputs are float32: if so it fails to compile, and that is the
result.
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

    # 1. HBM -> SBUF. Q and K land transposed, d on the partition axis, ready for nc_matmul.
    qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
    kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
    v_sb = nl.ndarray((n, d), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_transpose(dst=qT, src=q)                                          # VERIFY fp32
    nisa.dma_transpose(dst=kT, src=k)                                          # VERIFY fp32
    nisa.dma_copy(dst=v_sb, src=v)

    # 2. S = Q @ K.T -> PSUM [n, n], float32, unscaled
    s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

    # 3. Softmax along each row, in float32. -max straight from the reduction, then * scale for
    # the exp bias: exp(scale * S - scale * max).
    neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
    bias = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bias, data=neg_max, op0=nl.multiply, operand0=scale)

    p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=bias, scale=scale,
                           reduce_op=nl.add, reduce_res=row_sum)

    # 4. Transpose P so keys are on the partition axis: P @ V contracts over keys
    pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)
    pT = nl.ndarray((n, n), dtype=v.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps)

    # 1 / row_sum, issued after the pT copy so it does not delay P @ V on the Vector queue
    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)

    # 5. O = P @ V -> PSUM [n, d], then normalise while copying out of PSUM
    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
    o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

    # 6. SBUF -> HBM
    nisa.dma_copy(dst=out, src=o_sb)
    return out
