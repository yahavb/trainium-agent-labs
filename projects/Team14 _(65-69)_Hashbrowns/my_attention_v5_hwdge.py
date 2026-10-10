"""
V5 of the level-8 attention kernel: the V1 schedule (my_attention.py) with every DMA built by the
hardware descriptor generator. Same interface, same float32 inputs and output, same algorithm.
Written and measured on seat-68 as "V3 hwdge" (results/seat68/attention); renumbered V5 here because
this folder's V3 is my_attention_v3_dmaT.py.

    out = softmax(Q @ K.T / sqrt(d)) @ V        q, k, v: [n, d], n <= 128, d <= 128

MEASURED on seat-68, nki 0.6.0, LNC 1, 2000 timed iterations, three runs: 1.114x, 1.109x and 1.108x
faster than V1, about 1.7 us saved on every level-8 shape (17.69 -> 16.06 us at seq 128 dim 64).
GpSimd busy fell from 24-26% to 13-15%. Correct on the device, 3/3 in the simulator. The bf16 half
of V2 stacks on top of this: see my_attention_v6_hwdge_bf16.py.

    python nkibench.py --level 8 --check my_attention_v5_hwdge.py
    python bench_device.py my_attention.py my_attention_v5_hwdge.py --all-shapes     # device timing against V1

Why: V1's profile shows GpSimd busy 24% of the kernel although the source never calls it. That is
software descriptor generation -- dma_copy defaults to swdge, where GpSimd builds every DMA's
descriptors at run time, and all four transfers sit in one queue (qPoolDynamic), which runs in order.
The Trainium2 arch guide recommends hardware DGE wherever possible: the Sync or Scalar sequencer
asks a DGE hardware block for the descriptors, ~600 ns per dma_copy, and GpSimd stays free.

  * Q and K are triggered from different engines, Sync and Scalar, so they sit in different queues
    and can load at the same time. V goes behind Q on Sync: it is not needed until P @ V. The store
    goes on Sync, which is idle by then.
  * hwdge needs src and dst of the same dtype. All four transfers are float32 -> float32 here, which
    is why this builds on V1 rather than on V2 (whose bf16 cast could otherwise ride the DMA).
  * Every compute instruction is unchanged. The engine= pins on the two copies only fix the split the
    compiler already chose for V1, so the DMAs are the only real difference from V1.
"""

import nki
import nki.isa as nisa
import nki.language as nl

HW = nisa.dge_mode.hwdge


@nki.jit
def nki_attention_(q, k, v):
    n, d = q.shape
    assert n <= 128 and d <= 128, f"single-tile kernel: needs seq <= 128 and dim <= 128, got {n}, {d}"
    scale = 1.0 / (d ** 0.5)

    out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # 1. HBM -> SBUF: rows on the partition axis. Q and K on different trigger engines, so different
    # DMA queues; V queues behind Q because nothing needs it until step 6.
    q_sb = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
    k_sb = nl.ndarray((n, d), dtype=k.dtype, buffer=nl.sbuf)
    v_sb = nl.ndarray((n, d), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q, dge_mode=HW, engine=nisa.engine.sync)
    nisa.dma_copy(dst=k_sb, src=k, dge_mode=HW, engine=nisa.engine.scalar)
    nisa.dma_copy(dst=v_sb, src=v, dge_mode=HW, engine=nisa.engine.sync)

    # 2. Transpose Q and K so d is on the partition axis: nc_matmul contracts over partitions
    qT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
    kT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=qT_ps, data=q_sb)
    nisa.nc_transpose(dst=kT_ps, data=k_sb)
    qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
    kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
    # qT leaves PSUM on Scalar with the softmax scale folded in; kT on Vector, in parallel.
    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)
    nisa.tensor_copy(dst=kT, src=kT_ps, engine=nisa.engine.vector)

    # 3. S = (scale * Q) @ K.T -> PSUM [n, n], float32, already scaled
    s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

    # 4. Softmax along each row, in float32. negate=True returns -max, the exp bias as it stands.
    neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)

    # p = exp(S - max), with the row sum accumulated by the same instruction on the Scalar engine.
    p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
                           reduce_op=nl.add, reduce_res=row_sum)

    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys
    pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)
    pT = nl.ndarray((n, n), dtype=v.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps, engine=nisa.engine.vector)

    # 6. O = P @ V -> PSUM [n, d], float32, not yet normalised
    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)

    # 7. Normalise last: n reciprocals, then n*d multiplies (tensor_scalar has no divide)
    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)
    o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

    # 8. SBUF -> HBM, triggered from Sync, which is idle by now
    nisa.dma_copy(dst=out, src=o_sb, dge_mode=HW, engine=nisa.engine.sync)
    return out
