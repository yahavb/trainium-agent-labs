"""
V6 of the level-8 attention kernel, and the fastest single-head version measured with the device
benchmark: V5's hardware-DGE DMAs (my_attention_v5_hwdge.py) plus V2's bfloat16 P @ V half
(my_attention_v2_bf16.py). Same interface, same float32 inputs and output. Written and measured on
seat-68 as "V4 hwdge+bf16" (results/seat68/attention); renumbered V6 here because this folder's V4 is
my_attention_v4_kchunk.py.

    out = softmax(Q @ K.T / sqrt(d)) @ V        q, k, v: [n, d], n <= 128, d <= 128

MEASURED on seat-68, nki 0.6.0, LNC 1, 2000 timed iterations, two runs: 1.127x and 1.129x faster than
V1. Per shape, run 1 -> run 2: seq 128 dim 64 1.122x -> 1.130x (17.69 -> 15.76 us), seq 64 dim 128
1.145x -> 1.144x, seq 96 dim 32 1.115x -> 1.113x. Correct on all three on the device, and 3/3 in the
simulator. Single-tile shapes only -- no number exists for anything larger. Its bf16 P @ V does
not scale: run once per head over 16 heads (mha_v6_loop.py) it misses nkibench's tolerance.

    python nkibench.py --level 8 --check my_attention_v6_hwdge_bf16.py
    python bench_device.py my_attention.py my_attention_v6_hwdge_bf16.py --all-shapes     # device timing against V1

The two changes touch different parts of the kernel, and they mostly stack: V3 changed only how the
four DMAs get their descriptors (V5, 1.109x over V1), V2 only the dtype of P, its transpose and P @ V
(1.025x); multiplied, 1.137x, against 1.128x measured. At dim 32 bf16 adds nothing over V3 (15.19 vs
15.14 us, within noise): with so little P @ V work, its rate does not matter. V is still loaded as float32 and cast on the Vector engine, not by the DMA:
hwdge requires src and dst of the same dtype. Q @ K.T stays float32, for the reason V2 gives -- the
large-value scores would shift by whole units in bf16, and that error goes through exp().
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

    # V to bf16 for P @ V, on Vector, which is idle while exp runs on Scalar. Not on the load: hwdge
    # needs matching dtypes.
    v_bf = nl.ndarray((n, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=v_bf, src=v_sb, engine=nisa.engine.vector)

    # p = exp(S - max) written straight out as bf16; the row sum is still accumulated in float32.
    p = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
                           reduce_op=nl.add, reduce_res=row_sum)

    # 5. Transpose P in bf16 so keys are on the partition axis: a Tensor-engine transpose keeps its
    # input's dtype
    pT_ps = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)
    pT = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps, engine=nisa.engine.vector)

    # 6. O = P @ V with bf16 operands -> PSUM [n, d], float32, not yet normalised
    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_bf)

    # 7. Normalise last: n reciprocals, then n*d multiplies (tensor_scalar has no divide)
    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)
    o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

    # 8. SBUF -> HBM, triggered from Sync, which is idle by now
    nisa.dma_copy(dst=out, src=o_sb, dge_mode=HW, engine=nisa.engine.sync)
    return out
