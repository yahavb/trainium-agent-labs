"""
Single-tile attention kernel for level 8 of the ladder in nkibench.py:

    out = softmax(Q @ K.T / sqrt(d)) @ V        q, k, v: [n, d], n <= 128, d <= 128

VERIFIED IN THE SIMULATOR, NOT ON THE DEVICE. This V1 schedule passed verify_sdk.py on seat-65
against nki 0.6.0 / neuronx-cc 2.27.5334: every API name and keyword exists, correct on all 8
cases (the three level-8 shapes, seq 1/37/127, large values, identical rows; worst error 6.4e-4 of
output RMS, on large values), 1.00x the HBM byte floor, no hardware-hazard warnings; nkibench
level 8 passes 3/3 shapes. V0 (my_attention_v0.py) passed the same checks. Neither version has
run on a NeuronCore -- no latency numbers exist.

V1 changes the schedule, not the algorithm. V0 put 7 of its 17 instructions on the Vector engine,
which runs its queue in order, so steps the data does not order still waited in line:
  * the row sum sat ahead of the pT copy and held up P @ V. The exp instruction now accumulates it
    itself (activation_reduce), on the Scalar engine.
  * a separate step multiplied the row max by -scale only to build the exp bias. The scale is now
    applied to Q as it leaves PSUM, so S comes out of the matmul already scaled, and
    tensor_reduce(negate=True) returns the bias directly.
  * the qT copy runs on the Scalar engine, so the kT copy no longer queues behind it on Vector.
17 instructions become 16. Whether that is faster only a device profile can say.

Checked against the Neuron Agentic Development NKI docs (SDK 2.29 API reference), which changed:
  * nc_transpose requires dst to have data's dtype, so the qT/kT PSUM tiles now take the input
    dtype instead of float32. No change for float32 inputs; bf16 inputs would have failed.
  * nl.divide is not in the ISA's supported-operator table, so the final normalise is now
    reciprocal (Vector, n elements) then multiply. Confirmed on seat-65: the device compiler rejects
    op0=nl.divide ("unsupported operator 'divide'") although the simulator accepted it, so V0 and V2
    now multiply by the reciprocal too.
  * nc_transpose, nl.copy (the Identity activation), activation_reduce and a per-row (n, 1)
    tensor_scalar operand are all documented, so those VERIFY marks are gone.
  * tensor_reduce(negate=) is in the API reference but listed as removed in the skill's migration
    rules. The installed SDK has it (verify_sdk.py section 2b), so it stays.

    python verify_sdk.py
    python verify_sdk.py my_attention_v0.py      # the baseline, for comparison
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
    # nc_transpose requires dst to have the same dtype as data (NKI API docs), so these PSUM tiles
    # take the input dtype rather than float32 -- identical for float32 inputs, required for bf16.
    qT_ps = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.psum)
    kT_ps = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=qT_ps, data=q_sb)
    nisa.nc_transpose(dst=kT_ps, data=k_sb)
    qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
    kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
    # The two PSUM evacuations go to different engines, so the kT copy does not queue behind the
    # qT copy on Vector. The qT one also applies the softmax scale on the way through, so S comes
    # out of the matmul already scaled. Fine in float32: one extra rounding per element of Q.
    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)   # nl.copy = Identity activation
    nisa.tensor_copy(dst=kT, src=kT_ps)

    # 3. S = (scale * Q) @ K.T -> PSUM [n, n], float32, already scaled
    s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)

    # 4. Softmax along each row, in float32. Subtracting the row max keeps exp() from overflowing.
    # negate=True returns -max, which is the exp bias as it stands: no separate negate step.
    neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)

    # p = exp(S - max), with the row sum accumulated by the same instruction on the Scalar engine.
    # A separate tensor_reduce here would sit ahead of the pT copy in Vector's queue and hold up
    # P @ V. The max entry gives exp(0), so row_sum is never below about 1 and the divide is safe.
    p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
                           reduce_op=nl.add, reduce_res=row_sum)

    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys.
    # pT_ps matches p's float32, as nc_transpose requires.
    pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=pT_ps, data=p)
    pT = nl.ndarray((n, n), dtype=v.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pT, src=pT_ps)

    # 1 / row_sum, for the normalise in step 7. Issued after the pT copy so this Vector-engine
    # instruction does not sit ahead of it and delay P @ V; it only has to finish before step 7.
    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum, data=row_sum)

    # 6. O = P @ V -> PSUM [n, d], float32, not yet normalised
    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)

    # 7. Normalise last: n*d multiplies instead of n*n. Multiply by the reciprocal, not divide:
    # nl.divide is not in the ISA's supported-operator table. operand0 is a per-row (n, 1) float32
    # vector, which tensor_scalar accepts.
    o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)

    # 8. SBUF -> HBM
    nisa.dma_copy(dst=out, src=o_sb)
    return out
