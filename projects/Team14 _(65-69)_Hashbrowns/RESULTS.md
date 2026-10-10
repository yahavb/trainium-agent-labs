# Attention kernel comparison

Measured on **seat-65** at 2026-10-10 21:05. NKI version 0.6.0, neuronx-cc 2.27.5334.0, neuron-explorer 2.32.0.498-b1f3998.  
Each kernel: 1 correctness run on the device, then **5 profiled runs** (separate neuron-explorer captures, 2nd execution each). Latency = neuron-explorer `total_time`. Reproduce: `python compare_all.py --repeats 5`.

**How to read the verdicts:** a version is *faster* or *slower* than NKI version 0 only if every one of its runs beats, or loses to, every NKI version 0 run. Otherwise the ranges overlap and the difference is not distinguishable from run-to-run noise.

## Headline: seq x dim = 128x64

| Version | What it changes | Correct on device (error / RMS) | Median us | Range us | vs NKI version 0 | Verdict |
|---|---|---|---|---|---|---|
| **NKI version 0** | Baseline. | yes (3.4e-05) | 23.71 | 23.00 – 23.91 | 1.000x | baseline |
| **NKI version 1** | NKI version 0 rescheduled, same maths. | yes (3.4e-05) | 23.72 | 23.42 – 24.20 | 1.000x | within noise |
| **NKI version 2** | NKI version 1 with the P transpose and P @ V in bfloat16 instead of float32. | yes (1.2e-02) | 23.02 | 22.57 – 23.25 | 1.030x | within noise |
| **NKI version 3** | NKI version 1 with Q and K transposed by the DMA engine on the way in (dma_transpose), removing both Tensor-engine transposes and both PSUM copies. | yes (3.4e-05) | 22.10 | 21.53 – 22.27 | 1.073x | **faster** (every run) |
| **NKI version 4** | NKI version 1 with the back half (transpose P, copy, P @ V) in chunks of 32 keys, accumulating P @ V in PSUM. | yes (3.3e-05) | 25.01 | 24.95 – 25.26 | 0.948x | **slower** (every run) |

## Median latency on every shape (us)

| Version | 128x64 | 64x128 | 96x32 |
|---|---|---|---|
| **NKI version 0** | 23.71 | 23.11 | 22.86 |
| **NKI version 1** | 23.72 (1.00x, within noise) | 23.25 (0.99x, within noise) | 22.92 (1.00x, within noise) |
| **NKI version 2** | 23.02 (1.03x, within noise) | 22.92 (1.01x, within noise) | 22.26 (1.03x, within noise) |
| **NKI version 3** | 22.10 (1.07x, **faster** (every run)) | 21.38 (1.08x, **faster** (every run)) | 21.25 (1.08x, **faster** (every run)) |
| **NKI version 4** | 25.01 (0.95x, **slower** (every run)) | 23.62 (0.98x, within noise) | 23.27 (0.98x, within noise) |

Speedups are NKI version 0 median / version median: above 1.00x is faster.

## Where the time goes (128x64, medians)

Busy % is the share of the kernel's total time each engine was executing. Instruction counts are what the hardware ran after compilation, not the nisa calls in the source.

| Version | Total us | Tensor % | Vector % | Scalar % | GpSimd % | DMA % | HW instr T / V / S / G | HBM read / write B | MFU % |
|---|---|---|---|---|---|---|---|---|---|
| **NKI version 0** | 23.71 | 28 | 25 | 24 | 26 | 39 | 75 / 66 / 63 / 66 | 98304 / 32768 | 0.0045 |
| **NKI version 1** | 23.72 | 29 | 22 | 25 | 27 | 37 | 80 / 64 / 63 / 66 | 98304 / 32768 | 0.0045 |
| **NKI version 2** | 23.02 | 27 | 23 | 27 | 28 | 40 | 78 / 65 / 64 / 67 | 98304 / 32768 | 0.0029 |
| **NKI version 3** | 22.10 | 27 | 23 | 26 | 24 | 41 | 75 / 63 / 63 / 64 | 98304 / 32768 | 0.0048 |
| **NKI version 4** | 25.01 | 33 | 24 | 24 | 32 | 39 | 101 / 67 / 63 / 69 | 98304 / 32768 | 0.0043 |

## What each version changes

### NKI version 0 — `my_attention_v0.py`

Baseline. Q and K transposed on the Tensor engine and copied out of PSUM on Vector; S = Q K^T; row max; bias = -scale * max; exp; row sum; transpose P; P V; normalise by reciprocal-and-multiply.

Its 18 nisa instructions, in source order:

```text
nisa.dma_copy(dst=q_sb, src=q)
nisa.dma_copy(dst=k_sb, src=k)
nisa.dma_copy(dst=v_sb, src=v)
nisa.nc_transpose(dst=qT_ps, data=q_sb)
nisa.nc_transpose(dst=kT_ps, data=k_sb)
nisa.tensor_copy(dst=qT, src=qT_ps)
nisa.tensor_copy(dst=kT, src=kT_ps)
nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=s_ps, axis=1)
nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-scale)
nisa.activation(dst=p, op=nl.exp, data=s_ps, bias=neg_max, scale=scale)
nisa.tensor_reduce(dst=row_sum, op=nl.add, data=p, axis=1)
nisa.nc_transpose(dst=pT_ps, data=p)
nisa.tensor_copy(dst=pT, src=pT_ps)
nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
nisa.reciprocal(dst=inv_sum, data=row_sum)
nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
nisa.dma_copy(dst=out, src=o_sb)
```

### NKI version 1 — `my_attention.py`

NKI version 0 rescheduled, same maths. The softmax scale rides on Q's PSUM copy (Scalar engine), tensor_reduce(negate=True) yields the exp bias directly, and the exp instruction accumulates the row sum (activation_reduce), taking work off the in-order Vector queue.

Built from **NKI version 0**. Instructions in source order: 18 → 16 (`[loop]` = inside a loop, so issued more than once).

```diff
@@ -5,13 +5,11 @@
 nisa.nc_transpose(dst=kT_ps, data=k_sb)
-nisa.tensor_copy(dst=qT, src=qT_ps)
+nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)
 nisa.tensor_copy(dst=kT, src=kT_ps)
 nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
-nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=s_ps, axis=1)
-nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-scale)
-nisa.activation(dst=p, op=nl.exp, data=s_ps, bias=neg_max, scale=scale)
-nisa.tensor_reduce(dst=row_sum, op=nl.add, data=p, axis=1)
+nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
+nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max, reduce_op=nl.add, reduce_res=row_sum)
 nisa.nc_transpose(dst=pT_ps, data=p)
 nisa.tensor_copy(dst=pT, src=pT_ps)
+nisa.reciprocal(dst=inv_sum, data=row_sum)
 nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
-nisa.reciprocal(dst=inv_sum, data=row_sum)
 nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
```

<details><summary>Full code diff NKI version 0 → NKI version 1 (83 lines)</summary>

```diff
--- NKI version 0
+++ NKI version 1
@@ -2,4 +2,5 @@
 def nki_attention_(q, k, v):
     n, d = q.shape
+    assert n <= 128 and d <= 128, f"single-tile kernel: needs seq <= 128 and dim <= 128, got {n}, {d}"
     scale = 1.0 / (d ** 0.5)
 
@@ -15,35 +16,46 @@
 
     # 2. Transpose Q and K so d is on the partition axis: nc_matmul contracts over partitions
-    qT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
-    kT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
-    nisa.nc_transpose(dst=qT_ps, data=q_sb)                                    # VERIFY
-    nisa.nc_transpose(dst=kT_ps, data=k_sb)                                    # VERIFY
+    # nc_transpose requires dst to have the same dtype as data (NKI API docs), so these PSUM tiles
+    # take the input dtype rather than float32 -- identical for float32 inputs, required for bf16.
+    qT_ps = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.psum)
+    kT_ps = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.psum)
+    nisa.nc_transpose(dst=qT_ps, data=q_sb)
+    nisa.nc_transpose(dst=kT_ps, data=k_sb)
     qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
     kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
-    nisa.tensor_copy(dst=qT, src=qT_ps)
+    # The two PSUM evacuations go to different engines, so the kT copy does not queue behind the
+    # qT copy on Vector. The qT one also applies the softmax scale on the way through, so S comes
+    # out of the matmul already scaled. Fine in float32: one extra rounding per element of Q.
+    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)   # nl.copy = Identity activation
     nisa.tensor_copy(dst=kT, src=kT_ps)
 
-    # 3. S = Q @ K.T -> PSUM [n, n], float32
+    # 3. S = (scale * Q) @ K.T -> PSUM [n, n], float32, already scaled
     s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
     nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
 
     # 4. Softmax along each row, in float32. Subtracting the row max keeps exp() from overflowing.
-    row_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=s_ps, axis=1)          # VERIFY
+    # negate=True returns -max, which is the exp bias as it stands: no separate negate step.
     neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.tensor_scalar(dst=neg_max, data=row_max, op0=nl.multiply, operand0=-scale)
+    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
 
-    # p = exp(scale * S - scale * max): scale and max-subtraction folded into one instruction
+    # p = exp(S - max), with the row sum accumulated by the same instruction on the Scalar engine.
+    # A separate tensor_reduce here would sit ahead of the pT copy in Vector's queue and hold up
+    # P @ V. The max entry gives exp(0), so row_sum is never below about 1 and the divide is safe.
     p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.activation(dst=p, op=nl.exp, data=s_ps, bias=neg_max, scale=scale)    # VERIFY
+    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
+    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
+                           reduce_op=nl.add, reduce_res=row_sum)
 
-    row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.tensor_reduce(dst=row_sum, op=nl.add, data=p, axis=1)                 # VERIFY
-
-    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys
+    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys.
+    # pT_ps matches p's float32, as nc_transpose requires.
     pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
-    nisa.nc_transpose(dst=pT_ps, data=p)                                       # VERIFY
+    nisa.nc_transpose(dst=pT_ps, data=p)
     pT = nl.ndarray((n, n), dtype=v.dtype, buffer=nl.sbuf)
     nisa.tensor_copy(dst=pT, src=pT_ps)
+
+    # 1 / row_sum, for the normalise in step 7. Issued after the pT copy so this Vector-engine
+    # instruction does not sit ahead of it and delay P @ V; it only has to finish before step 7.
+    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
+    nisa.reciprocal(dst=inv_sum, data=row_sum)
 
     # 6. O = P @ V -> PSUM [n, d], float32, not yet normalised
@@ -51,8 +63,7 @@
     nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
 
-    # 7. Normalise last: n*d multiplies instead of n*n divides. The device compiler has no divide in
-    # tensor_scalar, so multiply by the reciprocal of the row sums.
-    inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.reciprocal(dst=inv_sum, data=row_sum)
+    # 7. Normalise last: n*d multiplies instead of n*n. Multiply by the reciprocal, not divide:
+    # nl.divide is not in the ISA's supported-operator table. operand0 is a per-row (n, 1) float32
+    # vector, which tensor_scalar accepts.
     o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
     nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
```
</details>

### NKI version 2 — `my_attention_v2_bf16.py`

NKI version 1 with the P transpose and P @ V in bfloat16 instead of float32. Q K^T stays float32.

Built from **NKI version 1**. Instructions in source order: 16 → 17 (`[loop]` = inside a loop, so issued more than once).

```diff
@@ -9,2 +9,3 @@
 nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
+nisa.tensor_copy(dst=v_bf, src=v_sb)
 nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max, reduce_op=nl.add, reduce_res=row_sum)
@@ -13,3 +14,3 @@
 nisa.reciprocal(dst=inv_sum, data=row_sum)
-nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
+nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_bf)
 nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
```

<details><summary>Full code diff NKI version 1 → NKI version 2 (81 lines)</summary>

```diff
--- NKI version 1
+++ NKI version 2
@@ -16,16 +16,12 @@
 
     # 2. Transpose Q and K so d is on the partition axis: nc_matmul contracts over partitions
-    # nc_transpose requires dst to have the same dtype as data (NKI API docs), so these PSUM tiles
-    # take the input dtype rather than float32 -- identical for float32 inputs, required for bf16.
-    qT_ps = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.psum)
-    kT_ps = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.psum)
-    nisa.nc_transpose(dst=qT_ps, data=q_sb)
-    nisa.nc_transpose(dst=kT_ps, data=k_sb)
+    qT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
+    kT_ps = nl.ndarray((d, n), dtype=nl.float32, buffer=nl.psum)
+    nisa.nc_transpose(dst=qT_ps, data=q_sb)                                    # VERIFY
+    nisa.nc_transpose(dst=kT_ps, data=k_sb)                                    # VERIFY
     qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
     kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
-    # The two PSUM evacuations go to different engines, so the kT copy does not queue behind the
-    # qT copy on Vector. The qT one also applies the softmax scale on the way through, so S comes
-    # out of the matmul already scaled. Fine in float32: one extra rounding per element of Q.
-    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)   # nl.copy = Identity activation
+    # Scale folded into Q on its way out of PSUM, on the Scalar engine; kT on Vector in parallel.
+    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)               # VERIFY nl.copy
     nisa.tensor_copy(dst=kT, src=kT_ps)
 
@@ -34,36 +30,35 @@
     nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
 
-    # 4. Softmax along each row, in float32. Subtracting the row max keeps exp() from overflowing.
-    # negate=True returns -max, which is the exp bias as it stands: no separate negate step.
+    # 4. Softmax along each row. negate=True returns -max, the exp bias as it stands.
     neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
+    nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)  # VERIFY negate
 
-    # p = exp(S - max), with the row sum accumulated by the same instruction on the Scalar engine.
-    # A separate tensor_reduce here would sit ahead of the pT copy in Vector's queue and hold up
-    # P @ V. The max entry gives exp(0), so row_sum is never below about 1 and the divide is safe.
-    p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
+    # V to bf16 for P @ V. Issued here, not after the load, so it does not sit ahead of the kT copy
+    # in Vector's queue: Vector is idle at this point while exp runs on the Scalar engine.
+    v_bf = nl.ndarray((n, d), dtype=nl.bfloat16, buffer=nl.sbuf)
+    nisa.tensor_copy(dst=v_bf, src=v_sb)
+
+    # p = exp(S - max) written straight out as bf16; the row sum is still accumulated in float32.
+    p = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.sbuf)
     row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
+    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,         # VERIFY
                            reduce_op=nl.add, reduce_res=row_sum)
 
-    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys.
-    # pT_ps matches p's float32, as nc_transpose requires.
-    pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
-    nisa.nc_transpose(dst=pT_ps, data=p)
-    pT = nl.ndarray((n, n), dtype=v.dtype, buffer=nl.sbuf)
+    # 5. Transpose P in bf16: a Tensor-engine transpose keeps its input's dtype
+    pT_ps = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.psum)
+    nisa.nc_transpose(dst=pT_ps, data=p)                                       # VERIFY bf16 PSUM
+    pT = nl.ndarray((n, n), dtype=nl.bfloat16, buffer=nl.sbuf)
     nisa.tensor_copy(dst=pT, src=pT_ps)
 
-    # 1 / row_sum, for the normalise in step 7. Issued after the pT copy so this Vector-engine
-    # instruction does not sit ahead of it and delay P @ V; it only has to finish before step 7.
+    # 1 / row_sum, issued after the pT copy so it does not delay P @ V on the Vector queue.
+    # The device compiler has no divide in tensor_scalar, so step 7 multiplies by this.
     inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
     nisa.reciprocal(dst=inv_sum, data=row_sum)
 
-    # 6. O = P @ V -> PSUM [n, d], float32, not yet normalised
+    # 6. O = P @ V with bf16 operands -> PSUM [n, d], float32, not yet normalised
     o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
-    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
+    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_bf)
 
-    # 7. Normalise last: n*d multiplies instead of n*n. Multiply by the reciprocal, not divide:
-    # nl.divide is not in the ISA's supported-operator table. operand0 is a per-row (n, 1) float32
-    # vector, which tensor_scalar accepts.
+    # 7. Normalise last: n*d multiplies instead of n*n
     o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
     nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
```
</details>

### NKI version 3 — `my_attention_v3_dmaT.py`

NKI version 1 with Q and K transposed by the DMA engine on the way in (dma_transpose), removing both Tensor-engine transposes and both PSUM copies.

Built from **NKI version 1**. Instructions in source order: 16 → 13 (`[loop]` = inside a loop, so issued more than once).

```diff
@@ -1,11 +1,8 @@
-nisa.dma_copy(dst=q_sb, src=q)
-nisa.dma_copy(dst=k_sb, src=k)
+nisa.dma_transpose(dst=qT, src=q)
+nisa.dma_transpose(dst=kT, src=k)
 nisa.dma_copy(dst=v_sb, src=v)
-nisa.nc_transpose(dst=qT_ps, data=q_sb)
-nisa.nc_transpose(dst=kT_ps, data=k_sb)
-nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)
-nisa.tensor_copy(dst=kT, src=kT_ps)
 nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
 nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
-nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max, reduce_op=nl.add, reduce_res=row_sum)
+nisa.tensor_scalar(dst=bias, data=neg_max, op0=nl.multiply, operand0=scale)
+nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=bias, scale=scale, reduce_op=nl.add, reduce_res=row_sum)
 nisa.nc_transpose(dst=pT_ps, data=p)
```

<details><summary>Full code diff NKI version 1 → NKI version 3 (85 lines)</summary>

```diff
--- NKI version 1
+++ NKI version 3
@@ -7,46 +7,29 @@
     out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)
 
-    # 1. HBM -> SBUF: rows on the partition axis
-    q_sb = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
-    k_sb = nl.ndarray((n, d), dtype=k.dtype, buffer=nl.sbuf)
+    # 1. HBM -> SBUF. Q and K land transposed, d on the partition axis, ready for nc_matmul.
+    qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
+    kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
     v_sb = nl.ndarray((n, d), dtype=v.dtype, buffer=nl.sbuf)
-    nisa.dma_copy(dst=q_sb, src=q)
-    nisa.dma_copy(dst=k_sb, src=k)
+    nisa.dma_transpose(dst=qT, src=q)                                          # VERIFY fp32
+    nisa.dma_transpose(dst=kT, src=k)                                          # VERIFY fp32
     nisa.dma_copy(dst=v_sb, src=v)
 
-    # 2. Transpose Q and K so d is on the partition axis: nc_matmul contracts over partitions
-    # nc_transpose requires dst to have the same dtype as data (NKI API docs), so these PSUM tiles
-    # take the input dtype rather than float32 -- identical for float32 inputs, required for bf16.
-    qT_ps = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.psum)
-    kT_ps = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.psum)
-    nisa.nc_transpose(dst=qT_ps, data=q_sb)
-    nisa.nc_transpose(dst=kT_ps, data=k_sb)
-    qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
-    kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
-    # The two PSUM evacuations go to different engines, so the kT copy does not queue behind the
-    # qT copy on Vector. The qT one also applies the softmax scale on the way through, so S comes
-    # out of the matmul already scaled. Fine in float32: one extra rounding per element of Q.
-    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)   # nl.copy = Identity activation
-    nisa.tensor_copy(dst=kT, src=kT_ps)
-
-    # 3. S = (scale * Q) @ K.T -> PSUM [n, n], float32, already scaled
+    # 2. S = Q @ K.T -> PSUM [n, n], float32, unscaled
     s_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
     nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
 
-    # 4. Softmax along each row, in float32. Subtracting the row max keeps exp() from overflowing.
-    # negate=True returns -max, which is the exp bias as it stands: no separate negate step.
+    # 3. Softmax along each row, in float32. -max straight from the reduction, then * scale for
+    # the exp bias: exp(scale * S - scale * max).
     neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
     nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
+    bias = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
+    nisa.tensor_scalar(dst=bias, data=neg_max, op0=nl.multiply, operand0=scale)
 
-    # p = exp(S - max), with the row sum accumulated by the same instruction on the Scalar engine.
-    # A separate tensor_reduce here would sit ahead of the pT copy in Vector's queue and hold up
-    # P @ V. The max entry gives exp(0), so row_sum is never below about 1 and the divide is safe.
     p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
     row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
-    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max,
+    nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=bias, scale=scale,
                            reduce_op=nl.add, reduce_res=row_sum)
 
-    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys.
-    # pT_ps matches p's float32, as nc_transpose requires.
+    # 4. Transpose P so keys are on the partition axis: P @ V contracts over keys
     pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
     nisa.nc_transpose(dst=pT_ps, data=p)
@@ -54,20 +37,15 @@
     nisa.tensor_copy(dst=pT, src=pT_ps)
 
-    # 1 / row_sum, for the normalise in step 7. Issued after the pT copy so this Vector-engine
-    # instruction does not sit ahead of it and delay P @ V; it only has to finish before step 7.
+    # 1 / row_sum, issued after the pT copy so it does not delay P @ V on the Vector queue
     inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
     nisa.reciprocal(dst=inv_sum, data=row_sum)
 
-    # 6. O = P @ V -> PSUM [n, d], float32, not yet normalised
+    # 5. O = P @ V -> PSUM [n, d], then normalise while copying out of PSUM
     o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
     nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
-
-    # 7. Normalise last: n*d multiplies instead of n*n. Multiply by the reciprocal, not divide:
-    # nl.divide is not in the ISA's supported-operator table. operand0 is a per-row (n, 1) float32
-    # vector, which tensor_scalar accepts.
     o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
     nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
 
-    # 8. SBUF -> HBM
+    # 6. SBUF -> HBM
     nisa.dma_copy(dst=out, src=o_sb)
     return out
```
</details>

### NKI version 4 — `my_attention_v4_kchunk.py`

NKI version 1 with the back half (transpose P, copy, P @ V) in chunks of 32 keys, accumulating P @ V in PSUM.

Built from **NKI version 1**. Instructions in source order: 16 → 16 (`[loop]` = inside a loop, so issued more than once).

```diff
@@ -2,3 +2,2 @@
 nisa.dma_copy(dst=k_sb, src=k)
-nisa.dma_copy(dst=v_sb, src=v)
 nisa.nc_transpose(dst=qT_ps, data=q_sb)
@@ -10,6 +9,7 @@
 nisa.activation_reduce(dst=p, op=nl.exp, data=s_ps, bias=neg_max, reduce_op=nl.add, reduce_res=row_sum)
-nisa.nc_transpose(dst=pT_ps, data=p)
-nisa.tensor_copy(dst=pT, src=pT_ps)
+[loop] nisa.dma_copy(dst=v_c, src=v[c * kc:(c + 1) * kc, :])
+[loop] nisa.nc_transpose(dst=pT_ps, data=p[:, c * kc:(c + 1) * kc])
+[loop] nisa.tensor_copy(dst=pT, src=pT_ps)
+[loop] nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_c)
 nisa.reciprocal(dst=inv_sum, data=row_sum)
-nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
 nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
```

<details><summary>Full code diff NKI version 1 → NKI version 4 (88 lines)</summary>

```diff
--- NKI version 1
+++ NKI version 4
@@ -4,4 +4,6 @@
     assert n <= 128 and d <= 128, f"single-tile kernel: needs seq <= 128 and dim <= 128, got {n}, {d}"
     scale = 1.0 / (d ** 0.5)
+    kc = 32 if n % 32 == 0 else n
+    chunks = n // kc
 
     out = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.shared_hbm)
@@ -10,12 +12,8 @@
     q_sb = nl.ndarray((n, d), dtype=q.dtype, buffer=nl.sbuf)
     k_sb = nl.ndarray((n, d), dtype=k.dtype, buffer=nl.sbuf)
-    v_sb = nl.ndarray((n, d), dtype=v.dtype, buffer=nl.sbuf)
     nisa.dma_copy(dst=q_sb, src=q)
     nisa.dma_copy(dst=k_sb, src=k)
-    nisa.dma_copy(dst=v_sb, src=v)
 
-    # 2. Transpose Q and K so d is on the partition axis: nc_matmul contracts over partitions
-    # nc_transpose requires dst to have the same dtype as data (NKI API docs), so these PSUM tiles
-    # take the input dtype rather than float32 -- identical for float32 inputs, required for bf16.
+    # 2. Transpose Q and K so d is on the partition axis; scale folded into Q on the Scalar engine
     qT_ps = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.psum)
     kT_ps = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.psum)
@@ -24,8 +22,5 @@
     qT = nl.ndarray((d, n), dtype=q.dtype, buffer=nl.sbuf)
     kT = nl.ndarray((d, n), dtype=k.dtype, buffer=nl.sbuf)
-    # The two PSUM evacuations go to different engines, so the kT copy does not queue behind the
-    # qT copy on Vector. The qT one also applies the softmax scale on the way through, so S comes
-    # out of the matmul already scaled. Fine in float32: one extra rounding per element of Q.
-    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)   # nl.copy = Identity activation
+    nisa.activation(dst=qT, op=nl.copy, data=qT_ps, scale=scale)
     nisa.tensor_copy(dst=kT, src=kT_ps)
 
@@ -34,12 +29,7 @@
     nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
 
-    # 4. Softmax along each row, in float32. Subtracting the row max keeps exp() from overflowing.
-    # negate=True returns -max, which is the exp bias as it stands: no separate negate step.
+    # 4. Softmax along each row, whole rows: the chunks below need the final max and sum
     neg_max = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
     nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=s_ps, axis=1, negate=True)
-
-    # p = exp(S - max), with the row sum accumulated by the same instruction on the Scalar engine.
-    # A separate tensor_reduce here would sit ahead of the pT copy in Vector's queue and hold up
-    # P @ V. The max entry gives exp(0), so row_sum is never below about 1 and the divide is safe.
     p = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.sbuf)
     row_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
@@ -47,27 +37,22 @@
                            reduce_op=nl.add, reduce_res=row_sum)
 
-    # 5. Transpose P so keys are on the partition axis: P @ V contracts over keys.
-    # pT_ps matches p's float32, as nc_transpose requires.
-    pT_ps = nl.ndarray((n, n), dtype=nl.float32, buffer=nl.psum)
-    nisa.nc_transpose(dst=pT_ps, data=p)
-    pT = nl.ndarray((n, n), dtype=v.dtype, buffer=nl.sbuf)
-    nisa.tensor_copy(dst=pT, src=pT_ps)
+    # 5. O = P @ V, one chunk of keys at a time, summed in PSUM. Every tile here starts at
+    # partition 0: V is loaded per chunk rather than sliced, because slicing v_sb by keys would put
+    # the matmul's moving operand at partition 32, 64, ...
+    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
+    for c in nl.affine_range(chunks):
+        v_c = nl.ndarray((kc, d), dtype=v.dtype, buffer=nl.sbuf)
+        nisa.dma_copy(dst=v_c, src=v[c * kc:(c + 1) * kc, :])
+        pT_ps = nl.ndarray((kc, n), dtype=nl.float32, buffer=nl.psum)
+        nisa.nc_transpose(dst=pT_ps, data=p[:, c * kc:(c + 1) * kc])
+        pT = nl.ndarray((kc, n), dtype=v.dtype, buffer=nl.sbuf)
+        nisa.tensor_copy(dst=pT, src=pT_ps)
+        nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_c)
 
-    # 1 / row_sum, for the normalise in step 7. Issued after the pT copy so this Vector-engine
-    # instruction does not sit ahead of it and delay P @ V; it only has to finish before step 7.
+    # 6. Normalise while copying out of PSUM, then SBUF -> HBM
     inv_sum = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
     nisa.reciprocal(dst=inv_sum, data=row_sum)
-
-    # 6. O = P @ V -> PSUM [n, d], float32, not yet normalised
-    o_ps = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
-    nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
-
-    # 7. Normalise last: n*d multiplies instead of n*n. Multiply by the reciprocal, not divide:
-    # nl.divide is not in the ISA's supported-operator table. operand0 is a per-row (n, 1) float32
-    # vector, which tensor_scalar accepts.
     o_sb = nl.ndarray((n, d), dtype=out.dtype, buffer=nl.sbuf)
     nisa.tensor_scalar(dst=o_sb, data=o_ps, op0=nl.multiply, operand0=inv_sum)
-
-    # 8. SBUF -> HBM
     nisa.dma_copy(dst=out, src=o_sb)
     return out
```
</details>

## Caveats

- Correctness is one random input per shape on the device (scale 1). The hostile cases (large values, identical rows, ragged lengths) are covered in the simulator by `verify_sdk.py`, not here.
- Repeats re-execute the same compiled NEFF, so they measure execution noise, not compile variation.
- These shapes are tiny; at this size the kernels are bound by the dependency chain between instructions, not by any engine's throughput, so expect small differences.
