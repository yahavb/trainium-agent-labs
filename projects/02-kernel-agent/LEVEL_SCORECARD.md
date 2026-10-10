# Level scorecard

Latest original baseline: five repetitions, 0/5 solved on EACH level. Scores below are from this pod only. CPU-checker verification is distinct from device execution. No device-verified kernels. Current A–D evaluation is still running; completed-arm results are provisional at this timestamp.

| Level | Operation | Best reward | Cases passed | Cases total | Verified cold-start solves recorded | Best experiment | Dominant remaining error |
|---|---|---:|---:|---:|---:|---|---|
| 1 | Average pooling | 0.30 | 0 | 4 | 0 | Original/pilot | DMA window/allocation mismatch; invalid reductions |
| 2 | Free-axis transpose per partition | 0.30 historical; recent live run pending | 0 | 4 | 0 | Original baseline, private forensic replay | DMA ignores P/free layout |
| 3 | Single-tile matmul | 1.00 | 1 | 1 | 4 | A_full, A_current, B_targeted, D_full_adaptive | Other perspectives: DMA/PSUM shape failures |
| 4 | Tiled matmul | 0.625 | 1 | 4 | 0 | Historical private replay / current targeted arm | Whole-input DMA limits, M/N/K tiling/accumulation |

The four Level 3 successes are different cold-start model requests, all at round zero, usually candidate 2 / shapes_buffers. They are not repair-driven solves or an intentionally balanced five-repeat study. One A_full kernel was independently replayed immediately; verified-archive-20261010-initial2 replays the completed earlier successes. The newest D_full_adaptive success will be archived once its full experiment finishes.

## Exact benchmark specifications

### Level 1: average pooling 2D

Entry: `tensor_avgpool_kernel(x, pool_size)`. Cases: `[{"shape": [32, 32, 32], "pool_size": 2}, {"shape": [128, 16, 16], "pool_size": 4}, {"shape": [8, 24, 24], "pool_size": 3}, {"shape": [64, 8, 8], "pool_size": 2}]`. Input dtype float32; output preserves input dtype.
Mathematics: average each complete non-overlapping p×p spatial window; output [C,H//p,W//p]. Preserve C; reduce window axes only; divisor p².
Constraints: on-chip rank at least two, partition <=128; ordinary matmul K/M<=128 and N<=512 per instruction. The 512-element PSUM bank/matmul rule is not a universal allocation limit. Larger cases require appropriate tiling. Existing checker input-integrity, numerical RMS-normalized tolerance 0.02 and hardware-hazard checks remain authoritative; no level 1–4 traffic-threshold override.

### Level 2: 2D transpose

Entry: `tensor_transpose2D_kernel_(x, shape2D)`. Cases: `[{"shape": [32, 12], "shape2D": [3, 4]}, {"shape": [128, 64], "shape2D": [8, 8]}, {"shape": [64, 128], "shape2D": [4, 32]}, {"shape": [8, 35], "shape2D": [5, 7]}]`. Input dtype float32; output preserves input dtype.
Mathematics: preserve P and reorder each flattened F1×F2 row into F2×F1; output remains [P,F1*F2]. A direct partition/free transpose of [P,F] is a different operation.
Constraints: on-chip rank at least two, partition <=128; ordinary matmul K/M<=128 and N<=512 per instruction. The 512-element PSUM bank/matmul rule is not a universal allocation limit. Larger cases require appropriate tiling. Existing checker input-integrity, numerical RMS-normalized tolerance 0.02 and hardware-hazard checks remain authoritative; no level 1–4 traffic-threshold override.

### Level 3: matmul, single tile

Entry: `nki_matmul_basic_(lhsT, rhs)`. Cases: `[{"K": 128, "M": 64, "N": 512}]`. Input dtype float32; output preserves input dtype.
Mathematics: lhsT[K,M] transposed logically times rhs[K,N], output [M,N]. On-chip operands in SBUF; ordinary result-shaped FP32 PSUM then separate SBUF-to-HBM result transfer.
Constraints: on-chip rank at least two, partition <=128; ordinary matmul K/M<=128 and N<=512 per instruction. The 512-element PSUM bank/matmul rule is not a universal allocation limit. Larger cases require appropriate tiling. Existing checker input-integrity, numerical RMS-normalized tolerance 0.02 and hardware-hazard checks remain authoritative; no level 1–4 traffic-threshold override.

### Level 4: matmul, tiled

Entry: `nki_matmul_tiled_(lhsT, rhs)`. Cases: `[{"K": 128, "M": 128, "N": 512}, {"K": 256, "M": 256, "N": 1024}, {"K": 512, "M": 128, "N": 512}, {"K": 256, "M": 512, "N": 1024}]`. Input dtype float32; output preserves input dtype.
Mathematics: lhsT[K,M] transposed logically times rhs[K,N], output [M,N]. On-chip operands in SBUF; ordinary result-shaped FP32 PSUM then separate SBUF-to-HBM result transfer.
Constraints: on-chip rank at least two, partition <=128; ordinary matmul K/M<=128 and N<=512 per instruction. The 512-element PSUM bank/matmul rule is not a universal allocation limit. Larger cases require appropriate tiling. Existing checker input-integrity, numerical RMS-normalized tolerance 0.02 and hardware-hazard checks remain authoritative; no level 1–4 traffic-threshold override.
