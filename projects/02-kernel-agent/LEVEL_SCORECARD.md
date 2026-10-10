# Level scorecard

Observed results on seat-265 only. Original five-repeat baseline: zero of five solved on each level. Official checker unchanged. No device-verified kernels.

| Level | Operation | Best autonomous cold-start reward | Cases passed/total | Current status |
|---|---|---:|---:|---|
| 1 | Average pooling | 0.50 (active, provisional) | 0/4 | Planner/legalizer cold run active; prior full runs 0.30. 0.50 is execution credit, not numerical correctness. |
| 2 | Free-axis transpose within each partition | 0.30 | 0/4 | Full agent run completed at 0.30; DMA/indexing failures. New primitives verified independently; new model evaluation pending. |
| 3 | Single-tile matmul | 1.00 | 1/1 | Four recorded cold-start successes across different configurations; initial shapes_buffers generation, not proof of repair benefit. |
| 4 | Tiled matmul | 0.625 | 1/4 | Small single-tile case passes; larger K/M/N loads and coverage remain invalid. |

Autonomous cold-start solved levels: **1/4 (Level 3)**. No matched repeated reliability gain or LoRA benchmark gain established. Separate Level 1 results: expert-edited warm diagnostic 1.00/4 cases; generic instruction-legalizer warm replay 1.00/4 cases, explicitly excluded from cold-start solve count.

Latest full Level 1/2 run: runs/controlled-20261010T192716-jp07xp0l. Planner Level 1: runs/controlled-20261010T194114-ln8yl9ak and runs/controlled-20261010T195135-qyw82zco, both .30. Current cold planner/legalizer: runs/controlled-20261010T201403-vxbwe5dd. Automatic warm replay: runs/level1-automatic-primitive-replay-72i8u7g7. Formal original/LoRA x legacy/planner comparison remains pending; see LORA_COMPARISON.md rather than substituting historical scores into its cells.

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
