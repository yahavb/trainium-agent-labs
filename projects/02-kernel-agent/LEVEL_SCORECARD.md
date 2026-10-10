# Level scorecard

## Current system: generated planner + legalizer, revision dd85856, base Qwen3-8B on Trainium2 (as of 21:47 UTC)

| Level | Score | Shapes | Run |
|---|---:|---:|---|
| 1 Average pooling | **1.00** (warm start) | 4/4 | runs/controlled-20261010T214022-7d05zuet |
| 2 Transpose | **1.00** (cold, round 0) | 4/4 | runs/controlled-20261010T214002-44j993c1 |
| 3 Single-tile matmul | **1.00** (cold, round 0) | 1/1 | runs/controlled-20261010T214002-44j993c1 |
| 4 Tiled matmul | running | – | runs/controlled-20261010T214002-44j993c1 |
| 5 Matmul, hoisted loads | running | – | runs/controlled-20261010T214002-44j993c1 |
| 6 Matmul, M/N blocked | 0.625 after round 0 (running) | 1/4 | runs/controlled-20261010T214002-44j993c1 |
| 7 Matmul, M/N/K blocked | 0.30 after round 0 (running) | 0/4 | runs/controlled-20261010T214002-44j993c1 |
| 8 Single-head attention | 0.30 after round 0 (running) | 0/3 | runs/controlled-20261010T214002-44j993c1 |

The tables below are earlier runs; each run directory records the exact source revision used.

Observed results on seat-265 only. Original five-repeat baseline: zero of five solved on each level. Official checker unchanged. No device-verified kernels.

| Level | Operation | Best cold-start agent reward | Shapes | Evidence / status (updated 20:58 UTC) |
|---|---|---:|---:|---|
| 1 | Average pooling | **1.00 (warm start, main L1 approach)** | 4/4 | Full agent run with --warm-start (saved Qwen candidate 99bd31bd…, legalizer opcode_namespace + hbm_scalar_staging): controlled-20261010T210340-n8lcsja5, round 0, 1.00. The graded kernel is byte-identical to locked verified-level1-locked-gowa0nvv. Best cold start: 0.50 (cold runs stopped once warm start became the L1 approach). |
| 2 | Free-axis transpose within each partition | **1.00** | 4/4 | **Cold-start solve**, round 0, raw Qwen (legalizer no-op): controlled-20261010T203725-onnn7nmy; locked verified-level2-locked-coeoi3p4, replay 1.00. |
| 3 | Single-tile matmul | **1.00** | 1/1 | Four earlier cold-start successes (Codex runs). |
| 4 | Tiled matmul | **1.00** | 4/4 | **Cold-start solve**, round 0, raw Qwen (legalizer no-op): controlled-20261010T205651-l9444j4q; locked verified-level4-locked-n1kmsi6q, replay 1.00, traffic passes every shape. Earlier runs: 0.625/0.30. |
| 5 | Matmul, loads hoisted (traffic <=1.6x) | **1.00** | 4/4 | **Cold-start solve**, round 0, raw Qwen (legalizer no-op): controlled-20261010T205124-kcpiu5w7; locked verified-level5-locked-u5ybjn5a, replay 1.00, traffic passes every shape. |
| 6 | Matmul, M/N blocked (traffic <=1.25x) | **1.00** | 4/4 | **Cold-start solve**, round 0, raw Qwen (legalizer no-op): controlled-20261010T205124-kcpiu5w7; locked verified-level6-locked-r79vct0e, replay 1.00, traffic passes every shape. |
| 7 | Matmul, M/N/K blocked (traffic <=1.05x) | **1.00** | 4/4 | **Cold-start solve**, round 0, raw Qwen (legalizer no-op): controlled-20261010T205124-kcpiu5w7; locked verified-level7-locked-nriea9vh, replay 1.00, traffic passes every shape. |
| 8 | Single-head attention | 0.30 (active) | 0/3 | Two parallel cold runs active: controlled-20261010T205302-lg75w1b2 (round 0: 0.30, invented nc_matmul kwargs) and controlled-20261010T210530-i_w0aqpz. agent.grade crashed (KeyError 'M') on any correct L8 shape before the fix in this sprint; nkibench.py --check has the same latent bug (benchmark file left unchanged). |

Planner: `kernel_planner.py` generates each level's compact plan from the benchmark specification (reference signature and reference outputs on the official shapes), the installed NKI tile limits and instruction signatures named by the task prompt, and the level's HBM traffic budget. Every score in these tables is tied to its run directory, which freezes the exact source used. Generated-planner re-run of Levels 2–8: in progress. Fine-tuned (LoRA) runs on L1–L4 are active on a separate CPU endpoint (runs/lora-fast-eval-20261010T203757-0pFD); no LoRA scores yet.

Locked simulator-verified kernels (warm-start or cold-start, preserved read-only):

| Level | Locked kernel | sha256 | Full official replays | Origin |
|---|---|---|---|---|
| 1 | runs/verified-level1-locked-gowa0nvv/kernel.py | 3443e646…abfe | 1.00, 4/4 shapes, three times: level1-automatic-primitive-replay-72i8u7g7, full-level1-w6_3_6vt (20:30 UTC), full-level1-eghvyzqb (20:32 UTC) | Qwen candidate plus generic instruction legalizer; warm-start, no expert edits; excluded from cold-start counts |
| 2 | runs/verified-level2-locked-coeoi3p4/kernel.py | 0c6de407…74aa | 1.00, 4/4: live run + full-level2-uhd3faw3 | Cold-start Qwen, round 0, raw output (legalizer no-op), planner on |
| 6 | runs/verified-level6-locked-r79vct0e/kernel.py | 80e8a6a6…75e4 | 1.00, 4/4, traffic pass: live run + full-level6-kwp8cwhx | Cold-start Qwen, round 0, raw output (legalizer no-op) |
| 4 | runs/verified-level4-locked-n1kmsi6q/kernel.py | 94f06b9d…c0cb | 1.00, 4/4, traffic pass: live run + full-level4-_yi1ddsh | Cold-start Qwen, round 0, raw output |
| 5 | runs/verified-level5-locked-u5ybjn5a/kernel.py | e97f64c8…5dcb | 1.00, 4/4, traffic pass: live run + full-level5-gxsvzj8e | Cold-start Qwen, round 0, raw output |
| 7 | runs/verified-level7-locked-nriea9vh/kernel.py | a565ad7d…ffec | 1.00, 4/4, traffic pass: live run + full-level7-zun2gvdb | Cold-start Qwen, round 0 candidate 2, raw output |

Each replay used the unchanged checker in a private grade directory. Every shape passed numerics, input integrity, traffic and hardware-hazard checks. CPU simulation only; not device verified. Re-run: `python -B runs/verified-level1-locked-gowa0nvv/evaluate.py` (writes a new unique directory; asserts the locked sha256).

Solved levels: **7/8** = cold start **6/8 (Levels 2, 3, 4, 5, 6, 7)** + Level 1 by warm start (full agent run). Level 8 in progress. No matched repeated reliability gain or LoRA benchmark gain established. Separate Level 1 results: expert-edited warm diagnostic 1.00/4 cases; generic instruction-legalizer warm replay 1.00/4 cases, explicitly excluded from cold-start solve count.

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
