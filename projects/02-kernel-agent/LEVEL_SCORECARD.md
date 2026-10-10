# Level scorecard

Observed results on seat-265 only. Original five-repeat baseline: zero of five solved on each level. Official checker unchanged. No device-verified kernels.

| Level | Operation | Best cold-start agent reward | Shapes | Evidence / status (updated 20:58 UTC) |
|---|---|---:|---:|---|
| 1 | Average pooling | 0.50 | 0/4 | Warm-start 1.00 locked (verified-level1-locked-gowa0nvv). Cold run with plan v2 + new legalizer rules active: controlled-20261010T205651-l9444j4q. Offline, the anonymous_tile_dataflow rule applied to a saved cold-start candidate from controlled-20261010T204737-vleoyh30 scores 1.00 (post-hoc; not a live solve). |
| 2 | Free-axis transpose within each partition | **1.00** | 4/4 | **Cold-start solve**, round 0, raw Qwen (legalizer no-op), planner on: controlled-20261010T203725-onnn7nmy; locked verified-level2-locked-coeoi3p4, replay 1.00. |
| 3 | Single-tile matmul | **1.00** | 1/1 | Four earlier cold-start successes (Codex runs). |
| 4 | Tiled matmul | 0.625 | 1/4 | Plan v1/v2 runs stopped at 0.625/0.30. Load-once plan v3 run active: controlled-20261010T205651-l9444j4q. |
| 5 | Matmul, loads hoisted (traffic <=1.6x) | pending | – | First live run active: controlled-20261010T205124-kcpiu5w7. |
| 6 | Matmul, M/N blocked (traffic <=1.25x) | **1.00** | 4/4 | **Cold-start solve**, round 0, raw Qwen (legalizer no-op), plan v3: controlled-20261010T205124-kcpiu5w7; locked verified-level6-locked-r79vct0e, replay 1.00, traffic passes every shape. |
| 7 | Matmul, M/N/K blocked (traffic <=1.05x) | pending | – | First live run active: controlled-20261010T205124-kcpiu5w7. |
| 8 | Single-head attention | pending | – | First live run active: controlled-20261010T205302-lg75w1b2. agent.grade crashed (KeyError 'M') on any correct L8 shape before the fix in this sprint; nkibench.py --check has the same latent bug (benchmark file left unchanged). |

Planner plans for L4–L8 now prescribe structure in prose: tile sizes, loop order, API roles, and for L8 the full softmax pipeline. Each structure was first simulator-checked at 1.00 with hand-written kernels, which are kept in a scratchpad outside the repo and never put in prompts. Solves with these plans are **plan-guided cold starts**, not unaided generation. Fine-tuned (LoRA) runs on L1–L4 are active on a separate CPU endpoint (runs/lora-fast-eval-20261010T203757-0pFD); no LoRA scores yet.

Locked simulator-verified kernels (warm-start or cold-start, preserved read-only):

| Level | Locked kernel | sha256 | Full official replays | Origin |
|---|---|---|---|---|
| 1 | runs/verified-level1-locked-gowa0nvv/kernel.py | 3443e646…abfe | 1.00, 4/4 shapes, three times: level1-automatic-primitive-replay-72i8u7g7, full-level1-w6_3_6vt (20:30 UTC), full-level1-eghvyzqb (20:32 UTC) | Qwen candidate plus generic instruction legalizer; warm-start, no expert edits; excluded from cold-start counts |
| 2 | runs/verified-level2-locked-coeoi3p4/kernel.py | 0c6de407…74aa | 1.00, 4/4: live run + full-level2-uhd3faw3 | Cold-start Qwen, round 0, raw output (legalizer no-op), planner on |
| 6 | runs/verified-level6-locked-r79vct0e/kernel.py | 80e8a6a6…75e4 | 1.00, 4/4, traffic pass: live run + full-level6-kwp8cwhx | Cold-start Qwen, round 0, raw output (legalizer no-op), load-once plan v3 |

Each replay used the unchanged checker in a private grade directory. Every shape passed numerics, input integrity, traffic and hardware-hazard checks. CPU simulation only; not device verified. Re-run: `python -B runs/verified-level1-locked-gowa0nvv/evaluate.py` (writes a new unique directory; asserts the locked sha256).

Cold-start solved levels so far: **3/8 (Levels 2, 3, 6)**, plus Level 1 warm-start. No matched repeated reliability gain or LoRA benchmark gain established. Separate Level 1 results: expert-edited warm diagnostic 1.00/4 cases; generic instruction-legalizer warm replay 1.00/4 cases, explicitly excluded from cold-start solve count.

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
