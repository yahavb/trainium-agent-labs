# Symbolic diagnosis evaluation

Actual simulator-verified mutations: 24, from twelve independent clean kernels. Held-out records are evaluated offline but never indexed. Raw errors are excluded from guidance scoring.

| Policy | Root-cause coverage |
|---|---:|
| Existing legacy guidance | 18/24 |
| Targeted guidance | 22/24 |
| Targeted plus SymPy | 22/24 |

SymPy alone directly detected six supported shape causes. Five analyses returned UNKNOWN; absent shape violations on an API/numerical bug do not indicate kernel legality. Twelve simulator-verified clean kernels produced zero static false-positive violations. These finite examples do not prove a zero population false-positive rate. Total first-pass symbolic analysis time: 1.0056768389968056 seconds (includes warm-up).

Every emitted source location corresponds to an actual allocation/call AST line. This is syntactic localization evidence, not independent runtime trace confirmation. The six detected shape causes were reviewed against the actual mutation and simulator error; generic numerical causes remain unconfirmed. No independent human annotation was obtained.

Representative review: DMA mismatch counts 6 versus 8; explicit rank-1 tile; sum/max rank loss; PSUM (1,2) versus mathematical (2,3); partition tile 256 >128. Matmul destination is checked by axis, not output element count. A separate symbolic instantiation K=3,M=2,N=4 plus offset passed clean/restored NumPy checks and failed after wrong PSUM mutation. It is not an official benchmark candidate or retrieval record.

Interpretation: current corpus does not show additional diagnosis coverage beyond the integer AST analyzer. Symbolic algebra expands supported expressions and retains uncertainty; live correctness improvement remains to be measured. Invalid-API variants dominate this corpus, and the two numerical causes are not specifically diagnosed. See SYNTHETIC_DIAGNOSTIC_EVALUATION.md for per-category counts and primary-task/macro results.

Artifacts: synthetic_nki/data_v2/symbolic_evaluation.json; synthetic_nki/symbolic_data/record.json. Reproduction commands are in SYMPY_NKI_DESIGN.md.

## Verified curriculum extension

Original datasets remain intact. Default retrieval now uses data_v4: 16 distinct clean kernels, 28 verified repair pairs, 26 train and 2 held-out pairs. Added independent offset transpose, grouped sum, row-output tiling and ragged K accumulation. The initial extension rejected one boundary mutation with a wrongly anticipated DMA category; actual OUT_OF_BOUNDS feedback was retained in data_v3/summary.json and the expectation was corrected/reverified. No additional model calls and no device validation.

Final 28-pair guidance check: legacy 20/28, targeted 25/28, targeted plus symbolic 25/28. Symbolic constraints detect nine injected shape/layout causes, with zero false-positive violations on sixteen verified clean kernels. The three numerical bugs remain unconfirmed specifically. All emitted source locations refer to actual AST calls/allocations; this is not independent runtime localization proof. Current concrete corpus has no unresolved supported-shape analyses after bounded loop reasoning; API and numerical legality remain outside this status. Exact timings and records: synthetic_nki/data_v4/symbolic_evaluation_final.json and diagnostic_evaluation.json.


## Operation-aware primitive and LoRA update (2026-10-10)

Latest evidence is summarized in LEVEL_SCORECARD.md, PRIMITIVE_CORRECTNESS_SPRINT.md, LORA_TRAINING_REPORT.md and LORA_COMPARISON.md. Full untuned L1/L2 remained .30, zero numerical cases. Current cold L1 planner/legalizer has a provisional .50 execution-only candidate, zero shapes; no cold solve claimed. A generic deterministic instruction-legalizer warm replay reached L1 1.00, 4/4 cases, separately labeled and excluded from cold-start rates/training. L3 remains 1.00; L4 .625 (1/4). No hardware performance measured.

Corpus data_v8: 21 independently simulator/NumPy-verified clean kernels, 38 verified error/restoration pairs, 36 training/retrieval candidates and two held-out records. Existing retrieval and actual LoRA training remain on frozen data_v4; later data did not enter the running training. Diagnosis rubric on 38 actual injected failures: legacy 21 correct, targeted 30, targeted+SymPy 30; ten supported symbolic causes, four UNKNOWN, zero clean static false-positive kernels among 21. These are task-specific offline checks, not general diagnostic accuracy or proof of live improvement.

LoRA has completed 82 optimizer steps on 41 examples (15 generation, 26 repair), with three held-out examples from one completely excluded family. Rank 8/alpha 16, q_proj/v_proj, lr1e-4, completion-only loss, batch1; CPU training avoided Neuron device resources. Held-out mean example loss .9951169491 -> .1274786662 is not correctness evidence. Adapter saved in runs/qwen3-nki-lora-mdzxuq1k/training/adapter. The preserved full-adapter evaluation waits for active cold-run completion; a matched original/LoRA x legacy/planner study follows sequentially on the same isolated CPU backend. New held-out repairs are queued afterwards. Shared Qwen/Neuron vLLM was not restarted or modified.

SDK-verified miniature compositions exercise access-pattern reduction/scalar scaling, free-axis permutation, partition/free transpose and explicit K accumulation. Optional --primitive-policy legalize corrects narrowly verified instruction/opcode namespaces and known HBM scalar-result staging; it never supplies an algorithm or benchmark-specific code. Original generated source and transformed source are both recorded. Original grading/references/shapes/tolerances/hazards remain unchanged.
