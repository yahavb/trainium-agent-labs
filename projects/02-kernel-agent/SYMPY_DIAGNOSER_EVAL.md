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
