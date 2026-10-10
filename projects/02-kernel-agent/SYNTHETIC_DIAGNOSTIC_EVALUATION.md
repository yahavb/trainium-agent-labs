# Synthetic diagnosis evaluation

Independent tasks: 12. Accepted mutations/verified restorations: 24. Rejected: 0. Train pairs: 22; held out: 2. No physical-device validation.

Added guidance only is scored; raw checker text is excluded. These predeclared semantic checks measure bug-specific actionable diagnosis coverage, not unrestricted accuracy or causal model benefit. Numerical cases without specific source evidence are scored incorrect/uncertain. Seven samples reviewed by this coding agent, not an independent human.

| Failure category | Cases | Legacy | Targeted |
|---|---:|---:|---:|
| DMA_SHAPE_MISMATCH | 1 | 1 | 1 |
| INVALID_API_ARGUMENT | 1 | 1 | 1 |
| INVALID_API_FUNCTION | 14 | 14 | 14 |
| INVALID_BUFFER_PLACEMENT | 1 | 1 | 1 |
| INVALID_TENSOR_DIMENSIONS | 5 | 1 | 5 |
| NUMERICAL_MISMATCH | 2 | 0 | 0 |

Total: legacy 18/24 (75.0%); targeted 22/24 (91.7%). Invalid API variants account for 14/24 cases, so do not extrapolate the overall percentage. Both numerical bugs remain undiagnosed specifically.

SDK inspection confirms nl.load(src, dtype=None) and nl.store(dst, value) exist. No absence rule was added. DMA mismatches receive actual transfer/allocation evidence before later matmul hypotheses; unresolved cases stay unresolved. Invalid keyword advice names the actual call, keyword and installed signature.

Artifacts: synthetic_nki/data_v2/diagnostic_evaluation_final.json; manual_review.json; train.jsonl; heldout.jsonl; summary.json. Original data/ corpus and earlier diagnostic assessments remain intact.
