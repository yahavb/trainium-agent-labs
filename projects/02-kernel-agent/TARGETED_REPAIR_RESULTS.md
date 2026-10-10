# Synthetic NKI experiments

These are sequential cold-start single-repeat experiments. Reward 1.00 requires the unchanged official checker. A disappearing first error is not proof that its constraint was fixed; successful repairs below require increased verified-shape coverage. CPU numerics do not establish physical-device correctness or speed.

| Run | Level | Arm | Best reward | Fully solved | Shapes verified | Candidates | AST unique/round | Prompt tokens | Completion tokens | Total tokens | Wall seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| controlled-20261010T182312-kmg57ng4 | 3 | A_full | 1.000 | 1/1 | 1 | 4 | 4.00 | 2903 | 1493 | 4396 | 75.21348284400119 |
| controlled-20261010T182312-kmg57ng4 | 3 | B_shape_aware | 0.300 | 0/1 | 0 | 16 | 1.75 | 10967 | 5923 | 16890 | 276.3829310279998 |
| controlled-20261010T182312-kmg57ng4 | 4 | A_full | 0.625 | 0/1 | 1 | 16 | 2.25 | 11131 | 6017 | 17148 | 284.02514547099963 |
| controlled-20261010T182312-kmg57ng4 | 4 | B_shape_aware | 0.300 | 0/1 | 0 | 16 | 2.50 | 12659 | 7899 | 20558 | 363.89034093900045 |

## Failure and repair evidence

- 3 / A_full: {'INVALID_TENSOR_DIMENSIONS': 3}; truncated=0; verified-progress repairs=0; synthetic retrieval candidates=0; checker time=1.093s. Raw log: `runs/controlled-20261010T182312-kmg57ng4/level3/A_full/attempts.jsonl`.
  Selected source transitions: []
- 3 / B_shape_aware: {'INVALID_BUFFER_PLACEMENT': 15, 'UNKNOWN': 1}; truncated=0; verified-progress repairs=0; synthetic retrieval candidates=0; checker time=1.118s. Raw log: `runs/controlled-20261010T182312-kmg57ng4/level3/B_shape_aware/attempts.jsonl`.
  Selected source transitions: [{"round": 1, "source_changed": false, "prior_failure": "INVALID_BUFFER_PLACEMENT", "new_failure": "INVALID_BUFFER_PLACEMENT", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}, {"round": 2, "source_changed": false, "prior_failure": "INVALID_BUFFER_PLACEMENT", "new_failure": "INVALID_BUFFER_PLACEMENT", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}, {"round": 3, "source_changed": false, "prior_failure": "INVALID_BUFFER_PLACEMENT", "new_failure": "INVALID_BUFFER_PLACEMENT", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}]
- 4 / A_full: {'INVALID_TENSOR_DIMENSIONS': 13, 'OUT_OF_BOUNDS': 1, 'INVALID_BUFFER_PLACEMENT': 2}; truncated=0; verified-progress repairs=0; synthetic retrieval candidates=0; checker time=4.360s. Raw log: `runs/controlled-20261010T182312-kmg57ng4/level4/A_full/attempts.jsonl`.
  Selected source transitions: [{"round": 1, "source_changed": false, "prior_failure": "INVALID_TENSOR_DIMENSIONS", "new_failure": "INVALID_TENSOR_DIMENSIONS", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}, {"round": 2, "source_changed": false, "prior_failure": "INVALID_TENSOR_DIMENSIONS", "new_failure": "INVALID_TENSOR_DIMENSIONS", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}, {"round": 3, "source_changed": false, "prior_failure": "INVALID_TENSOR_DIMENSIONS", "new_failure": "INVALID_TENSOR_DIMENSIONS", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}]
- 4 / B_shape_aware: {'UNKNOWN': 3, 'INVALID_BUFFER_PLACEMENT': 13}; truncated=0; verified-progress repairs=0; synthetic retrieval candidates=0; checker time=3.855s. Raw log: `runs/controlled-20261010T182312-kmg57ng4/level4/B_shape_aware/attempts.jsonl`.
  Selected source transitions: [{"round": 1, "source_changed": true, "prior_failure": "INVALID_BUFFER_PLACEMENT", "new_failure": "INVALID_BUFFER_PLACEMENT", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}, {"round": 2, "source_changed": false, "prior_failure": "INVALID_BUFFER_PLACEMENT", "new_failure": "INVALID_BUFFER_PLACEMENT", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}, {"round": 3, "source_changed": false, "prior_failure": "INVALID_BUFFER_PLACEMENT", "new_failure": "INVALID_BUFFER_PLACEMENT", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}]

See SYNTHETIC_DIAGNOSTIC_EVALUATION.md for the newer 24-bug coverage assessment. Improved deterministic diagnoses are separate from live model correctness; the standard-generation smoke achieved no numerical passes in either arm.
