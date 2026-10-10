# Synthetic NKI experiments

These are sequential cold-start single-repeat experiments. Reward 1.00 requires the unchanged official checker. A disappearing first error is not proof that its constraint was fixed; successful repairs below require increased verified-shape coverage. CPU numerics do not establish physical-device correctness or speed.

| Run | Level | Arm | Best reward | Fully solved | Shapes verified | Candidates | AST unique/round | Prompt tokens | Completion tokens | Total tokens | Wall seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| controlled-20261010T184729-sye489_l | 3 | A_legacy_standard | 0.300 | 0/1 | 0 | 8 | 2.00 | 6056 | 2510 | 8566 | 116.5432258460005 |
| controlled-20261010T184729-sye489_l | 3 | B_targeted_standard | 0.300 | 0/1 | 0 | 8 | 2.50 | 6136 | 2537 | 8673 | 123.52420337900003 |

## Failure and repair evidence

- 3 / A_legacy_standard: {'DMA_SHAPE_MISMATCH': 5, 'INVALID_TENSOR_DIMENSIONS': 3}; truncated=0; verified-progress repairs=0; synthetic retrieval candidates=0; checker time=1.083s. Raw log: `runs/controlled-20261010T184729-sye489_l/level3/A_legacy_standard/attempts.jsonl`.
  Selected source transitions: [{"round": 1, "source_changed": false, "prior_failure": "DMA_SHAPE_MISMATCH", "new_failure": "DMA_SHAPE_MISMATCH", "prior_signature_no_longer_first": false, "verified_shapes_delta": 0}]
- 3 / B_targeted_standard: {'DMA_SHAPE_MISMATCH': 3, 'INVALID_TENSOR_DIMENSIONS': 4, 'UNKNOWN': 1}; truncated=0; verified-progress repairs=0; synthetic retrieval candidates=0; checker time=1.094s. Raw log: `runs/controlled-20261010T184729-sye489_l/level3/B_targeted_standard/attempts.jsonl`.
  Selected source transitions: [{"round": 1, "source_changed": true, "prior_failure": "DMA_SHAPE_MISMATCH", "new_failure": "INVALID_TENSOR_DIMENSIONS", "prior_signature_no_longer_first": true, "verified_shapes_delta": 0}]
