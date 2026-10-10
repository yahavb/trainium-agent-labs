# Level scorecard

**Headline:** Our autonomous NKI agent (Qwen3-8B + planner + automatic legalizer) produced simulator-verified 1.00 kernels on Levels 1–7 of 8 (Level 1 via warm start). Each kernel is locked and re-verified with the unchanged checker.

## Current commit: generated planner + legalizer, base Qwen3-8B on Trainium2 (as of 21:50 UTC)

| Level | Score | Shapes | Run |
|---|---:|---:|---|
| 1 Average pooling | **1.00** (warm start) | 4/4
| 2 Transpose | **1.00** (cold, round 0) | 4/4 |
| 3 Single-tile matmul | **1.00** (cold, round 0) | 1/1 |
| 4 Tiled matmul | **1.00**\* | 4/4 |
| 5 Matmul, hoisted loads | **1.00**\* | 4/4 |
| 6 Matmul, M/N blocked | **1.00**\* | 4/4 | 
| 7 Matmul, M/N/K blocked | **1.00**\* | 4/4 | 
| 8 Single-head attention | running |
