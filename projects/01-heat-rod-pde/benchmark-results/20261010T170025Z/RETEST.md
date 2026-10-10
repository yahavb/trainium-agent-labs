# Default-agent retest

Real Qwen3-8B inference on seat-85. Experimental concise mode: False.
Both variants use the same eight problem configurations, two samples per round, three rounds, 512 output tokens per request, one calculator exchange per attempt and a 180-second case deadline. Full-score answers require the independent verifier.

| Agent | Validated solved | Timeouts | Service failures | Total seconds |
|---|---:|---:|---:|---:|
| baseline | 3/8 | 0 | 0 | 813.5 |
| improved | 3/8 | 1 | 0 | 687.2 |

Validated solved difference: +0 cases. Total measured time difference: -126.3s (-15.5%). Lower elapsed time alone does not demonstrate better mathematical answers.

| Problem | Baseline status | Improved status | Baseline seconds | Improved seconds |
|---|---|---|---:|---:|
| 0.1 seed 0 | solved | solved | 39.6 | 36.6 |
| 0.2 seed 0 | unsolved | solved | 128.2 | 70.2 |
| 0.3 seed 0 | solved | solved | 75.8 | 72.0 |
| 1.1 seed 0 | unsolved | budget_timeout | 120.5 | 180.0 |
| 1.2 seed 0 | unsolved | unsolved | 120.5 | 120.0 |
| 1.3 seed 0 | unsolved | unsolved | 103.9 | 57.9 |
| 1.3 seed 1 | unsolved | unsolved | 125.9 | 86.1 |
| 1.3 seed 2 | solved | unsolved | 99.1 | 64.4 |

Limits: a single paired run per problem configuration, not repeated model seeds; stochastic generation, sequential order, possible cache and load effects. Timeouts are inconclusive, not zero mathematical rewards. The earlier benchmark measured concise mode and is a separate experiment. Nine regression tests passed again before this retest. Evidence is comparison.json plus all case logs and attempt traces in this directory.
