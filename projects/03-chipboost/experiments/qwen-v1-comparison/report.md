# Agent comparison report

Status: **COMPLETE**; 72 recorded attempts.

| Arm | Runs complete/started | Attempts | Correct | Failed | Failure rate | Median best | Min best | Max best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| referee | 3/3 | 24 | 2 | 22 | 91.7% | 1.000000x | 1.000000x | 1.000000x |
| model_alone | 3/3 | 24 | 22 | 2 | 8.3% | 1.000000x | 1.000000x | 1.000000x |
| random_search | 3/3 | 24 | 24 | 0 | 0.0% | 3.125919x | 2.525760x | 3.323522x |

Best means maximum verified `faster` speedup per run, defaulting to 1x when no candidate improves. Correct means `slower`, `no_gain`, or `faster`; it does not imply held-out validation for every candidate. Failed means `rules`, `wrong`, or `heldout_fail`. Failure-rate denominator is recorded attempts. Held-out rejection rate uses only `heldout_fail` plus `faster` as its denominator.

| Arm | rules | wrong | heldout_fail | slower | no_gain | faster |
| --- | --- | --- | --- | --- | --- | --- |
| referee | 0 | 22 | 0 | 0 | 2 | 0 |
| model_alone | 0 | 2 | 0 | 0 | 22 | 0 |
| random_search | 0 | 0 | 0 | 2 | 0 | 22 |

Random search uses P2's expert-template cap prior. Agent arms start from the reference baseline. These results compare complete search setups; the three-arm comparison does not isolate feedback alone.

This batch remains pinned to P1 434e5f9, P2 919c6be, and P3 2ce9416. It predates the subsequent representative-feedback fix and does not evaluate that fix. Keep these results separate from any future batch using the corrected feedback implementation.

Infrastructure: 1 separately logged events; excluded from all attempt counts and failure rates.
