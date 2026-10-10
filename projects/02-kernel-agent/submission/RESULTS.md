# Feedback modes compared

From 11 log file(s): logs/c4a-directed4.jsonl, logs/directed.jsonl, logs/located-5runs.jsonl, logs/located.jsonl, logs/p1-directed3.jsonl, logs/p1-enriched.jsonl, logs/probe1-directed4.jsonl, logs/probe2-directed5.jsonl, logs/probe3-directed5.jsonl, logs/probe5-directed5.jsonl, logs/seat36-c4c-directed4.jsonl

Each row is one run of one level. A repair step is a round that answers a checker message. It *moved* if the score rose or the checker named a mistake not seen before in that run, *went back* if it returned to a mistake made earlier in the run, and was *stuck* if it repeated the mistake of the round before. Two messages count as the same mistake when they match after removing the quoted line and all numbers.

## Per level

| level | feedback | best score | solved | rounds | repair steps | moved | went back | stuck | different mistakes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | located | 0.30 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 1 | located | 0.30 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 1 | located | 0.30 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 1 | located | 0.30 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 1 | located | 0.30 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 1 | located | 0.30 | no | 8 | 7 | 4 | 0 | 3 | 5 |
| 1 | directed | 0.30 | no | 8 | 7 | 6 | 0 | 1 | 7 |
| 2 | enriched | 0.30 | no | 4 | 3 | 0 | 0 | 3 | 1 |
| 2 | located | 0.30 | no | 3 | 2 | 1 | 0 | 1 | 2 |
| 2 | located | 1.00 | round 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | located | 0.30 | no | 7 | 6 | 1 | 5 | 0 | 2 |
| 2 | located | 1.00 | round 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | located | 0.30 | no | 5 | 4 | 1 | 0 | 3 | 2 |
| 2 | located | 0.30 | no | 5 | 4 | 1 | 1 | 2 | 2 |
| 2 | directed | 1.00 | round 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | directed3 | 0.50 | no | 5 | 4 | 1 | 0 | 3 | 2 |
| 2 | directed4 | 0.50 | no | 6 | 5 | 1 | 0 | 4 | 2 |
| 3 | enriched | 0.30 | no | 6 | 5 | 1 | 0 | 4 | 2 |
| 3 | located | 0.30 | no | 6 | 5 | 2 | 0 | 3 | 3 |
| 3 | located | 0.30 | no | 5 | 4 | 1 | 0 | 3 | 2 |
| 3 | located | 0.30 | no | 7 | 6 | 1 | 5 | 0 | 2 |
| 3 | located | 0.30 | no | 7 | 6 | 1 | 5 | 0 | 2 |
| 3 | located | 0.30 | no | 6 | 5 | 2 | 0 | 3 | 3 |
| 3 | directed | 0.30 | no | 6 | 5 | 1 | 1 | 3 | 2 |
| 3 | directed3 | 0.30 | no | 6 | 5 | 2 | 0 | 3 | 3 |
| 3 | directed4 | 0.30 | no | 5 | 4 | 1 | 0 | 3 | 2 |
| 3 | directed5 | 0.30 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 3 | directed5 | 0.30 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 3 | directed5 | 0.30 | no | 1 | 0 | 0 | 0 | 0 | 1 |
| 4 | enriched | 0.62 | no | 4 | 3 | 0 | 0 | 3 | 1 |
| 4 | located | 0.75 | no | 6 | 5 | 2 | 0 | 3 | 3 |
| 4 | located | 0.30 | no | 5 | 4 | 1 | 0 | 3 | 2 |
| 4 | located | 0.62 | no | 8 | 7 | 4 | 2 | 1 | 5 |
| 4 | located | 0.75 | no | 6 | 5 | 2 | 0 | 3 | 3 |
| 4 | located | 0.75 | no | 5 | 4 | 2 | 0 | 2 | 3 |
| 4 | directed | 0.75 | no | 6 | 5 | 2 | 0 | 3 | 3 |
| 4 | directed3 | 0.62 | no | 6 | 5 | 2 | 0 | 3 | 3 |
| 4 | directed4 | 1.00 | round 4 | 5 | 4 | 4 | 0 | 0 | 4 |
| 4 | directed4 | 0.75 | no | 8 | 7 | 5 | 0 | 2 | 6 |
| 4 | directed4 | 0.62 | no | 1 | 0 | 0 | 0 | 0 | 1 |
| 4 | directed5 | 0.62 | no | 1 | 0 | 0 | 0 | 0 | 1 |

## Per run

| feedback | session | levels run | levels solved | repair steps | moved | went back | stuck | moved share |
|---|---|---|---|---|---|---|---|---|
| enriched | 20261010-195043 | 3 | 0 | 11 | 1 | 0 | 10 | 9% |
| located | 20261010-154431 | 2 | 0 | 9 | 6 | 0 | 3 | 67% |
| located | 20261010-163428 | 4 | 1 | 17 | 9 | 0 | 8 | 53% |
| located | 20261010-163428 | 4 | 0 | 21 | 8 | 5 | 8 | 38% |
| located | 20261010-163428 | 4 | 1 | 20 | 10 | 7 | 3 | 50% |
| located | 20261010-163428 | 4 | 0 | 22 | 9 | 5 | 8 | 41% |
| located | 20261010-163428 | 4 | 0 | 20 | 9 | 1 | 10 | 45% |
| directed | 20261010-161200 | 4 | 1 | 17 | 9 | 1 | 7 | 53% |
| directed3 | 20261010-193145 | 3 | 0 | 14 | 5 | 0 | 9 | 36% |
| directed4 | 20261010-200712 | 3 | 1 | 13 | 6 | 0 | 7 | 46% |
| directed4 | 20261010-203420 | 1 | 0 | 7 | 5 | 0 | 2 | 71% |
| directed4 | 20261010-204429 | 1 | 0 | 0 | 0 | 0 | 0 | n/a |
| directed5 | 20261010-202354 | 2 | 0 | 7 | 5 | 0 | 2 | 71% |
| directed5 | 20261010-205002 | 1 | 0 | 7 | 5 | 0 | 2 | 71% |
| directed5 | 20261010-205823 | 1 | 0 | 0 | 0 | 0 | 0 | n/a |
