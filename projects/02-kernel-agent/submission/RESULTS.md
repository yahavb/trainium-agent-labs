# Feedback modes compared

From 2 log file(s): logs/located.jsonl, logs/directed.jsonl

Each row is one run of one level. A repair step is a round that answers a checker message. It *moved* if the score rose or the checker named a different mistake next time, and was *stuck* if the same mistake came back. Two messages count as the same mistake when they match after removing the quoted line and all numbers.

## Per level

| level | feedback | best score | solved | rounds | repair steps | moved | stuck | different mistakes |
|---|---|---|---|---|---|---|---|---|
| 1 | located | 0.30 | no | 8 | 7 | 5 | 2 | 6 |
| 1 | directed | 0.30 | no | 8 | 7 | 6 | 1 | 7 |
| 2 | located | 0.30 | no | 3 | 2 | 1 | 1 | 2 |
| 2 | directed | 1.00 | round 0 | 1 | 0 | 0 | 0 | 0 |
| 3 | directed | 0.30 | no | 6 | 5 | 2 | 3 | 2 |
| 4 | directed | 0.75 | no | 6 | 5 | 2 | 3 | 3 |

## Per run

| feedback | session | levels run | levels solved | repair steps | moved | stuck | moved share |
|---|---|---|---|---|---|---|---|
| located | 20261010-154431 | 2 | 0 | 9 | 6 | 3 | 67% |
| directed | 20261010-161200 | 4 | 1 | 17 | 10 | 7 | 59% |
