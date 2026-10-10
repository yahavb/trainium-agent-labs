# Failure taxonomy

80 attempts in 10 level-runs from 2 log file(s); 47 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 3 | 5/5 |
| 4 | 5/5 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L3 | L4 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|---|
| index arithmetic | `broadcast` | assigns a value of the wrong shape |  | 20 | 20 | 43% | 5 | 0% | ValueError: shape mismatch: value array of shape (65536,) could not be broadcast to indexing result of shape ( |
| tiling rules | `partition_over_128` | a tile or contraction larger than 128 partitions |  | 20 | 20 | 43% | 5 | 0% | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 `lhs_tile` and `rhs_tile` are bigger  |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) | 7 |  | 7 | 15% | 4 | - | AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data |

By family: **index arithmetic** 20 (43%), **tiling rules** 20 (43%), **memory model** 7 (15%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `broadcast` | `partition_over_128` | 5 |

## Where unsolved runs ended

(every run solved)

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 3 | VERIFIED | 5 |
| 4 | VERIFIED | 5 |

Calibration over 10 verdicts: Brier score 0.073; confident (>= 0.5) but wrong 0 time(s).

