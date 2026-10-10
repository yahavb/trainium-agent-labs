# Failure taxonomy

84 attempts in 3 level-runs from 3 log file(s); 84 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 5 | 0/1 |
| 6 | 0/1 |
| 7 | 0/1 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L5 | L6 | L7 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|---|---|
| index arithmetic | `broadcast` | assigns a value of the wrong shape | 15 | 24 | 19 | 58 | 69% | 3 | 0% | ValueError: shape mismatch: value array of shape (65536,) could not be broadcast to indexing result of shape ( |
| index arithmetic | `out_of_bounds` | indexes past the end of a tensor |  | 3 |  | 3 | 4% | 1 | 50% | AssertionError: Out-of-bound access for tensor `unnamed` on dimension 0: index range [0, 511] exceed dimension |
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes | 1 |  | 1 | 2 | 2% | 2 | - | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=16384, dst=65536 |
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 4 | 1 | 4 | 9 | 11% | 3 | 71% | 1 of 4 shapes passed. On K=256 M=256 N=1024: NUMERICAL MISMATCH: worst error 3.42 of the output's RMS (15.98), |
| silent wrong numbers | `too_much_traffic` | correct but over the level's HBM byte bar | 1 | 2 | 1 | 4 | 5% | 3 | 0% | 3 of 4 shapes passed. On K=256 M=512 N=1024: CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving 2.00x th |
| tiling rules | `partition_over_128` | a tile or contraction larger than 128 partitions | 1 | 2 | 1 | 4 | 5% | 3 | 0% | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 `lhsT_tile` and `rhs_tile` are bigger |
| API | `invented_name` | calls an NKI function or attribute that does not exist | 2 |  |  | 2 | 2% | 1 | - | AttributeError: module 'nki.language' has no attribute 'tile_shape' `nki.language` has no `tile_shape`. |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) |  |  | 2 | 2 | 2% | 1 | 0% | AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data |

By family: **index arithmetic** 63 (75%), **silent wrong numbers** 13 (15%), **tiling rules** 4 (5%), **API** 2 (2%), **memory model** 2 (2%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `partition_over_128` | `too_much_traffic` | 4 |
| `too_much_traffic` | `numeric_mismatch` | 3 |
| `numeric_mismatch` | `wrong_buffer` | 1 |
| `wrong_buffer` | `numeric_mismatch` | 1 |
| `broadcast` | `partition_over_128` | 1 |
| `numeric_mismatch` | `out_of_bounds` | 1 |
| `out_of_bounds` | `partition_over_128` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 5 | `numeric_mismatch` | 1 |
| 6 | `too_much_traffic` | 1 |
| 7 | `numeric_mismatch` | 1 |

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 5 | NOT SOLVED | 1 |
| 6 | NOT SOLVED | 1 |
| 7 | NOT SOLVED | 1 |

Calibration over 3 verdicts: Brier score 0.000; confident (>= 0.5) but wrong 0 time(s).

