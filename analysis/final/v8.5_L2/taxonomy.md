# Failure taxonomy

104 attempts in 6 level-runs from 6 log file(s); 99 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 2 | 4/6 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L2 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes | 67 | 67 | 68% | 6 | 43% | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=384, dst=16384 `til |
| index arithmetic | `out_of_bounds` | indexes past the end of a tensor | 5 | 5 | 5% | 3 | 0% | AssertionError: Out-of-bound access for tensor `unnamed` on dimension 1: index 3 exceed dimension size of 3. |
| index arithmetic | `transpose_whole_input` | level 2: transposes all of x, rows included, instead of each row's F1-by-F2 block | 2 | 2 | 2% | 2 | - | AssertionError: Partition dim size must be preserved, got 1 -> 3 |
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 11 | 11 | 11% | 5 | 50% | 0 of 4 shapes passed. On shape=(32, 12) as 3x4: NUMERICAL MISMATCH: worst error 4.49 of the output's RMS (1.01 |
| silent wrong numbers | `non_finite` | NaN or Inf in the output | 3 | 3 | 3% | 2 | 50% | 1 of 4 shapes passed. On shape=(32, 12) as 3x4: NON-FINITE OUTPUT: 96 NaN and 0 Inf, first at (0, 3). Usually  |
| rules / format | `rule_violation` | banned call, missing @nki.jit or wrong entry name | 4 | 4 | 4% | 3 | - | Rule violations, which score zero however fast the kernel is. |
| rules / format | `truncated` | the answer hit max_tokens and was cut off (finish=length); the checker then sees broken code | 3 | 3 | 3% | 3 | - | Your previous answer was cut off at the token limit before the code was complete. Reply with a shorter kernel: |
| API | `invented_name` | calls an NKI function or attribute that does not exist | 2 | 2 | 2% | 2 | 0% | AttributeError: 'NkiTensor' object has no attribute 'transpose' |
| other | `other` |  | 1 | 1 | 1% | 1 | - | AssertionError: dma_copy requires HBM or SBUF tensors, got src=MemoryRegion.psum, dst=MemoryRegion.shared_hbm  |
| tiling rules | `tile_1d` | a 1-D tile (every SBUF/PSUM tile needs 2 dims) | 1 | 1 | 1% | 1 | 0% | AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) |

By family: **index arithmetic** 74 (75%), **silent wrong numbers** 14 (14%), **rules / format** 7 (7%), **API** 2 (2%), **other** 1 (1%), **tiling rules** 1 (1%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `copy_size_mismatch` | `out_of_bounds` | 2 |
| `invented_name` | `non_finite` | 1 |
| `non_finite` | `numeric_mismatch` | 1 |
| `copy_size_mismatch` | `numeric_mismatch` | 1 |
| `out_of_bounds` | `copy_size_mismatch` | 1 |
| `copy_size_mismatch` | `tile_1d` | 1 |
| `tile_1d` | `copy_size_mismatch` | 1 |
| `invented_name` | `numeric_mismatch` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 2 | `copy_size_mismatch` | 1 |
| 2 | `numeric_mismatch` | 1 |

3 attempt(s) were cut off by the token budget (finish=length).

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 2 | NOT SOLVED | 2 |
| 2 | VERIFIED | 4 |

Calibration over 6 verdicts: Brier score 0.007; confident (>= 0.5) but wrong 0 time(s).

