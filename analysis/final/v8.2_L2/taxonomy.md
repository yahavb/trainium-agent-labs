# Failure taxonomy

192 attempts in 9 level-runs from 5 log file(s); 186 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 2 | 5/9 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L2 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes | 132 | 132 | 71% | 9 | 11% | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=384, dst=16384 `til |
| index arithmetic | `out_of_bounds` | indexes past the end of a tensor | 3 | 3 | 2% | 3 | - | AssertionError: Out-of-bound access for tensor `unnamed` on dimension 1: index range [0, 127] exceed dimension |
| index arithmetic | `broadcast` | assigns a value of the wrong shape | 2 | 2 | 1% | 1 | 50% | ValueError: shape mismatch: value array of shape (4,) could not be broadcast to indexing result of shape (1,) |
| index arithmetic | `reshape` | reshapes instead of slicing (or changes the partition size) | 1 | 1 | 1% | 1 | 0% | ValueError: cannot reshape array of size 12 into shape (3,1) |
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 26 | 26 | 14% | 7 | 78% | 0 of 4 shapes passed. On shape=(32, 12) as 3x4: NUMERICAL MISMATCH: worst error 4.21 of the output's RMS (1.01 |
| silent wrong numbers | `non_finite` | NaN or Inf in the output | 4 | 4 | 2% | 1 | 67% | 0 of 4 shapes passed. On shape=(32, 12) as 3x4: NON-FINITE OUTPUT: 256 NaN and 0 Inf, first at (0, 4). Usually |
| rules / format | `rule_violation` | banned call, missing @nki.jit or wrong entry name | 8 | 8 | 4% | 5 | - | Rule violations, which score zero however fast the kernel is. |
| rules / format | `truncated` | the answer hit max_tokens and was cut off (finish=length); the checker then sees broken code | 1 | 1 | 1% | 1 | - | Your previous answer was cut off at the token limit before the code was complete. Reply with a shorter kernel: |
| API | `invented_name` | calls an NKI function or attribute that does not exist | 4 | 4 | 2% | 3 | 33% | AttributeError: module 'nki.isa' has no attribute 'fill' `nki.isa` has no `fill`, and nothing similar exists.  |
| API | `wrong_signature` | right function, wrong arguments | 2 | 2 | 1% | 2 | 0% | TypeError: NkiTensor.reshape() takes 2 positional arguments but 4 were given |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) | 2 | 2 | 1% | 1 | 50% | AssertionError: memset dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.memset` only moves data between o |
| other | `other` |  | 1 | 1 | 1% | 1 | - | AssertionError: dma_copy requires HBM or SBUF tensors, got src=MemoryRegion.psum, dst=MemoryRegion.shared_hbm  |

By family: **index arithmetic** 138 (74%), **silent wrong numbers** 30 (16%), **rules / format** 9 (5%), **API** 6 (3%), **memory model** 2 (1%), **other** 1 (1%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `copy_size_mismatch` | `numeric_mismatch` | 6 |
| `copy_size_mismatch` | `non_finite` | 1 |
| `non_finite` | `invented_name` | 1 |
| `invented_name` | `wrong_buffer` | 1 |
| `wrong_buffer` | `non_finite` | 1 |
| `invented_name` | `wrong_signature` | 1 |
| `wrong_signature` | `reshape` | 1 |
| `reshape` | `broadcast` | 1 |
| `broadcast` | `copy_size_mismatch` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 2 | `numeric_mismatch` | 3 |
| 2 | `non_finite` | 1 |

1 attempt(s) were cut off by the token budget (finish=length).

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 2 | NOT SOLVED | 4 |
| 2 | VERIFIED | 5 |

Calibration over 9 verdicts: Brier score 0.006; confident (>= 0.5) but wrong 0 time(s).

