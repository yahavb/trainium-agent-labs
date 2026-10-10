# Failure taxonomy

184 attempts in 8 level-runs from 8 log file(s); 181 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 9 | 2/2 |
| 10 | 1/1 |
| 11 | 0/1 |
| 12 | 0/2 |
| 13 | 0/1 |
| 14 | 0/1 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L9 | L10 | L11 | L12 | L13 | L14 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memory model | `python_op_on_tile` | Python arithmetic (+=, *) on a tile |  | 1 |  | 50 |  |  | 51 | 28% | 3 | 0% | TypeError: unsupported operand type(s) for -: 'NkiTensor' and 'NkiTensor' Python operators like `-` do not wor |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) |  | 1 |  |  |  |  | 1 | 1% | 1 | - | AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data |
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes |  |  | 28 |  |  | 2 | 30 | 17% | 2 | 50% | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=8192, dst=65536 `ti |
| index arithmetic | `broadcast` | assigns a value of the wrong shape |  |  | 2 |  |  |  | 2 | 1% | 1 | 0% | ValueError: shape mismatch: value array of shape (8192,) could not be broadcast to indexing result of shape (6 |
| index arithmetic | `reshape` | reshapes instead of slicing (or changes the partition size) |  |  | 1 |  |  |  | 1 | 1% | 1 | 0% | ValueError: cannot reshape array of size 8192 into shape (128,512) |
| API | `invented_name` | calls an NKI function or attribute that does not exist |  |  | 1 |  | 5 | 22 | 28 | 15% | 3 | 33% | AttributeError: module 'nki.language' has no attribute 'clip' `nki.language` has no `clip`. |
| API | `wrong_signature` | right function, wrong arguments | 3 |  |  | 5 | 4 |  | 12 | 7% | 4 | 60% | TypeError: tensor_scalar() missing 1 required positional argument: 'operand0' |
| tiling rules | `partition_over_128` | a tile or contraction larger than 128 partitions | 14 | 8 |  |  |  |  | 22 | 12% | 3 | 0% | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 |
| other | `other` |  | 2 | 1 |  | 5 | 6 | 1 | 15 | 8% | 7 | 38% | Correct in the simulator on every shape, but the trn2 compiler rejects it: the chip has no such instruction, s |
| silent wrong numbers | `non_finite` | NaN or Inf in the output |  |  |  | 4 |  | 5 | 9 | 5% | 2 | 83% | 0 of 3 shapes passed. On R=128 C=64: NON-FINITE OUTPUT: 8192 NaN and 0 Inf, first at (0, 0). Usually an uninit |
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 3 |  |  |  | 5 |  | 8 | 4% | 3 | 100% | 0 of 3 shapes passed. On R=128 C=64: NUMERICAL MISMATCH: worst error 39.4 of the output's RMS (4.66), toleranc |
| rules / format | `truncated` | the answer hit max_tokens and was cut off (finish=length); the checker then sees broken code |  |  |  |  |  | 2 | 2 | 1% | 1 | - | Your previous answer was cut off at the token limit before the code was complete. Reply with a shorter kernel: |

By family: **memory model** 52 (29%), **API** 40 (22%), **index arithmetic** 33 (18%), **tiling rules** 22 (12%), **silent wrong numbers** 17 (9%), **other** 15 (8%), **rules / format** 2 (1%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `partition_over_128` | `other` | 3 |
| `copy_size_mismatch` | `broadcast` | 2 |
| `copy_size_mismatch` | `non_finite` | 1 |
| `non_finite` | `invented_name` | 1 |
| `invented_name` | `non_finite` | 1 |
| `invented_name` | `copy_size_mismatch` | 1 |
| `broadcast` | `reshape` | 1 |
| `reshape` | `copy_size_mismatch` | 1 |
| `python_op_on_tile` | `other` | 1 |
| `other` | `non_finite` | 1 |
| `python_op_on_tile` | `wrong_signature` | 1 |
| `wrong_signature` | `other` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 11 | `broadcast` | 1 |
| 12 | `non_finite` | 1 |
| 12 | `wrong_signature` | 1 |
| 13 | `numeric_mismatch` | 1 |
| 14 | `non_finite` | 1 |

2 attempt(s) were cut off by the token budget (finish=length).

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 9 | UNVERIFIED | 2 |
| 10 | UNVERIFIED | 1 |
| 11 | UNVERIFIED | 1 |
| 12 | UNVERIFIED | 2 |
| 13 | UNVERIFIED | 1 |
| 14 | UNVERIFIED | 1 |

