# Failure taxonomy

424 attempts in 20 level-runs from 1 log file(s); 420 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 1 | 0/5 |
| 2 | 3/5 |
| 3 | 0/5 |
| 4 | 0/5 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L1 | L2 | L3 | L4 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|---|---|---|
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes | 40 | 30 | 26 |  | 96 | 23% | 15 | 62% | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384 |
| index arithmetic | `out_of_bounds` | indexes past the end of a tensor |  | 32 | 16 | 16 | 64 | 15% | 5 | 100% | AssertionError: Out-of-bound access for tensor `unnamed` on dimension 0: index 3 exceed dimension size of 3. |
| index arithmetic | `reshape` | reshapes instead of slicing (or changes the partition size) |  | 1 | 48 | 1 | 50 | 12% | 5 | 100% | ValueError: cannot reshape array of size 32768 into shape (1,64) |
| index arithmetic | `broadcast` | assigns a value of the wrong shape |  |  |  | 9 | 9 | 2% | 5 | - | ValueError: shape mismatch: value array of shape (65536,) could not be broadcast to indexing result of shape ( |
| API | `invented_name` | calls an NKI function or attribute that does not exist | 80 |  |  | 1 | 81 | 19% | 6 | 100% | AttributeError: module 'nki.isa' has no attribute 'multiply' `nki.isa` has no `multiply`, and nothing similar  |
| API | `wrong_signature` | right function, wrong arguments | 20 | 1 |  |  | 21 | 5% | 6 | 0% | TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving' |
| tiling rules | `partition_over_128` | a tile or contraction larger than 128 partitions |  |  |  | 55 | 55 | 13% | 4 | 100% | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 |
| tiling rules | `tile_1d` | a 1-D tile (every SBUF/PSUM tile needs 2 dims) |  |  | 22 |  | 22 | 5% | 4 | 50% | AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) | 20 |  |  | 1 | 21 | 5% | 6 | 0% | AssertionError: dst must be in ['psum'], got sbuf |
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong |  |  |  | 1 | 1 | 0% | 1 | 0% | 0 of 4 shapes passed. On K=128 M=128 N=512: NUMERICAL MISMATCH: worst error 324 of the output's RMS (11.32), t |

By family: **index arithmetic** 219 (52%), **API** 102 (24%), **tiling rules** 77 (18%), **memory model** 21 (5%), **silent wrong numbers** 1 (0%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `copy_size_mismatch` | `wrong_signature` | 5 |
| `wrong_signature` | `wrong_buffer` | 5 |
| `wrong_buffer` | `invented_name` | 5 |
| `tile_1d` | `reshape` | 3 |
| `copy_size_mismatch` | `out_of_bounds` | 3 |
| `numeric_mismatch` | `out_of_bounds` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 1 | `invented_name` | 5 |
| 2 | `out_of_bounds` | 2 |
| 3 | `reshape` | 3 |
| 3 | `out_of_bounds` | 1 |
| 3 | `copy_size_mismatch` | 1 |
| 4 | `partition_over_128` | 4 |
| 4 | `out_of_bounds` | 1 |

