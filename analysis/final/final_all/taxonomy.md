# Failure taxonomy

88 attempts in 4 level-runs from 1 log file(s); 85 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 1 | 1/1 |
| 2 | 1/1 |
| 3 | 0/1 |
| 4 | 1/1 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L1 | L2 | L3 | L4 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|---|---|---|
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes | 2 | 23 | 4 |  | 29 | 34% | 3 | 83% | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=32, dst=8192 |
| index arithmetic | `broadcast` | assigns a value of the wrong shape | 5 |  | 11 | 10 | 26 | 31% | 3 | 60% | ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (32,16,2) and  |
| index arithmetic | `axis_mismatch` | an operation given a tile with more axes than it takes | 8 |  |  |  | 8 | 9% | 1 | 0% | AssertionError: tensor_reduce axis must be the last contiguous dim(s) of the tile. For a 3D tensor, expected a |
| index arithmetic | `out_of_bounds` | indexes past the end of a tensor |  | 5 |  |  | 5 | 6% | 1 | 67% | AssertionError: Out-of-bound access for tensor `unnamed` on dimension 0: index 1 exceed dimension size of 1. |
| other | `other` |  | 3 | 1 |  |  | 4 | 5% | 2 | 0% | TypeError: object of type 'int' has no len() |
| rules / format | `truncated` | the answer hit max_tokens and was cut off (finish=length); the checker then sees broken code | 3 | 1 |  |  | 4 | 5% | 2 | - | Your previous answer was cut off at the token limit before the code was complete. Reply with a shorter kernel: |
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 2 | 1 |  |  | 3 | 4% | 2 | 0% | 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: NUMERICAL MISMATCH: worst error 5.22 of the output's RMS ( |
| memory model | `python_op_on_tile` | Python arithmetic (+=, *) on a tile | 2 |  |  |  | 2 | 2% | 1 | - | TypeError: unsupported operand type(s) for +=: 'float' and 'NkiTensor' |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) |  |  | 1 |  | 1 | 1% | 1 | - | AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data |
| tiling rules | `tile_1d` | a 1-D tile (every SBUF/PSUM tile needs 2 dims) | 2 |  |  |  | 2 | 2% | 1 | - | AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) |
| tiling rules | `partition_over_128` | a tile or contraction larger than 128 partitions |  |  |  | 1 | 1 | 1% | 1 | 0% | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 `lhs_tile` and `rhs_tile` are bigger  |

By family: **index arithmetic** 68 (80%), **other** 4 (5%), **rules / format** 4 (5%), **silent wrong numbers** 3 (4%), **memory model** 3 (4%), **tiling rules** 3 (4%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `axis_mismatch` | `broadcast` | 1 |
| `broadcast` | `other` | 1 |
| `copy_size_mismatch` | `out_of_bounds` | 1 |
| `out_of_bounds` | `numeric_mismatch` | 1 |
| `broadcast` | `partition_over_128` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 3 | `copy_size_mismatch` | 1 |

4 attempt(s) were cut off by the token budget (finish=length).

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 1 | VERIFIED | 1 |
| 2 | VERIFIED | 1 |
| 3 | NOT SOLVED | 1 |
| 4 | VERIFIED | 1 |

Calibration over 4 verdicts: Brier score 0.007; confident (>= 0.5) but wrong 0 time(s).

