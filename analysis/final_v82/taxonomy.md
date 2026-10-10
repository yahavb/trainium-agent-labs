# Failure taxonomy

180 attempts in 12 level-runs from 9 log file(s); 168 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 1 | 4/4 |
| 2 | 3/5 |
| 3 | 1/1 |
| 4 | 2/2 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L1 | L2 | L3 | L4 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|---|---|---|
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes | 1 | 68 |  |  | 69 | 41% | 6 | 0% | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=32, dst=1 |
| index arithmetic | `broadcast` | assigns a value of the wrong shape | 3 |  | 2 | 20 | 25 | 15% | 4 | 25% | ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (32,16,2) and  |
| index arithmetic | `axis_mismatch` | an operation given a tile with more axes than it takes | 7 |  |  |  | 7 | 4% | 3 | 0% | ValueError: input operand has more dimensions than allowed by the axis remapping |
| index arithmetic | `out_of_bounds` | indexes past the end of a tensor | 1 | 2 |  |  | 3 | 2% | 3 | - | AssertionError: Out-of-bound access for tensor `unnamed` on dimension 2: index 2 exceed dimension size of 2. |
| index arithmetic | `bad_access_pattern` | a strided .ap() view that does not fit the tile | 2 |  |  |  | 2 | 1% | 2 | - | AssertionError: ap() pattern has invalid partition stride. Partition step 64 must equal tensor free dimension  |
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 4 | 16 |  |  | 20 | 12% | 6 | 67% | 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: NUMERICAL MISMATCH: worst error 4.71 of the output's RMS ( |
| silent wrong numbers | `non_finite` | NaN or Inf in the output |  | 4 |  |  | 4 | 2% | 1 | 67% | 0 of 4 shapes passed. On shape=(32, 12) as 3x4: NON-FINITE OUTPUT: 256 NaN and 0 Inf, first at (0, 4). Usually |
| other | `other` |  | 8 |  |  |  | 8 | 5% | 3 | 0% | Correct in the simulator on every shape, but the trn2 compiler rejects it: the chip has no such instruction, s |
| rules / format | `truncated` | the answer hit max_tokens and was cut off (finish=length); the checker then sees broken code | 7 | 1 |  |  | 8 | 5% | 4 | - | Your previous answer was cut off at the token limit before the code was complete. Reply with a shorter kernel: |
| rules / format | `rule_violation` | banned call, missing @nki.jit or wrong entry name |  | 4 |  |  | 4 | 2% | 3 | - | Rule violations, which score zero however fast the kernel is. |
| memory model | `python_op_on_tile` | Python arithmetic (+=, *) on a tile | 5 |  |  |  | 5 | 3% | 3 | - | TypeError: unsupported operand type(s) for /: 'NkiTensor' and 'int' Python operators like `/` do not work on t |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) | 1 | 2 |  |  | 3 | 2% | 2 | 50% | AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data |
| API | `invented_name` | calls an NKI function or attribute that does not exist | 1 | 2 |  |  | 3 | 2% | 3 | 0% | AttributeError: module 'nki.language' has no attribute 'reshape' nl has no reshape. |
| API | `wrong_signature` | right function, wrong arguments | 1 | 1 |  |  | 2 | 1% | 2 | - | TypeError: sum() got an unexpected keyword argument 'dst' |
| tiling rules | `tile_1d` | a 1-D tile (every SBUF/PSUM tile needs 2 dims) | 3 |  |  |  | 3 | 2% | 2 | 0% | AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) |
| tiling rules | `partition_over_128` | a tile or contraction larger than 128 partitions |  |  |  | 2 | 2 | 1% | 2 | 0% | AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 `lhs_tile` and `rhs_tile` are bigger  |

By family: **index arithmetic** 106 (63%), **silent wrong numbers** 24 (14%), **rules / format** 12 (7%), **other** 8 (5%), **memory model** 8 (5%), **API** 5 (3%), **tiling rules** 5 (3%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `copy_size_mismatch` | `numeric_mismatch` | 3 |
| `broadcast` | `partition_over_128` | 2 |
| `axis_mismatch` | `other` | 1 |
| `tile_1d` | `other` | 1 |
| `numeric_mismatch` | `broadcast` | 1 |
| `broadcast` | `other` | 1 |
| `copy_size_mismatch` | `non_finite` | 1 |
| `non_finite` | `invented_name` | 1 |
| `invented_name` | `wrong_buffer` | 1 |
| `wrong_buffer` | `non_finite` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 2 | `non_finite` | 1 |
| 2 | `numeric_mismatch` | 1 |

8 attempt(s) were cut off by the token budget (finish=length).

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 1 | VERIFIED | 4 |
| 2 | NOT SOLVED | 2 |
| 2 | VERIFIED | 3 |
| 3 | VERIFIED | 1 |
| 4 | VERIFIED | 2 |

Calibration over 12 verdicts: Brier score 0.019; confident (>= 0.5) but wrong 0 time(s).

