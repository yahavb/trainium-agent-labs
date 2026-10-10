# Failure taxonomy

676 attempts in 46 level-runs from 38 log file(s); 648 failed. Each attempt is classified by the checker's error, first match wins (`scripts/taxonomy.py`).

## Solve rate per level

| level | solved runs |
|---|---|
| 1 | 7/7 |
| 2 | 9/11 |
| 3 | 5/5 |
| 4 | 5/5 |
| 5 | 0/7 |
| 6 | 0/7 |
| 7 | 0/4 |

## Failure modes

*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried attempt had this mode, how often the next round's did too.

| family | mode | what it means | L1 | L2 | L3 | L4 | L5 | L6 | L7 | total | % of failures | runs | stuck | example |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| silent wrong numbers | `numeric_mismatch` | runs, numbers wrong | 9 | 8 |  |  | 83 | 33 | 62 | 195 | 30% | 30 | 45% | 1 of 4 shapes passed. On K=256 M=256 N=1024: NUMERICAL MISMATCH: worst error 3.42 of the output's RMS (15.98), |
| silent wrong numbers | `too_much_traffic` | correct but over the level's HBM byte bar |  |  |  |  | 40 | 72 |  | 112 | 17% | 14 | 80% | 3 of 4 shapes passed. On K=256 M=512 N=1024: CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving 2.00x th |
| silent wrong numbers | `non_finite` | NaN or Inf in the output | 3 | 6 |  |  | 9 |  |  | 18 | 3% | 9 | 40% | 1 of 4 shapes passed. On K=256 M=256 N=1024: NON-FINITE OUTPUT: 131072 NaN and 0 Inf, first at (0, 0). Usually |
| index arithmetic | `copy_size_mismatch` | copies between tiles of different sizes | 6 | 103 |  | 2 |  | 4 |  | 115 | 18% | 19 | 0% | AssertionError: dma_copy requires src and dst to have the same number of elements, got src=32768, dst=16384 `l |
| index arithmetic | `broadcast` | assigns a value of the wrong shape | 6 | 4 | 19 | 47 |  |  |  | 76 | 12% | 14 | 25% | ValueError: shape mismatch: value array of shape (32768,) could not be broadcast to indexing result of shape ( |
| index arithmetic | `axis_mismatch` | an operation given a tile with more axes than it takes | 16 |  |  |  |  |  |  | 16 | 2% | 6 | 0% | ValueError: input operand has more dimensions than allowed by the axis remapping |
| index arithmetic | `bad_access_pattern` | a strided .ap() view that does not fit the tile | 7 |  |  |  |  |  |  | 7 | 1% | 4 | - | AssertionError: ap() pattern has invalid partition stride. Partition step 64 must equal tensor free dimension  |
| index arithmetic | `out_of_bounds` | indexes past the end of a tensor | 1 | 4 |  |  |  |  |  | 5 | 1% | 5 | - | AssertionError: Out-of-bound access for tensor `unnamed` on dimension 1: index 3 exceed dimension size of 3. |
| index arithmetic | `transpose_whole_input` | level 2: transposes all of x, rows included, instead of each row's F1-by-F2 block |  | 4 |  |  |  |  |  | 4 | 1% | 2 | 33% | AssertionError: Partition dim size must be preserved, got 1 -> 3 |
| index arithmetic | `wrong_output_shape` | returns the wrong output shape |  | 4 |  |  |  |  |  | 4 | 1% | 1 | 100% | 0 of 4 shapes passed. On shape=(32, 12) as 3x4: WRONG SHAPE: returned (), reference is (32, 12). Check the out |
| index arithmetic | `reshape` | reshapes instead of slicing (or changes the partition size) |  | 1 |  |  |  |  |  | 1 | 0% | 1 | 0% | ValueError: cannot reshape array of size 12 into shape (3,1) |
| other | `other` |  | 15 | 2 |  |  |  | 2 |  | 19 | 3% | 9 | 0% | NameError: name 'psum_tile' is not defined |
| API | `invented_name` | calls an NKI function or attribute that does not exist | 3 | 8 |  |  |  | 6 | 1 | 18 | 3% | 12 | 17% | AttributeError: module 'nki.isa' has no attribute 'tensor_fill' `nki.isa` has no `tensor_fill`. |
| API | `wrong_signature` | right function, wrong arguments | 4 | 2 |  |  |  |  |  | 6 | 1% | 5 | 0% | TypeError: sum() got an unexpected keyword argument 'dst' |
| rules / format | `truncated` | the answer hit max_tokens and was cut off (finish=length); the checker then sees broken code | 11 | 1 |  |  |  |  |  | 12 | 2% | 7 | - | Your previous answer was cut off at the token limit before the code was complete. Reply with a shorter kernel: |
| rules / format | `rule_violation` | banned call, missing @nki.jit or wrong entry name |  | 3 |  |  |  |  |  | 3 | 0% | 3 | - | Rule violations, which score zero however fast the kernel is. |
| memory model | `wrong_buffer` | a tile in the wrong memory (SBUF / PSUM / HBM) | 2 |  | 3 |  |  | 6 | 1 | 12 | 2% | 10 | 0% | AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data |
| memory model | `python_op_on_tile` | Python arithmetic (+=, *) on a tile | 8 |  |  |  |  |  |  | 8 | 1% | 5 | - | TypeError: unsupported operand type(s) for /: 'NkiTensor' and 'int' Python operators like `/` do not work on t |
| tiling rules | `partition_over_128` | a tile or contraction larger than 128 partitions |  |  |  | 5 |  | 2 |  | 7 | 1% | 7 | 0% | AssertionError: dma_copy dst partition dimension 512 exceeds maximum 128 |
| tiling rules | `tile_1d` | a 1-D tile (every SBUF/PSUM tile needs 2 dims) | 6 |  |  | 1 |  |  |  | 7 | 1% | 5 | 0% | AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) |
| tiling rules | `illegal_allocation` | runs in the simulator, but allocates a tile the chip cannot hold |  |  |  |  |  | 3 |  | 3 | 0% | 3 | - | 0 of 4 shapes passed. On K=128 M=128 N=512: ILLEGAL ON HARDWARE: the CPU simulator ran this kernel, but it doe |

By family: **silent wrong numbers** 325 (50%), **index arithmetic** 228 (35%), **API** 24 (4%), **memory model** 20 (3%), **other** 19 (3%), **tiling rules** 17 (3%), **rules / format** 15 (2%)

## Fixed one thing, broke another

Round-to-round changes in the carried attempt's failure mode (not counting a solve).

| was | became | times |
|---|---|---|
| `numeric_mismatch` | `too_much_traffic` | 9 |
| `too_much_traffic` | `numeric_mismatch` | 8 |
| `broadcast` | `partition_over_128` | 5 |
| `axis_mismatch` | `other` | 2 |
| `broadcast` | `other` | 2 |
| `invented_name` | `non_finite` | 2 |
| `tile_1d` | `other` | 2 |
| `numeric_mismatch` | `broadcast` | 1 |
| `numeric_mismatch` | `other` | 1 |
| `copy_size_mismatch` | `transpose_whole_input` | 1 |
| `invented_name` | `transpose_whole_input` | 1 |
| `transpose_whole_input` | `invented_name` | 1 |

## Where unsolved runs ended

| level | last failure | runs |
|---|---|---|
| 2 | `numeric_mismatch` | 1 |
| 2 | `wrong_output_shape` | 1 |
| 5 | `too_much_traffic` | 7 |
| 6 | `too_much_traffic` | 7 |
| 7 | `numeric_mismatch` | 4 |

12 attempt(s) were cut off by the token budget (finish=length).

## Held-out check

After the loop, each level's best kernel ran once on shapes and values it never saw (`nkibench.py --eval`). Confidence was stated before that check.

| level | verdict | runs |
|---|---|---|
| 1 | VERIFIED | 7 |
| 2 | NOT SOLVED | 2 |
| 2 | VERIFIED | 9 |
| 3 | PASSES THE LOOP'S SHAPES ONLY | 1 |
| 3 | VERIFIED | 4 |
| 4 | VERIFIED | 5 |
| 5 | NOT SOLVED | 6 |
| 6 | NOT SOLVED | 6 |
| 7 | NOT SOLVED | 4 |

Solved on the loop's shapes, failed held-out (verdicts keep the first 8 failing cases per kernel):

| level | value kind | mode | cases |
|---|---|---|---|
| 3 | float16 | `other` | 1 |

Calibration over 44 verdicts: Brier score 0.026; confident (>= 0.5) but wrong 1 time(s).

