# Failure taxonomy

44 attempts, levels [1, 2], 1 session(s): 20261010-154431

| failure mode | what it means | L1 | L2 | total | share |
|---|---|---|---|---|---|
| invented API name | called an nl/nisa function that does not exist | 16 |  | 16 | 36% |
| tile/slice size mismatch | source and destination hold different element counts | 4 | 4 | 8 | 18% |
| invented argument | passed a keyword the function does not take | 8 |  | 8 | 18% |
| read past the end | indexed beyond the tensor, usually a padded final tile |  | 8 | 8 | 18% |
| wrong buffer | operand or result in the wrong memory (sbuf / psum / hbm) | 4 |  | 4 | 9% |

## How the failure moved, round to round

9 repair steps: 4 repeated the same failure (44%), 5 changed it, and 0 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| tile/slice size mismatch | invented argument | 1 |
| invented argument | wrong buffer | 1 |
| wrong buffer | invented API name | 1 |
| invented API name | invented argument | 1 |
| tile/slice size mismatch | read past the end | 1 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 2 | 408 | 1581 | 0 | 0 | 0 | 508 | 730 | 274 |
| repair | 7 | 0 | 0 | 853 | 554 | 0 | 200 | 497 | 280 |
| repair+ledger | 2 | 0 | 0 | 877 | 660 | 651 | 269 | 756 | 286 |

## Line-located feedback

The checker quoted the failing line on 44 of 44 attempts that raised (100%).

## One real checker message per failure mode

**tile/slice size mismatch** (level 1, round 0, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384 The tile you allocated holds 16384 elements but you copied 4 into it. nisa.dma_copy does not slice or broadcast: allocate the destination with EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a bigger tensor, write t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then nisa.dma_copy(dst=t, src=a

**invented argument** (level 1, round 1, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving' Remove the `transpose_moving=` argument. The real signature is isa.nc_matmul(dst: 'NkiTensor', stationary: 'NkiTensor', moving: 'NkiTensor', is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=<matmul_perf_mode.none: 'none'>, name=None). The failing line is line 19 of your kernel: `

**wrong buffer** (level 1, round 2, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dst must be in ['psum'], got sbuf Allocate the `dst` tile with buffer=nl.psum instead of nl.sbuf. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving must both be in nl.sbuf. Copy between them with nisa.tensor_copy. The failing line is line 19 of your kernel: `nisa.nc_matmul(dst=tile2, stationary=tile, moving=tile, is_transpose=True)`. That is the line to change.

**invented API name** (level 1, round 3, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AttributeError: module 'nki.isa' has no attribute 'multiply' `nki.isa` has no `multiply`, and nothing similar exists. Its real names include: NkiInstruction, NkiValidationError, VirtualRegister, activate2, activation, activation_reduce, affine_select, bn_aggr, bn_stats, core_barrier, dge_mode, dma_compute, dma_copy, dma_engine, dma_transpose, dropout, engine, exponential, get_nc_sub_version, get_nc_version, gpsimd_engine, iota, local_gat

**read past the end** (level 2, round 1, reward 0.30)

> 0 of 4 shapes passed. On shape=(32, 12) as 3x4: raised AssertionError: Out-of-bound access for tensor `unnamed` on dimension 0: index 3 exceed dimension size of 3. The failing line is line 23 of your kernel: `tile[i, j] = tile[j, i]`. That is the line to change.

