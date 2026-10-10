# Failure taxonomy

480 attempts, levels [1, 2, 3, 4], 1 session(s): 20261010-163428

| failure mode | what it means | L1 | L2 | L3 | L4 | total | share |
|---|---|---|---|---|---|---|---|
| tile/slice size mismatch | source and destination hold different element counts | 20 | 41 | 34 | 25 | 120 | 25% |
| invented API name | called an nl/nisa function that does not exist | 84 |  |  | 8 | 92 | 19% |
| read past the end | indexed beyond the tensor, usually a padded final tile |  | 32 | 40 |  | 72 | 15% |
| no tiling: M or N over the limit | K is chunked, but the output dimensions are not |  |  |  | 44 | 44 | 9% |
| invented argument | passed a keyword the function does not take | 36 |  |  |  | 36 | 8% |
| other: IndexError | an exception no rule above names yet |  |  | 32 |  | 32 | 7% |
| wrong buffer | operand or result in the wrong memory (sbuf / psum / hbm) | 20 |  |  |  | 20 | 4% |
| no tiling: K over 128 | contracted more than one tile's worth in one nc_matmul |  |  |  | 12 | 12 | 2% |
| 1-D tile | allocated an SBUF/PSUM tile with one dimension |  |  | 10 |  | 10 | 2% |
| no tiling: partition over 128 | one tile for the whole tensor |  |  |  | 10 | 10 | 2% |
| result does not fit its tile | an NKI call's result is a different size from its destination; the simulator reports it as a reshape error |  |  | 8 | 1 | 9 | 2% |
| other: AssertionError | an exception no rule above names yet |  |  |  | 8 | 8 | 2% |
| other: TypeError | an exception no rule above names yet |  |  |  | 8 | 8 | 2% |
| NaN or Inf | read an uninitialised tile |  |  |  | 4 | 4 | 1% |
| solved | correct on every shape |  | 2 |  |  | 2 | 0% |
| framework shortcut | called numpy/torch to do the whole job, or used @ / .T |  | 1 |  |  | 1 | 0% |

## How the failure moved, round to round

100 repair steps: 43 repeated the same failure (43%), 57 changed it, and 2 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| tile/slice size mismatch | read past the end | 12 |
| read past the end | tile/slice size mismatch | 10 |
| tile/slice size mismatch | invented argument | 5 |
| invented argument | wrong buffer | 5 |
| wrong buffer | invented API name | 5 |
| invented API name | invented argument | 4 |
| no tiling: partition over 128 | no tiling: K over 128 | 3 |
| no tiling: K over 128 | no tiling: M or N over the limit | 3 |
| 1-D tile | result does not fit its tile | 2 |
| result does not fit its tile | other: IndexError | 2 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 20 | 399 | 1581 | 0 | 0 | 0 | 506 | 716 | 334 |
| repair | 73 | 0 | 0 | 1064 | 584 | 0 | 200 | 579 | 359 |
| repair+ledger | 27 | 0 | 0 | 1064 | 506 | 484 | 267 | 733 | 347 |

## Line-located feedback

The checker quoted the failing line on 473 of 473 attempts that raised (100%).

## One real checker message per failure mode

**tile/slice size mismatch** (level 1, round 0, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384 The tile you allocated holds 16384 elements but you copied 4 into it. nisa.dma_copy does not slice or broadcast: allocate the destination with EXACTLY the shape of the slice you are moving. If you want a 128x512 piece of a bigger tensor, write t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then nisa.dma_copy(dst=t, src=a

**invented argument** (level 1, round 1, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving' Remove the `transpose_moving=` argument. The real signature is isa.nc_matmul(dst: 'NkiTensor', stationary: 'NkiTensor', moving: 'NkiTensor', is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=<matmul_perf_mode.none: 'none'>, name=None). The failing line is line 19 of your kernel: `

**wrong buffer** (level 1, round 2, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dst must be in ['psum'], got sbuf Allocate the `dst` tile with buffer=nl.psum instead of nl.sbuf. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving must both be in nl.sbuf. Copy between them with nisa.tensor_copy. The failing line is line 19 of your kernel: `nisa.nc_matmul(dst=tile2, stationary=tile, moving=tile, is_transpose=True)`. That is the line to change.

**invented API name** (level 1, round 3, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AttributeError: module 'nki.isa' has no attribute 'multiply' `nki.isa` has no `multiply`, and nothing similar exists. Its real names include: NkiInstruction, NkiValidationError, VirtualRegister, activate2, activation, activation_reduce, affine_select, bn_aggr, bn_stats, core_barrier, dge_mode, dma_compute, dma_copy, dma_engine, dma_transpose, dropout, engine, exponential, get_nc_sub_version, get_nc_version, gpsimd_engine, iota, local_gat

**1-D tile** (level 3, round 0, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) Every SBUF and PSUM tile needs two dimensions: a partition dimension first, then a free dimension. A 1-D tile is not allowed, so write nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or (N, 1) depending on which axis you are reducing over. The failing line is line 16 of your kernel: `psum = nl.ndarray(shape=out.shape, dtype

**result does not fit its tile** (level 3, round 1, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised ValueError: cannot reshape array of size 32768 into shape (1,64) Do not reshape. Work with the shapes you were given and slice them into tiles, e.g. src=a[0:128, 0:64]. The failing line is line 18 of your kernel: `nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT, moving=sbuf_rhs)`. That is the line to change.

**other: IndexError** (level 3, round 2, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised IndexError: tuple index out of range The failing line is line 18 of your kernel: `nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT[0:1, 0:out.shape[0]], moving=sbuf_rhs[0:out.shape[0], 0:out.shape[1]])`. That is the line to change.

**no tiling: partition over 128** (level 4, round 0, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pa

**no tiling: K over 128** (level 4, round 1, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul contraction dimension 256 exceeds pmax=128 The contraction dimension K is 256 and one nc_matmul can only contract 128. Split K into chunks of 128 and accumulate: allocate ONE psum tile OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so the partial products add up there, and only after the loop copy it out with nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write part

**no tiling: M or N over the limit** (level 4, round 2, reward 0.75)

> 2 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128 The failing line is line 34 of your kernel: `nisa.nc_matmul(dst=psum, stationary=sbuf_stationary[i:i+chunk_size, :], moving=sbuf_moving[i:i+chunk_size, :])`. That is the line to change.

**read past the end** (level 2, round 1, reward 0.30)

> 0 of 4 shapes passed. On shape=(32, 12) as 3x4: raised AssertionError: Out-of-bound access for tensor `unnamed` on dimension 1: index range [0, 127] exceed dimension size of 12. You indexed up to 127 on dimension 1, which is only 12 long. Tile limits are a MAXIMUM, not a target. Derive every bound from the tensor's own shape -- use min(limit, size) and let the final chunk be partial -- rather than writing a fixed number. Note the two limits differ: the partition dimension (first) allows at most 

**other: AssertionError** (level 4, round 1, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul contraction dimension mismatch: stationary[0]=128 != moving[0]=256 The failing line is line 43 of your kernel: `nisa.nc_matmul(dst=psum_tile, stationary=lhsT_tile, moving=sbuf_rhs)`. That is the line to change.

**NaN or Inf** (level 4, round 3, reward 0.50)

> 0 of 4 shapes passed. On K=128 M=128 N=512: NON-FINITE OUTPUT: 65536 NaN and 0 Inf, first at (0, 0). Usually an uninitialised PSUM or SBUF tile being read before anything wrote to it.

**other: TypeError** (level 4, round 5, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised TypeError: 'engine' object is not callable The failing line is line 31 of your kernel: `nisa.gpsimd_engine(psum, 0.0)`. That is the line to change.

**framework shortcut** (level 2, round 0, reward 0.10)

> Rule violations, which score zero however fast the kernel is. Fix exactly these: line 34: calls `transpose`, which hands the whole operation to a framework. This level is about computing it in the kernel.

