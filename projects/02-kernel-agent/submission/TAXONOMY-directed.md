# Failure taxonomy

84 attempts, levels [1, 2, 3, 4], 1 session(s): 20261010-161200

| failure mode | what it means | L1 | L2 | L3 | L4 | total | share |
|---|---|---|---|---|---|---|---|
| tile/slice size mismatch | source and destination hold different element counts | 4 | 2 | 17 | 2 | 25 | 30% |
| no tiling: M or N over the limit | K is chunked, but the output dimensions are not |  |  |  | 16 | 16 | 19% |
| called a dtype name | wrote nl.float32(0.5): a type name used as a function | 8 |  |  |  | 8 | 10% |
| read past the end | indexed beyond the tensor, usually a padded final tile |  | 1 | 4 |  | 5 | 6% |
| invented argument | passed a keyword the function does not take | 4 |  |  |  | 4 | 5% |
| wrong buffer | operand or result in the wrong memory (sbuf / psum / hbm) | 4 |  |  |  | 4 | 5% |
| invented API name | called an nl/nisa function that does not exist | 4 |  |  |  | 4 | 5% |
| missing argument | left out an argument the function requires | 4 |  |  |  | 4 | 5% |
| number where a tile belongs | passed a plain number to an argument that takes a tile | 4 |  |  |  | 4 | 5% |
| no tiling: K over 128 | contracted more than one tile's worth in one nc_matmul |  |  |  | 4 | 4 | 5% |
| 1-D tile | allocated an SBUF/PSUM tile with one dimension |  |  | 3 |  | 3 | 4% |
| no tiling: partition over 128 | one tile for the whole tensor |  |  |  | 2 | 2 | 2% |
| solved | correct on every shape |  | 1 |  |  | 1 | 1% |

## How the failure moved, round to round

17 repair steps: 7 repeated the same failure (41%), 10 changed it, and 0 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| tile/slice size mismatch | invented argument | 1 |
| invented argument | wrong buffer | 1 |
| wrong buffer | invented API name | 1 |
| invented API name | missing argument | 1 |
| missing argument | number where a tile belongs | 1 |
| number where a tile belongs | called a dtype name | 1 |
| tile/slice size mismatch | read past the end | 1 |
| read past the end | tile/slice size mismatch | 1 |
| no tiling: partition over 128 | no tiling: K over 128 | 1 |
| no tiling: K over 128 | no tiling: M or N over the limit | 1 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 4 | 399 | 1581 | 0 | 0 | 0 | 506 | 716 | 300 |
| repair | 14 | 0 | 0 | 995 | 518 | 0 | 201 | 538 | 341 |
| repair+ledger | 3 | 0 | 0 | 1351 | 432 | 473 | 266 | 823 | 455 |

## Line-located feedback

The checker quoted the failing line on 83 of 83 attempts that raised (100%).

## One real checker message per failure mode

**tile/slice size mismatch** (level 1, round 0, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384 The destination you allocated has shape (128, 128) and the piece you are copying into it has shape (2, 2). nisa.dma_copy needs the two shapes to be IDENTICAL. Make them match: either allocate the destination as nl.ndarray((2, 2), dtype=..., buffer=nl.sbuf), or copy a slice whose shape is (128, 128). The failing line is line 18 of your 

**invented argument** (level 1, round 1, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving' Remove the `transpose_moving=` argument. The real signature is isa.nc_matmul(dst: 'NkiTensor', stationary: 'NkiTensor', moving: 'NkiTensor', is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=<matmul_perf_mode.none: 'none'>, name=None). The failing line is line 19 of your kernel: `

**wrong buffer** (level 1, round 2, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dst must be in ['psum'], got sbuf Allocate the `dst` tile with buffer=nl.psum instead of nl.sbuf. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving must both be in nl.sbuf. Copy between them with nisa.tensor_copy. The failing line is line 19 of your kernel: `nisa.nc_matmul(dst=tile2, stationary=tile, moving=tile, is_transpose=True)`. That is the line to change.

**invented API name** (level 1, round 3, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AttributeError: module 'nki.isa' has no attribute 'multiply' `multiply` is not in `nisa`. It exists as `nl.multiply`. Write `nl.multiply` instead of `nisa.multiply`. The failing line is line 21 of your kernel: `nisa.tensor_scalar(dst=tile, data=0.5, op0=nisa.multiply)`. That is the line to change.

**missing argument** (level 1, round 4, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised TypeError: tensor_scalar() missing 1 required positional argument: 'operand0' The failing line is line 21 of your kernel: `nisa.tensor_scalar(dst=tile, data=0.5, op0=nl.multiply)`. That is the line to change.

**number where a tile belongs** (level 1, round 5, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AttributeError: 'float' object has no attribute 'shape' A float is not a numpy array, so it has no `shape`. Use the nl/nisa functions instead. The failing line is line 21 of your kernel: `nisa.tensor_scalar(dst=tile, operand0=tile, data=0.5, op0=nl.multiply)`. That is the line to change.

**called a dtype name** (level 1, round 6, reward 0.30)

> 0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised TypeError: 'str' object is not callable The failing line is line 21 of your kernel: `nisa.tensor_scalar(dst=tile, operand0=tile, data=nl.float32(0.5), op0=nl.multiply)`. That is the line to change.

**read past the end** (level 2, round 0, reward 0.30)

> 0 of 4 shapes passed. On shape=(32, 12) as 3x4: raised AssertionError: Out-of-bound access for tensor `unnamed` on dimension 0: index range [0, 127] exceed dimension size of 32. You indexed up to 127 on dimension 0, which is only 32 long. Tile limits are a MAXIMUM, not a target. Derive every bound from the tensor's own shape -- use min(limit, size) and let the final chunk be partial -- rather than writing a fixed number. Note the two limits differ: the partition dimension (first) allows at most 

**1-D tile** (level 3, round 0, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) Every SBUF and PSUM tile needs two dimensions: a partition dimension first, then a free dimension. A 1-D tile is not allowed, so write nl.ndarray((rows, cols), ...) and give a length-N vector the shape (1, N) or (N, 1) depending on which axis you are reducing over. The failing line is line 10 of your kernel: `psum = nl.ndarray(shape=lhsT.shape[1:], 

**no tiling: partition over 128** (level 4, round 0, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pa

**no tiling: K over 128** (level 4, round 1, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul contraction dimension 256 exceeds pmax=128 The contraction dimension K is 256 and one nc_matmul can only contract 128. Split K into chunks of 128 and accumulate: allocate ONE psum tile OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so the partial products add up there, and only after the loop copy it out with nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write part

**no tiling: M or N over the limit** (level 4, round 2, reward 0.75)

> 2 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128 The failing line is line 34 of your kernel: `nisa.nc_matmul(dst=psum, stationary=sbuf_stationary[i:i+chunk_size, :], moving=sbuf_moving[i:i+chunk_size, :])`. That is the line to change.

