# Failure taxonomy

164 attempts, levels [2, 3, 4], 4 session(s): 20261010-200712, 20261010-202354, 20261010-203420, 20261010-205002

| failure mode | what it means | L2 | L3 | L4 | total | share |
|---|---|---|---|---|---|---|
| tile/slice size mismatch | source and destination hold different element counts | 4 | 34 | 4 | 42 | 26% |
| other: AssertionError | an exception no rule above names yet |  | 16 | 12 | 28 | 17% |
| wrong numbers: core arithmetic | most elements wrong | 20 |  |  | 20 | 12% |
| Python operator on a tile | used += or * on a tile instead of a nisa op |  | 16 |  | 16 | 10% |
| no tiling: M or N over the limit | K is chunked, but the output dimensions are not |  | 8 | 8 | 16 | 10% |
| result does not fit its tile | an NKI call's result is a different size from its destination; the simulator reports it as a reshape error |  | 8 | 2 | 10 | 6% |
| no tiling: K over 128 | contracted more than one tile's worth in one nc_matmul |  |  | 8 | 8 | 5% |
| NaN or Inf | read an uninitialised tile |  |  | 8 | 8 | 5% |
| wrong buffer | operand or result in the wrong memory (sbuf / psum / hbm) |  |  | 8 | 8 | 5% |
| no tiling: partition over 128 | one tile for the whole tensor |  |  | 5 | 5 | 3% |
| 1-D tile | allocated an SBUF/PSUM tile with one dimension |  | 2 |  | 2 | 1% |
| solved | correct on every shape |  |  | 1 | 1 | 1% |

## How the failure moved, round to round

34 repair steps: 15 repeated the same failure (44%), 19 changed it, and 3 made the score go DOWN.

| after this failure | the next round hit | times |
|---|---|---|
| no tiling: partition over 128 | no tiling: K over 128 | 2 |
| no tiling: K over 128 | NaN or Inf | 2 |
| NaN or Inf | no tiling: M or N over the limit | 2 |
| tile/slice size mismatch | no tiling: M or N over the limit | 2 |
| no tiling: M or N over the limit | result does not fit its tile | 2 |
| result does not fit its tile | tile/slice size mismatch | 2 |
| tile/slice size mismatch | other: AssertionError | 2 |
| tile/slice size mismatch | Python operator on a tile | 1 |
| no tiling: M or N over the limit | solved | 1 |
| tile/slice size mismatch | wrong numbers: core arithmetic | 1 |
| no tiling: M or N over the limit | wrong buffer | 1 |

## Where the prompt went (characters, mean per prompt)

| prompt kind | prompts | reference | docs | code | feedback | ledger | instructions | prompt tokens | answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| first | 7 | 418 | 1581 | 0 | 0 | 0 | 504 | 712 | 335 |
| repair | 26 | 0 | 0 | 1128 | 949 | 0 | 200 | 712 | 406 |
| repair+ledger | 8 | 0 | 0 | 1020 | 766 | 639 | 268 | 887 | 330 |

## Line-located feedback

The checker quoted the failing line on 135 of 135 attempts that raised (100%).

## One real checker message per failure mode

**tile/slice size mismatch** (level 3, round 0, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got src=8192, dst=32768 The destination has shape (64, 512) but the piece copied into it has shape (128, 64), which is smaller. nisa.dma_copy needs IDENTICAL shapes and cannot fill part of a tile. Allocate a separate tile for this copy with exactly the source's shape: nl.ndarray((128, 64), dtype=..., buffer=nl.sbuf). If one tile is being used for two different tens

**1-D tile** (level 3, round 0, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim) The shape on the failing line has one dimension and a tile needs two. Putting a 1 in front is only right when the data really is a single row. Otherwise give the tile the full shape of the data it will hold: a tile that receives the result of nisa.nc_matmul needs shape (M, N), where stationary is (K, M) and moving is (K, N); a tile that receives a c

**Python operator on a tile** (level 3, round 1, reward 0.30)

> 0 of 1 shapes passed. On K=128 M=64 N=512: raised TypeError: unsupported operand type(s) for *: 'int' and 'range' The failing line is line 14 of your kernel: `nisa.dma_copy(dst=tile_rhs, src=rhs, dst_offset=K*nl.affine_range(M))`. That is the line to change. For reference, the real signatures are nisa.dma_copy(dst: 'NkiTensor', src: 'NkiTensor', priority=None, oob_mode=<oob_mode.error: 0>, dge_mode=<dge_mode.unknown: 0>, engine=<engine.unknown: 0>, name=None) and nl.affine_range(start, stop=None

**no tiling: partition over 128** (level 4, round 0, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128 A tile may have at most 128 rows, and you asked for 256. Do not allocate one tile for the whole tensor: loop over the partition dimension in chunks of at most 128 with nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy one chunk at a time, e.g. src=a[i*128:(i+1)*128, :]. If a dimension is already 128 or smaller, use it whole -- do NOT pa

**result does not fit its tile** (level 4, round 0, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised ValueError: cannot reshape array of size 65536 into shape (512,512) Your code does not call reshape, and removing one will not help. This is raised inside the NKI call on the failing line: its result has 65536 elements and the destination tile you gave it has shape (512, 512), which holds 262144. The destination has the wrong shape. For nisa.nc_matmul with stationary of shape (K, M) and moving of shape (K, N), dst must have shape (M, N). Allocat

**no tiling: K over 128** (level 4, round 1, reward 0.62)

> 1 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul contraction dimension 256 exceeds pmax=128 The contraction dimension K is 256 and one nc_matmul can only contract 128. Split K into chunks of 128 and accumulate: allocate ONE psum tile OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so the partial products add up there, and only after the loop copy it out with nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write part

**NaN or Inf** (level 4, round 2, reward 0.50)

> 0 of 4 shapes passed. On K=128 M=128 N=512: NON-FINITE OUTPUT: 65536 NaN and 0 Inf, first at (0, 0). Every one of the 65536 output elements is NaN. Your kernel passes accumulate=True to nisa.nc_matmul. Remove that argument. With it the result here is NaN: a newly allocated psum tile holds NaN, and the product is added to it. Without it, calling nisa.nc_matmul into the same psum tile once per chunk already adds the chunks together. Change nothing else.

**no tiling: M or N over the limit** (level 4, round 3, reward 0.75)

> 2 of 4 shapes passed. On K=256 M=256 N=1024: raised AssertionError: Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128 One nisa.nc_matmul call accepts at most 128 on the stationary operand's free dimension (M) and at most 512 on the moving operand's free dimension (N); you passed 256 on the stationary one. Keep your loop over K as it is and add loops over the output: split M into chunks of at most 128 and N into chunks of at most 512. For each (M chunk, N chunk) pair, allocate

**wrong buffer** (level 4, round 4, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised AssertionError: tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm `nisa.tensor_copy` only moves data between on-chip buffers, sbuf and psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output. The failing line is line 45 of your kernel: `nisa.tensor_copy(dst=out[m

**wrong numbers: core arithmetic** (level 2, round 1, reward 0.50)

> 0 of 4 shapes passed. On shape=(32, 12) as 3x4: NUMERICAL MISMATCH: worst error 4.49 of the output's RMS (1.01), tolerance 0.02.
  at index (24, 9): expected +2.62617, got -1.9109
  81.8% of elements are outside tolerance
  most elements are wrong, so this is the core arithmetic or the operand layout, not an edge case Nothing is computed in this task: the correct output holds exactly the values of `x`, moved. On this test case (x has shape (32, 12)) the correct positions are: out[0, 1] = x[0, 4]

**other: AssertionError** (level 4, round 5, reward 0.30)

> 0 of 4 shapes passed. On K=128 M=128 N=512: raised AssertionError: dma_copy requires HBM or SBUF tensors, got src=MemoryRegion.psum, dst=MemoryRegion.shared_hbm The failing line is line 53 of your kernel: `nisa.dma_copy(dst=out[m0:m0+chunk_M, n0:n0+chunk_N], src=psum)`. That is the line to change. For reference, the real signature is nisa.dma_copy(dst: 'NkiTensor', src: 'NkiTensor', priority=None, oob_mode=<oob_mode.error: 0>, dge_mode=<dge_mode.unknown: 0>, engine=<engine.unknown: 0>, name=None

