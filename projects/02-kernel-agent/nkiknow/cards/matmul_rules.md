How matrix multiplication works in NKI (read before writing code):

- Inputs: `lhsT` has shape [K, M] (the left matrix, already transposed, K first). `rhs` has shape [K, N].
  The result is [M, N]. K is the contraction dimension.
- The instruction is `nisa.nc_matmul(dst=, stationary=, moving=)`. `stationary` is a [K, M] tile of lhsT,
  `moving` is a [K, N] tile of rhs, and `dst` receives the [M, N] product.
- Where data must live: `dst` must be in `nl.psum`; `stationary` and `moving` must both be in `nl.sbuf`.
  HBM tensors cannot be used directly: copy tiles HBM -> SBUF with `nisa.dma_copy` first.
- Index each input by ITS OWN dimensions: a `lhsT` tile is `lhsT[k-range, m-range]`, an `rhs` tile is
  `rhs[k-range, n-range]`, and the result goes to `output[m-range, n-range]`. Never reuse one slice for
  different tensors: lhsT, rhs and the output have different shapes.
- Tile limits for one nc_matmul: K at most 128 (it is the partition dimension), M at most 128, N at most 512.
  Bigger matrices are processed as a grid of tiles, with loops over M, N and K.
- Accumulating over K: allocate one PSUM tile per output tile BEFORE the K loop, then call nc_matmul into
  that same PSUM tile once per K chunk, with no `accumulate` argument: the first call overwrites and later
  calls add up automatically. Do not zero PSUM with memset. Do not create a new PSUM tile per K chunk and
  do not write partial results to HBM.
- Loops are ordinary for-loops: `for k in nl.affine_range(n):`. The same instructions run for every k;
  do not branch with Python `if`/`else` on the loop index.
- Getting the result out: after the K loop, copy PSUM -> SBUF with `nisa.tensor_copy`, then SBUF -> the
  output with `nisa.dma_copy`. The output is allocated with `buffer=nl.shared_hbm` and returned.
- Allocate every tile with `nl.ndarray(shape, dtype=..., buffer=nl.sbuf | nl.psum)`; memory regions are not
  functions. Every `nl.ndarray(...)` call is a new empty tile: assign it to a name once and
  pass that name around; never write `nl.ndarray(...)` inside another call's arguments. Slices given to dma_copy must have exactly the tile's shape.
- Derive loop counts from the tensor shapes (e.g. `K // 128`), never from fixed numbers.
