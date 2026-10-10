NKI guide (for nki 0.6.0). Read it before writing code.

## 0. Before you write code
Write a short plan as Python comments at the top of your code block, then the code:
1. Every tensor you will touch (inputs, output, each tile) and its shape.
2. For every instruction you will use, each operand's dimensions and that instruction's limit for them
   (lanes <= 128; matmul: K <= 128, M <= 128, N <= 512; max8 reads <= 16384 per lane).
3. A loop for EVERY dimension that can exceed its limit, including output dimensions, and the order of loops.
4. Which tiles are allocated where (outside or inside which loop) and which memory each lives in.
Then write the kernel exactly as planned.

## 1. What NKI is
NKI kernels use Python syntax but are NOT NumPy. They work like CUDA or Triton tile kernels: you move data
between memories yourself, and each operation works on a small tile. The Python runs once, when the kernel is
built (traced); it does not run line by line on the chip. Imports: `import nki`, `import nki.isa as nisa`,
`import nki.language as nl`. The entry point is a function decorated with `@nki.jit` that returns its output.

## 2. Memory
- HBM: large and slow. Kernel inputs arrive here; allocate the output here with
  `out = nl.ndarray(shape, dtype=..., buffer=nl.shared_hbm)` and `return out`.
- SBUF: on-chip, 128 lanes (partitions). Compute happens on SBUF tiles.
- PSUM: on-chip, where `nisa.nc_matmul` writes and accumulates results. `nc_matmul` can ONLY write to a PSUM
  tile: never to SBUF and never to the HBM output.
- Data moves only through explicit copies: HBM <-> SBUF with `nisa.dma_copy(dst=, src=)`; PSUM -> SBUF with
  `nisa.tensor_copy(dst=, src=)`. So a matmul result reaches the output in three steps: `nc_matmul` into PSUM,
  `tensor_copy` PSUM -> SBUF, `dma_copy` SBUF -> HBM.

## 3. Tiles
- A tile's FIRST dimension is the lane (partition) dimension: at most 128, always.
- The limit on the SECOND (free) dimension depends on the instruction that uses the tile: `nc_matmul`'s
  `stationary` operand allows at most 128, its `moving` operand at most 512, and a PSUM tile at most 512
  float32. Check every operand of every instruction against ITS limit; do not assume a free dimension may be
  large. Tiles are 2-D: a vector of n values is shape (n, 1) or (1, n).
- Result and accumulator tiles are TILE-sized too, not output-sized: a destination tile has the shape that one
  instruction produces from its operand tiles (for nc_matmul: [m-chunk, n-chunk], at most [128, 512]). Allocate
  it inside the loops over output tiles. Never allocate a PSUM or SBUF tile with the whole output's shape: it
  breaks the size limits and mixes different output tiles together.
- Allocate a tile once and give it a name: `t = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)`.
  Never write `nl.ndarray(...)` inside another call: each one is a new, empty tile.
- `nl.sbuf`, `nl.psum`, `nl.shared_hbm` are memory-region values for `buffer=`; they are not functions.
- A slice of a tensor or tile (`x[r0:r0 + rows, c0:c0 + cols]`) is a view; the destination of a copy must
  have exactly the same shape as its source.

## 4. Computing on tiles: two styles, both valid
- `nl.*` functions return a new tile: `e = nl.exp(t)`, `s = nl.sum(e, axis=1, keepdims=True)`,
  `m = nl.max(t, axis=1, keepdims=True)`, `nl.add(a, b)`, `nl.multiply(a, b)`, `nl.maximum(a, b)`,
  `nl.subtract`, `nl.divide`.
- `nisa.*` instructions write into a `dst=` tile you allocated (more control, maps to one hardware engine).
- Python operators (`+ - * / @`) and `np.*` do NOT work on tiles (TypeError).

## 5. Loops and sizes
- Loops are ordinary for-loops: `for i in nl.affine_range(n):` (use `nl.sequential_range(n)` when an
  iteration depends on the previous one). Never `with nl.affine_range(...)`.
- Loops are unrolled when the kernel is built, so the loop index is not a Python int: compute slice bounds
  from it (`r0 = i * 128`), but never branch on it (`if i == 0`).
- Large tensors are a grid of tiles: one loop per dimension that can exceed its limit; inside, allocate a
  tile-sized buffer, `dma_copy` one chunk in, compute, `dma_copy` the result out. Use
  `(n + 127) // 128` tiles and `min(128, n - r0)` for the last, partial tile.
- Reductions (a sum over a dimension, like matmul's K or a row sum's columns): the loop over the summed
  dimension goes INSIDE the loops over output tiles. For each output tile: allocate its accumulator (a PSUM
  tile for nc_matmul) BEFORE the reduction loop, add into it on every iteration (nc_matmul into the same PSUM
  tile adds automatically), and copy it to the output only AFTER the reduction loop. Writing the output inside
  the reduction loop keeps only the last chunk; sharing one accumulator across output tiles mixes them.
- Every tensor is indexed by its own dimensions. Work out each tensor's shape before slicing it; never reuse
  one slice expression for tensors of different shapes.

## 5b. Moving fewer bytes (when the task has a traffic limit)
- The checker counts every byte that `nisa.dma_copy` moves between HBM and on-chip memory. The minimum is
  reading each input once and writing the output once; a traffic limit is a multiple of that minimum.
- Load a tile once and reuse it while it is on chip. If a load does not depend on the index of an inner loop,
  move it out of that loop.
- Keep tiles that are reused many times resident in SBUF (224 KiB per lane on Trainium2) and process several
  output tiles while they are loaded, instead of reloading the same input for every output tile.
- Order loops so the data reused the most stays loaded the longest. Stay correct first: a wrong kernel scores
  zero however few bytes it moves.

## 6. Instruction cheat sheet (exact signatures; `dst` is always written)
- `nisa.dma_copy(dst, src)`: copy HBM <-> SBUF; shapes must match.
- `nisa.tensor_copy(dst, src)`: copy between on-chip tiles (e.g. PSUM -> SBUF).
- `nisa.tensor_scalar(dst, data, op0, operand0, op1=None, operand1=None)`: `dst = (data op0 operand0) op1 operand1`, ops like `nl.multiply`, `nl.add`; operands are floats or (rows, 1) tiles.
- `nisa.tensor_tensor(dst, data1, data2, op)`: element-wise op of two same-shape tiles.
- `nisa.tensor_reduce(dst, op, data, axis, keepdims=False)`: reduce along the free axis (`op=nl.add` or `nl.max`).
- `nisa.activation(dst, op, data, bias=None, scale=1.0)`: `dst = op(scale*data + bias)`, op like `nl.exp`. To also sum each row, pass `reduce_op=nl.add, reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=<(rows,1) tile>`; `reduce_op` without `reduce_cmd` raises an error.
- `nisa.nc_matmul(dst, stationary, moving)`: `dst[M, N] (+)= stationary[K, M].T @ moving[K, N]`; `dst` in PSUM, both inputs in SBUF; K <= 128, M <= 128, N <= 512. Repeated calls into the same PSUM tile add up (first call overwrites); no `accumulate` argument needed.
  K larger than 128: loop over K in chunks of 128 INSIDE the loops over output tiles (M, N), loading
  `lhsT[k-chunk, m-chunk]` and `rhs[k-chunk, n-chunk]` each time and calling nc_matmul into that output tile's
  PSUM tile; after the K loop, copy PSUM -> SBUF -> output once.
- `nisa.nc_transpose(dst, data)`: swap the lane and free axes of a tile.
- `nisa.max8(dst, src)`: the 8 largest values per lane, largest first (src free size 8..16384); `nisa.nc_find_index8(dst, data, vals)` gives their indices (lowest index on ties).
- `nisa.memset(dst, value)`: fill a tile with a constant. `nisa.reciprocal(dst, data)`: 1/x.

## 7. Errors and what they mean
- `partition dimension 256 exceeds maximum 128`: a tile has more than 128 rows; you allocated or copied a
  whole tensor. Loop over that dimension in chunks of at most 128.
- `Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128`: the stationary tile's second
  dimension (M) is over 128. Give M its own loop in chunks of at most 128.
- `Matmul moving free dimension 1024 exceeds max 512`: the moving tile's second dimension (N) is over 512.
  Give N its own loop in chunks of at most 512.
- `dma_copy requires src and dst to have the same number of elements`: the slice and the tile differ in shape.
- `Out-of-bound access ... index range [a, b] exceed dimension size of n`: a slice goes past a tensor's end,
  often because one tensor was sliced with another tensor's ranges.
- `'MemoryRegion' object is not callable`: you called `nl.sbuf(...)`; use `nl.ndarray(..., buffer=nl.sbuf)`.
- `'range' object does not support the context manager protocol`: you wrote `with nl.affine_range`; use `for`.
- `unsupported operand type(s) for +: 'NkiTensor'`: Python operator on a tile; use `nl.add` or `nisa.*`.
- NUMERICAL MISMATCH only on shapes where the summed dimension is large (e.g. K > 128): the reduction is not
  accumulated over all chunks — usually the output is written inside the reduction loop, or the reduction
  loop is outside the output-tile loops.
- NaN in the output: a tile was read before anything wrote to it (often an inline `nl.ndarray(...)`, or PSUM
  read before any `nc_matmul` wrote to it).
- `must be in ['psum']` / `['sbuf']`: an instruction's operand is in the wrong memory (see section 2).

## 8. Three complete, correct kernels (each for a different operation)

(a) y = a*x + b on a large 2-D input
```python
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def scale_shift_kernel(x, a, b):
    R, C = x.shape
    y = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range((R + 127) // 128):
        for j in nl.affine_range((C + 511) // 512):
            r0, c0 = i * 128, j * 512
            rows, cols = min(128, R - r0), min(512, C - c0)
            t_in = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
            t_out = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t_in, src=x[r0:r0 + rows, c0:c0 + cols])
            nisa.tensor_scalar(dst=t_out, data=t_in, op0=nl.multiply, operand0=a,
                               op1=nl.add, operand1=b)
            nisa.dma_copy(dst=y[r0:r0 + rows, c0:c0 + cols], src=t_out)
    return y
```

(b) per-row sum of a large [R, C] input -> [R, 1], accumulated across column tiles (sequential_range because acc carries across iterations)
```python
@nki.jit
def row_sum_kernel(x):
    R, C = x.shape
    y = nl.ndarray((R, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range((R + 127) // 128):
        r0 = i * 128
        rows = min(128, R - r0)
        acc = nl.ndarray((rows, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for j in nl.sequential_range((C + 511) // 512):
            c0 = j * 512
            cols = min(512, C - c0)
            t_in = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
            part = nl.ndarray((rows, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t_in, src=x[r0:r0 + rows, c0:c0 + cols])
            nisa.tensor_reduce(dst=part, op=nl.add, data=t_in, axis=1)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add)
        nisa.dma_copy(dst=y[r0:r0 + rows, 0:1], src=acc)
    return y
```

(c) 2-D transpose [R, C] -> [C, R] with the tensor engine (result lands in psum, copy it to sbuf)
```python
@nki.jit
def copy_t_kernel(x):
    R, C = x.shape
    y = nl.ndarray((C, R), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range((R + 127) // 128):
        for j in nl.affine_range((C + 127) // 128):
            r0, c0 = i * 128, j * 128
            rows, cols = min(128, R - r0), min(128, C - c0)
            t_in = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
            t_ps = nl.ndarray((cols, rows), dtype=x.dtype, buffer=nl.psum)
            t_out = nl.ndarray((cols, rows), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t_in, src=x[r0:r0 + rows, c0:c0 + cols])
            nisa.nc_transpose(dst=t_ps, data=t_in, engine=nisa.engine.tensor)
            nisa.tensor_copy(dst=t_out, src=t_ps)
            nisa.dma_copy(dst=y[c0:c0 + cols, r0:r0 + rows], src=t_out)
    return y
```

These are different operations; adapt the patterns, not the slices.
