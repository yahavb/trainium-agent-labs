What NKI is (read this first):

- NKI kernels are written in Python syntax but they are NOT NumPy. They work like CUDA or Triton tile
  kernels: you manage on-chip memory yourself and every instruction works on one small fixed-size tile.
  The Python is traced once to build the kernel; it does not run line by line on the chip.
- Three memories. HBM: large and slow; the inputs and the output live here. SBUF: on-chip, split into 128
  lanes (partitions). PSUM: on-chip, where matrix-multiply results add up. Data moves between them only
  through explicit instructions such as `nisa.dma_copy(dst=, src=)`.
- A tile is a fixed buffer, not an array value. Its first dimension is the lane dimension: at most 128.
  Allocate a tile once and give it a name: `t = nl.ndarray((rows, cols), dtype=..., buffer=nl.sbuf)`.
  Every instruction writes into a `dst=` tile you allocated. Never put `nl.ndarray(...)` inside another
  call: each one is a new, empty tile.
- No NumPy math on tiles: no `+`, `*`, `@`, no `np.*`. Use `nisa.*` instructions.
- Loops are ordinary for-loops over `nl.affine_range(n)`. They are unrolled when the kernel is built, so the
  loop index is not a Python int: compute slice bounds from it, but do not branch on it with `if`.
- Big tensors are processed as a grid of tiles: one loop per dimension that can exceed its limit, a
  tile-sized buffer allocated inside the loop, one `dma_copy` in, the work, one `dma_copy` out.

A complete, correct kernel for a DIFFERENT operation, y = a*x + b on a large 2-D input, showing that pattern
(including the partial last tile):

```python
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def scale_shift_kernel(x, a, b):
    R, C = x.shape
    y = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    TR, TC = 128, 512
    for i in nl.affine_range((R + TR - 1) // TR):
        for j in nl.affine_range((C + TC - 1) // TC):
            r0, c0 = i * TR, j * TC
            rows, cols = min(TR, R - r0), min(TC, C - c0)
            t_in = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
            t_out = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t_in, src=x[r0:r0 + rows, c0:c0 + cols])
            nisa.tensor_scalar(dst=t_out, data=t_in, op0=nl.multiply, operand0=a,
                               op1=nl.add, operand1=b)
            nisa.dma_copy(dst=y[r0:r0 + rows, c0:c0 + cols], src=t_out)
    return y
```
