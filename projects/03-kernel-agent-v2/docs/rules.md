# Kernel rules

## The five rules

Your `kernel(...)` function must obey, or it scores zero:

1. **Fixed tiles.** Work in tiles of at most 128 rows x 512 columns. One tile at a time.
2. **Explicit loops over tiles.** Wrap the work in `for r0 in range(0, R, 128)`-style
   loops. No whole-array operation that hides the tiling.
3. **Slices and arithmetic on slices only.** No boolean masks as indexing, no
   integer-array indexing, no `np.einsum`, no broadcasting tricks, no `np.outer`.
4. **Ragged tail.** Shapes that do not divide evenly by the tile size MUST work. The
   final tile is partial; `x[r0:r0+128]` already stops at the end of the array.
5. **No calling the operation itself.** Each level bans the library call that would do
   the whole job (`np.sum` inside a sum kernel, `np.matmul` inside a matmul kernel, ...).
   Calls on a TILE SLICE (e.g. `t = x[r0:r0+128]; t.sum(axis=1)`) are legal -- the ban is
   on handing the WHOLE input to the library.

## What a compliant kernel looks like

```python
import numpy as np

def kernel(x):
    R, C = x.shape
    out = np.zeros(R, dtype=np.float64)
    for r0 in range(0, R, 128):          # step = tile height
        for c0 in range(0, C, 512):      # step = tile width
            t = x[r0:r0+128, c0:c0+512]  # the slice stops itself at the ragged edge
            out[r0:r0+128] += t.sum(axis=1)
    return out
```

## What the checker rejects, statically

- banned calls **on the whole input** (`np.sum(x, axis=1)` -- on a slice is fine)
- boolean/list subscripts (`x[x > 0]`, `x[:, [1, 2]]`)
- arithmetic whose operand is the whole input array (`x * a` before any slicing)
- no `for` loop at all, when the level's shapes do not fit in one tile
- wrong entry point: it must be `def kernel(...)`, same parameter names and order
  as the reference

## Scoring

0.1 parses, 0.2 rules clean, 0.2 runs, 0.5 correct on every case. Partial credit is
deliberate: a near miss must stay distinguishable from nonsense.
