# Tiling patterns

## Pattern: preallocate the output — never concatenate tiles

Collecting tiles in a list and joining them is the most common way a tiled kernel dies on
the ragged tail:

```python
result = []
for r0 in range(0, R, 128):
    ...
    result.append(piece)              # 128 rows, except the last one
return np.concatenate(result, axis=0) # ValueError: size 128 vs size 1
```

The final tile is PARTIAL, so the pieces have different sizes and concatenate refuses
them. Preallocate instead, and assign each tile to its own slice — sizes then do not
matter:

```python
out = np.zeros((R, C), dtype=np.float64)     # BEFORE the loops
for r0 in range(0, R, 128):
    for c0 in range(0, C, 512):
        t = x[r0:r0+128, c0:c0+512]
        out[r0:r0+128, c0:c0+512] = do_something(t)
return out
```

## Pattern: row statistics that span column tiles

A statistic over the LAST axis (mean, RMS, variance, softmax denominator) belongs to the
WHOLE row. When the row is wider than one column tile, a statistic computed per tile is
simply a different quantity — normalising by it returns +-1 in a one-wide tile and a
slightly-wrong scale everywhere else. Two passes:

```python
sq = np.zeros(R, dtype=np.float64)           # pass 1: accumulate per-row sums
for c0 in range(0, C, 512):
    t = x[:, c0:c0+512].astype(np.float64)   # a whole-row strip of column tiles
    sq += (t * t).sum(axis=1)
rms = np.sqrt(sq / C) + eps                  # ONE statistic per row, whole-row scope

for r0 in range(0, R, 128):                  # pass 2: apply it per tile
    for c0 in range(0, C, 512):
        t = x[r0:r0+128, c0:c0+512].astype(np.float64)
        out[r0:r0+128, c0:c0+512] = t / rms[r0:r0+128, None]
```

The pass-1 accumulation loop steps over COLUMNS so each strip stays inside the tile
budget; the per-row statistic it builds is global to the row. Softmax needs the same
shape: pass 1 computes the row max and the sum of exponentials across tiles, pass 2
writes `exp(x - max) / sum`.


## Pattern: reduction along an axis, row tiles

Row-wise sum/max/norm all share one shape: loop over row tiles, reduce within the tile,
write the tile's result to its own output slice. Handling the ragged tail is free if you
slice and let the slice cut itself.

```python
def kernel(x):
    R, C = x.shape
    out = np.zeros(R, dtype=np.float64)
    for r0 in range(0, R, 128):
        t = x[r0:r0+128]                  # may be shorter than 128 on the last tile
        out[r0:r0+128] = t.max(axis=1)    # or sum / mean / your reduction
    return out
```

Identity values matter: max starts at `-inf`, never `0`. A row of all-negative values
with a `0` identity returns `0.0` for every row and passes nothing.

## Pattern: two-pass normalization (softmax, layernorm, RMSNorm)

Anything with an exp or a variance does a first pass for the row statistic, a second for
the transform. Keep the first-pass result as a per-row column so it broadcasts against
the tile WITHOUT a broadcast trick:

```python
for r0 in range(0, R, 128):
    t = x[r0:r0+128].astype(np.float64)
    m = t.max(axis=1, keepdims=True)      # (tile_rows, 1) -- subtraction is per-row
    e = np.exp(t - m)                     # stable: exponent is always <= 0
    out[r0:r0+128] = e / e.sum(axis=1, keepdims=True)
```

`keepdims=True` is what makes `t - m` legal per-tile arithmetic rather than a broadcast
trick. exp of a raw score is the classic overflow: scores of a few thousand are already
past float64's exp limit.

## Pattern: tiled matmul with accumulation

Three loops: m over row tiles of the output, n over column tiles, k over the contraction
dimension. One accumulator per (m, n) tile, reused across the k loop:

```python
def kernel(a, b):                  # a: (M, K), b: (K, N)
    M, K = a.shape; _, N = b.shape
    out = np.zeros((M, N), dtype=np.float64)
    for m0 in range(0, M, 128):
        for n0 in range(0, N, 512):
            acc = np.zeros((min(128, M - m0), min(512, N - n0)), dtype=np.float64)
            for k0 in range(0, K, 128):
                at = a[m0:m0+128, k0:k0+128]
                bt = b[k0:k0+128, n0:n0+512]
                acc += at @ bt      # small tile matmul: K-accumulation lives here
            out[m0:m0+128, n0:n0+512] = acc
    return out
```

The accumulator is `float64` and lives OUTSIDE the k loop. Allocating it inside the k
loop throws away every partial product.

## Pattern: cumulative quantities across tiles (running state)

A row-wise cumsum is independent per ROW, but a row is wider than one column tile -- so
the state that crosses a boundary is the row's running total at the tile edge. Keep one
per-row carry vector; add it to each tile's local cumsum, then update it from what you
wrote:

```python
for r0 in range(0, R, 128):
    prev = np.zeros((min(128, R - r0), 1))
    for c0 in range(0, C, 512):
        t = x[r0:r0+128, c0:c0+512].astype(np.float64)
        local = np.cumsum(t, axis=1)            # legal: library call on a tile slice
        out[r0:r0+128, c0:c0+512] = local + prev
        prev = out[r0:r0+128, c0:c0+512][:, -1:]  # each row's total at this tile edge
```

The carry is a COLUMN vector (one running total per row), read from the OUTPUT -- not
from `local` (which forgets the carry) and not a scalar (rows are independent; only
columns chain). Test with a SCRATCH line on a (2, 5) array with a 4-column tile before
answering: a carry bug is invisible on a single-tile shape.

## Pattern: transposing inside tiles

A transpose crosses the row/column boundary, so output tile (m0, n0) reads input tile
(n0, m0) -- the loop indices swap places between read and write:

```python
for m0 in range(0, R, 128):
    for n0 in range(0, C, 512):
        out[m0:m0+128, n0:n0+512] = x[n0:n0+512, m0:m0+128].T
```

`x.T` on the WHOLE input is banned; `.T` on a tile slice is the operation itself and is
fine when the level allows it.
