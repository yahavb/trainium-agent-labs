# Output shapes (the operation contract)

## The output shape comes from the OPERATION, never from the tiles

Tiles organise your LOOPS. They never appear in the output shape. Derive the output
shape from the reference's `return` expression:

| operation family | output shape | example |
|---|---|---|
| row-wise reduction (sum, max, min, argmax, mean) | `(R,)` — ONE value per row | `x.sum(axis=1)` of `(129, 513)` is `(129,)` |
| cumulative reduction (cumsum) | same shape as input `(R, C)` | one value per ELEMENT |
| normalisation (RMSNorm, layernorm, softmax) | same shape as input `(R, C)` | |
| elementwise (relu(a*x+b), x*2) | same shape as input | |
| transpose | `(C, R)` — axes swapped | |
| matmul `(M, K) @ (K, N)` | `(M, N)` | |
| 1D valid conv `(C_in, L)`, kernel K, stride s, dilation d | `(C_out, L_out)`, `L_out = (L - d*(K-1) - 1) // s + 1` | |

## The measured failure

A row-sum kernel that returned a `(ceil(R/128), ceil(C/512))` GRID of per-tile summary
values scored 0.5 on every attempt: it ran, it was tiled, and it answered a different
question. Before writing the loops, write one line:

```python
out = np.zeros((R,), dtype=np.float64)      # row-wise sum: ONE entry per row
```

then make every loop write a slice of THAT. If your output's shape contains
`ceil(... / 128)` or `ceil(... / 512)`, you are returning the tile grid, not the answer.

## Reductions over rows that span column tiles

For a row statistic (sum, max, softmax) the whole ROW must contribute, no matter how
many column tiles it spans. Combine partial results with the OPERATION ITSELF as you
walk the tiles:

```python
out = np.zeros(R, dtype=np.float64)          # sums
for c0 in range(0, C, 512):
    out += x[:, c0:c0+512].astype(np.float64).sum(axis=1)

out = np.full(R, -np.inf, dtype=np.float64)  # max: identity is -inf, never 0
for c0 in range(0, C, 512):
    out = np.maximum(out, x[:, c0:c0+512].astype(np.float64).max(axis=1))
```

Assigning (`out[...] = tile_stat`) instead of combining (`+=`, `np.maximum`) keeps only
the LAST tile's contribution -- every element wrong, no crash.
