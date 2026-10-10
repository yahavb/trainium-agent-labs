# Numeric gotchas

## "The last tile wins" -- assigning partial results in the inner loop

A per-row result computed inside a column-tile loop must COMBINE tiles, not assign:

```python
for r0 in range(0, R, 128):
    for c0 in range(0, C, 512):
        result[r0:r0+128] = tile.sum(axis=1)   # WRONG: each tile overwrites the last;
    return result                              # output == last column tile only
```

Every element is wrong and nothing crashes: the (129, 513) case returns column 512's
lone value per row. Combine with the operation itself — sums `+=` from a zeros init,
max `np.maximum(out, tile_max)` from a `-inf` init, min `np.minimum` from `+inf`.
Same rule for normalisations: compute the whole-row statistic first, then apply it
(row statistics that span column tiles, in the patterns card).


## Per-tile softmax returns 1.0 in a one-wide tile

Softmax computed per COLUMN tile looks almost right and is wrong everywhere: the
denominator belongs to the whole row. The loud signature: a one-wide final tile
normalizes its single element to exactly **1.0** while the reference value there is
~1/C — plausible numbers, no crash. The quiet version: every element of every row is
scaled by `sum_whole_row / sum_tile`, a small error within tolerance on friendly data
and far outside it on wide rows. Fix: compute the row max and the row sum of
exponentials over the WHOLE row (accumulate across column tiles), then normalize in a
second pass — see "row statistics that span column tiles" in the patterns card.


Every one of these produces PLAUSIBLE numbers, not a crash. That is why the test battery
exists and why each has a named fix.

## exp overflow (softmax, attention scores)

`np.exp` overflows near 709 in float64 and near 88 in float32. Raw scores reach
thousands. ALWAYS subtract the row max first: `e = np.exp(t - t.max(axis=1, keepdims=True))`.
The result is mathematically identical (the max cancels in the normalization) and cannot
overflow. Symptom of skipping it: NaN on large-magnitude inputs, perfect answers on
small ones.

## Catastrophic cancellation (variance, layernorm)

`E[x^2] - E[x]^2` subtracts two large, nearly-equal numbers. With means around 1e4 in
float32 it returns noise near zero instead of the true variance (~1). Fix: two-pass --
compute the mean first, then the mean of squared deviations `(t - mu)**2` -- or do the
whole computation in float64. The test battery has a "large" regime exactly for this.

## float32 accumulation drift

Summing float32 in a naive loop over n elements drifts up to ~n * 1.2e-7 relative to the
total. Over 513 elements that is ~6e-5 -- fine against the level tolerances, so this is
NOT a correctness problem, but do not chase exactness: accumulate in float64 inside the
tile and the drift vanishes for free.

## The -inf identity

`max` reduction starts from `-inf`. A kernel that starts from `0` returns 0 for
all-negative rows. The battery includes all-negative rows (the "large" regime has mean
+1e4 but a separate normal regime does not protect you -- only correct identities do).

## Output-size formulas

1D valid conv: `L_out = (L - dilation * (K - 1) - 1) // stride + 1`. Every term matters;
dilation multiplies (K-1), not K, and the whole thing floors BEFORE +1. A one-off here
produces a wrong-SHAPE verdict, which is the checker telling you the formula is wrong,
not the arithmetic.

## Ragged final tile

The last tile is smaller. Slices handle it for free (`x[r0:r0+128]` stops at the end),
but anything derived from the constant 128 -- output indices, carry positions, "number
of rows in this tile" -- must come from the SLICE's real shape (`t.shape[0]`), not the
constant. The battery always includes a prime-length dimension and a length of 1.

## Ties (argmax-style outputs)

When several elements share the maximum, the answer is the FIRST index. A kernel that
uses `>` where it needs `>=` (or iterates the tile backwards) returns the last one. The
battery has rows of repeated values to catch exactly this.

## Band edges (windowed attention)

Position i attends to [i-w, i+w]. Both ends of the band clip at the array boundary, and
rows near the edges have FEWER valid positions -- their softmax denominators differ. An
off-by-one at either edge moves a whole attention weight. And when w >= n, every position
attends to everything: the mask must never produce an all-masked row.
