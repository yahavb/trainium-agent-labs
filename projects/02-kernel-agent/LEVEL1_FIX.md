# Level 1: 2D average pooling

Level 1 computes non-overlapping average pooling on an input `x` with shape `(C, H, W)`. For
pool size `p`, the result has shape `(C, H // p, W // p)` and the input dtype. The reference drops
any trailing rows or columns that do not form a complete window. The four configured cases are
divisible by their pool sizes.

## Current agent guidance

The level-specific prompt describes pooling as a window reduction: load the full input tile, form
a strided access-pattern view, sum the two pool axes, scale by `1 / (p * p)`, and write the smaller
result. It keeps channel dimension `C` on the partition axis and uses separate SBUF tiles for
input and result plus a shared-HBM output. It does not use the matmul/PSUM workflow.

For an original input tile with shape `(C, H, W)`, the prompt specifies this access pattern; each
pair is `[element_stride, count]`:

```python
view = in_tile.ap([
    [H * W, C],
    [p * W, H // p],
    [p, W // p],
    [W, p],
    [1, p],
])
sums = nl.sum(view, axis=[3, 4])
```

The first stride must be `H * W`. The view is created once from the original full `(C, H, W)`
tile, not from a reshaped or already-viewed tile. The two final axes represent the pool window;
their reduction leaves `(C, H // p, W // p)`. The sum is scaled by `1 / (p * p)` and stored in
the input dtype.

The repair prompt can replace an incorrect pooling algorithm while preserving the entry point and
arguments. When the checker specifically reports an invalid `.ap()` partition stride, the repair
prompt narrows the requested change to the access pattern on the existing full-input tile and
repeats the shape-derived pattern. This is targeted guidance, not proof that every possible NKI
failure has a specialized repair.

## Recorded observations and validation

The README records an early Seat 21 baseline in which all 32 samples scored 0.30; candidates used
matmul for pooling, then encountered incompatible tiles or invented APIs. That is a historical
baseline, not evidence of the current prompt's solve rate. The repository does not record a
controlled before/after comparison showing that the prompt change improved the model's results.

`nkibench.py --selftest` checks the NumPy pooling reference on a host input. Checking
`reference_level1.py` with `nkibench.py --check` validates that supplied kernel against the
configured simulator checks; neither command measures whether the model can generate a passing
kernel. Report model performance as the fraction of repeated runs that pass all four configured
shapes, based on the run log.

Use the current checkout on the pod to gather fresh evidence:

```bash
python nkibench.py --selftest
python nkibench.py --level 1 --check reference_level1.py
python -u agent.py --level 1 --rounds 8 --samples 4 --context 8192 --repeat 5 \
  --log attempts-level1.jsonl
```

Analyze the resulting JSONL with `python analyze.py attempts-level1.jsonl`. Run agent experiments
sequentially: candidate files use the shared `/tmp/_agent_level1.py` path, so separate checkouts
alone do not isolate concurrent runs.
