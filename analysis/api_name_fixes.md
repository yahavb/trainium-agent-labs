# Invented APIs → the correct call (lookup table)

> Source: the 224 attempts of the first 3 runs of the seat-116 baseline (`analysis/baseline_seat116_attempts.csv`).
> Every entry was checked in the pod with `inspect.signature` against **nki 0.6.0**; documentation references point to `../neuron-agentic-development/skills/neuron-nki-docs/references/` (docs version NKI 0.4.0).
> Purpose: replace the fallback logic of `enrich()` in `projects/02-kernel-agent/agent.py`. Today, for a name that doesn't exist, it recommends names by spelling similarity with `difflib` (e.g. `scalar_engine, tensor_scalar_cumulative`), and the model still doesn't know what to write.

## 1. The errors that actually occurred in the baseline

| what the model wrote | count | level | what 0.6.0 actually has | suggested feedback (verdict → instruction) |
|---|---|---|---|---|
| `nisa.multiply(...)` | 36 | 1 | `nki.isa` has **no** `multiply`. `multiply` is in `nki.language`: `nl.multiply` can be passed as the op type of a nisa instruction, or called directly as `nl.multiply(x, y)` | `multiply` is not in nki.isa. To scale a tile by a constant write `nisa.tensor_scalar(dst=out, data=t, op0=nl.multiply, operand0=c)`; to multiply two tiles write `nisa.tensor_tensor(dst=out, data1=a, data2=b, op=nl.multiply)`. Note `nl.`, not `nisa.`, in front of `multiply`. |
| `nisa.scalar_mul(...)` | 12 | 1 | doesn't exist, not in `nl` either. The intent is "multiply by a constant" | Same as the first sentence above: there is no scalar_mul. To multiply a tile by a constant use `nisa.tensor_scalar(dst=out, data=t, op0=nl.multiply, operand0=c)`. |
| `nisa.nc_matmul(..., transpose_moving=...)` | 13 | 1, 2 | there is no such argument. The real signature: `nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=..., name=None)` | Remove `transpose_moving=`; nc_matmul has no such argument. (One extra sentence for L1: average pooling needs no matmul at all — reduce each pool window with `nl.sum(view, axis=[...])` over a strided `tile.ap([...])` view.) (One extra sentence for L2: a transpose does not need nc_matmul; `nisa.nc_transpose(dst=, data=)` exists in 0.6.0 — usage not verified; try it in the pod before putting it in the feedback.) |
| `nl.tile_size(...)` called as a function | 1 | 4 | `nl.tile_size` is a **set of constants** and can't be called (attributes such as `nl.tile_size.pmax`) | `nl.tile_size` is a set of constants, not a function: read `nl.tile_size.pmax` (=128) instead of calling it. |

**The feedback the harness gives today (for comparison)**
- `nisa.multiply` → *"`nki.isa` has no `multiply`, and nothing similar exists. Its real names include: NkiInstruction, …"* (25 names listed in alphabetical order)
- `nisa.scalar_mul` → *"The closest real names are: scalar_engine, tensor_scalar_cumulative, …"*
- `transpose_moving` → *"Remove the `transpose_moving=` argument. The real signature is …"* (this one is already good; it only lacks L1's "no matmul needed at all")

## 2. A likely root cause

The `API_CARD` in the prompt (`agent.py:171`) shows `nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)`. The model sees `multiply` and quite likely takes it for a function under `nisa`. One option is to say explicitly in the card: *"ops like nl.multiply / nl.add are passed as op= arguments; they are not nisa functions"*. After changing it, test with `--level 1 --repeat 5`, because a prompt change can make other levels worse (STATE.md records one such rollback).

## 3. Signatures checked in 0.6.0 (the ones levels 1–4 need)

```
nl.ndarray(shape, dtype, buffer=nl.sbuf, name='', address=None)
nl.affine_range(start, stop=None, step=1)
nl.sum(x, axis, dtype=None, keepdims=False)
nl.multiply(x, y=None, dtype=None)          # can also be passed as op=
nisa.dma_copy(dst, src, ...)
nisa.tensor_copy(dst, src, ...)
nisa.tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, ...)
nisa.tensor_tensor(dst, data1, data2, op, ...)
nisa.tensor_reduce(dst, op, data, axis, negate=False, keepdims=False)
nisa.nc_matmul(dst, stationary, moving, ..., is_transpose=False, accumulate=None, ...)
nisa.nc_transpose(dst, data, ...)
```

## 4. How to wire it into the harness (draft)

In `enrich()`, before the `module ... has no attribute` branch, look up a hand-written mapping first; only if it isn't there, fall back to `available_names()`:

```python
KNOWN_FIXES = {
    "nki.isa.multiply": "...",     # the suggested text from the table in section 1
    "nki.isa.scalar_mul": "...",
}
```

How to verify: `python agent.py --level 1 --rounds 8 --samples 4 --context 8192 --repeat 5 --log attempts_fix_api.jsonl`, compared with the baseline's level 1 (0.30 every time). Level 1 has always had zero variance, so any change in score can be attributed to this change.

## 5. Notes

- Do not put tutorial kernels such as `references/downloads/average_pool2d_nki_kernels.py` in the prompt: that is essentially the answer. The suggestions above give only API usage, never a full solution.
- Once L1's invented functions are fixed, the next wall is most likely "copy sizes don't match" (L1 has 24 of those too); be ready for the bottleneck to move.
