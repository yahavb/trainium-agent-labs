# NKI fix examples (nki 0.6.0)

One small, checked example kernel per common error, for the new agent's debugger to show next to an
error message. Each entry below has five fields:
- `levels`: where it may be shown;
- `when`: a regex over the error text;
- `error`: a real error it must match;
- `input` and `expect`: what the example computes, for the check;
- then a note and the example itself.

Use them through `nki_fix_examples.py`:

    from nki_fix_examples import example_for
    note = example_for(error_text, level)     # "" when nothing matches; same format as agent.fragment_note

    PYTHONDONTWRITEBYTECODE=1 python nki_fix_examples.py                              # check every example
    PYTHONDONTWRITEBYTECODE=1 python nki_fix_examples.py --coverage runs/*/attempts.jsonl

**Order matters: the first match wins.** Level-restricted entries come before the general entries
they overlap. For example, `block` also matches "cannot reshape", which at level 4+ is usually an
`nc_matmul` shape error.

**An example must never be a level's answer.** Each one is smaller than the level it serves and
shows a pattern, not the solution:
- `mmshape`, a single `nc_matmul`, is excluded from level 3, because that is level 3's answer. From
  level 4 on, it is a lower level's skill, and it has no K loop, which `withhold.json` reserves for
  levels 3–7.
- `nomatmul1` says only that pooling needs no matmul, and shows a plain row reduction. It used to
  show a strided view (`t.ap`) as well; that view is level 1's answer, so it was taken out on
  2026-10-10 at the user's request. The agent can still look `t.ap` up itself.

## Where these come from

- **The original five** (`scale`, `block`, `chunks`, `rows`, `columns`) are `agent.FRAGMENTS` from
  `31p` at `631f15c`, moved here unchanged except that `scale` also catches `scalar_mul`.
- **Three were added on 2026-10-10.** I counted the error kinds in 436 failed attempts on seat-35:
  the 8K baseline, `part1-1010-1858` and `l1-explore-1010-1933`. The largest uncovered ones were:

| error | count | levels | added |
|---|---|---|---|
| matmul misuse: `dst must be in ['psum']`, `transpose_moving=`, `stationary must be in ['sbuf']` | 76 | 1, 2 | `nomatmul1`, `nomatmul2` |
| `could not be broadcast to indexing result`, `contraction dimension mismatch` | 42 | 4 | `mmshape` |
| `nki.isa` has no `scalar_mul` | 16 | 1 | `scale` widened |

At levels 1–2 the matmul errors are the wrong algorithm, not a matmul bug. So those examples show
the tool the level needs instead.

**`agent.py`'s `enrich()` gives wrong advice for the broadcast error.** It tells the model to write
`out[i*128:(i+1)*128, :] = tile`, but NKI writes results with `nisa.dma_copy`.
- All 29 of these errors in the logs were level 4, from an `nc_matmul` whose PSUM `dst` was smaller
  than its result.
- A reproduction on seat-35 with a (128, 128) `dst` for a 128×512 result raised
  `cannot reshape array of size 65536 into shape (128,128)`. That text is matched by `block`, whose
  "never reshape" advice is wrong for it too.
- The new agent should use `mmshape`'s note for these errors at matmul levels.

## scale
- levels: `all`
- when: `module 'nki\.(?:isa|language)' has no attribute '(?:multiply|mul|scale|divide|div|mean|average|scalar_mul|scalar_multiply|mult)'`
- error: `AttributeError: module 'nki.isa' has no attribute 'multiply'`
- input: `(32, 16)`
- expect: `a * 0.25`

Arithmetic on a whole tile is nisa.tensor_scalar with an op from nki.language, for example scaling
a tile by 0.25:

```python
@nki.jit
def scale_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=s, data=t, op0=nl.multiply, operand0=0.25)
    nisa.dma_copy(dst=out, src=s)
    return out
```

## nomatmul1
- levels: `1`
- when: `must be in \['(?:psum|sbuf)'\], got|nc_matmul\(\) got an unexpected keyword|contraction dimension`
- error: `AssertionError: dst must be in ['psum'], got sbuf`
- input: `(64, 32)`
- expect: `a.sum(axis=1, keepdims=True)`

Average pooling needs no matmul: nc_matmul multiplies two matrices, and pooling does not. Compute it on
the tile itself with a reduction instead, for example summing each row:

```python
@nki.jit
def row_sum_kernel(a):
    rows, cols = a.shape
    out = nl.ndarray((rows, 1), dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.sum(t, axis=[1], keepdims=True)
    o = nl.ndarray((rows, 1), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=s)
    nisa.dma_copy(dst=out, src=o)
    return out
```

## nomatmul2
- levels: `2`
- when: `must be in \['(?:psum|sbuf)'\], got|nc_matmul\(\) got an unexpected keyword|contraction dimension`
- error: `AssertionError: stationary must be in ['sbuf'], got psum`
- input: `(32, 2)`
- expect: `a[:, ::-1]`

This transpose reorders values inside each row, so it needs no matmul: nc_matmul multiplies two
matrices. Copy single columns with nisa.tensor_copy and nl.ds, for example swapping the two columns
of every row:

```python
@nki.jit
def column_swap_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s[:, nl.ds(0, 1)], src=t[:, nl.ds(1, 1)])
    nisa.tensor_copy(dst=s[:, nl.ds(1, 1)], src=t[:, nl.ds(0, 1)])
    nisa.dma_copy(dst=out, src=s)
    return out
```

## mmshape
- levels: `4, 5, 6, 7, 8`
- when: `could not be broadcast to indexing result|contraction dimension mismatch|cannot reshape array of size \d+ into shape`
- error: `ValueError: shape mismatch: value array of shape (65536,) could not be broadcast to indexing result of shape (16384,)`
- input: `(128, 64)`
- expect: `a.T @ a`

At a matmul level this usually means an nc_matmul shape is off. Both operands need the same K rows
(K <= 128, the partition dim), and dst is a PSUM tile of exactly (M, N): stationary's free size by
moving's free size. (If you reshaped a tensor, slice it instead.) For example, a 128x64 tile times
itself, a.T @ a:

```python
@nki.jit
def gram_kernel(a):
    K, M = a.shape
    out = nl.ndarray((M, M), dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((K, M), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    p = nl.ndarray((M, M), dtype=nl.float32, buffer=nl.psum)   # (stationary M, moving M)
    nisa.nc_matmul(dst=p, stationary=t, moving=t)
    s = nl.ndarray((M, M), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s, src=p)
    nisa.dma_copy(dst=out, src=s)
    return out
```

## block
- levels: `all`
- when: `dma_copy requires src and dst to have the same number of elements|cannot reshape array of size`
- error: `AssertionError: dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384`
- input: `(256, 96)`
- expect: `a[128:256, 32:96]`

To work on part of a tensor, allocate a tile with exactly the part's shape and slice the source to
match. Never reshape. For example, copying rows 128-255 and columns 32-95:

```python
@nki.jit
def block_copy_kernel(a):
    out = nl.ndarray((128, 64), dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((128, 64), dtype=a.dtype, buffer=nl.sbuf)   # exactly the slice's shape
    nisa.dma_copy(dst=t, src=a[128:256, 32:96])
    nisa.dma_copy(dst=out, src=t)
    return out
```

## chunks
- levels: `all`
- when: `partition dimension \d+ exceeds maximum`
- error: `AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128`
- input: `(384, 64)`
- expect: `a`

A tile holds at most 128 rows, so a taller tensor is moved in 128-row chunks, one tile per chunk,
for example copying a 384-row tensor:

```python
@nki.jit
def chunked_copy_kernel(a):
    rows, cols = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(rows // 128):
        t = nl.ndarray((128, cols), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=a[i * 128:(i + 1) * 128, 0:cols])
        nisa.dma_copy(dst=out[i * 128:(i + 1) * 128, 0:cols], src=t)
    return out
```

## rows
- levels: `all`
- when: `must have at least 2 dimensions`
- error: `AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim)`
- input: `(64, 32)`
- expect: `a.sum(axis=1, keepdims=True)`

Every tile is 2-D, rows first. A result with one value per row has shape (rows, 1), for example
summing each row:

```python
@nki.jit
def row_sum_kernel(a):
    rows, cols = a.shape
    out = nl.ndarray((rows, 1), dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.sum(t, axis=[1], keepdims=True)
    o = nl.ndarray((rows, 1), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=s)
    nisa.dma_copy(dst=out, src=o)
    return out
```

## columns
- levels: `2`
- when: `Out-of-bound access for tensor .* on dimension 0`
- error: ``AssertionError: Out-of-bound access for tensor `unnamed` on dimension 0: index range [0, 127] exceed dimension size of 32.``
- input: `(32, 2)`
- expect: `a[:, ::-1]`

shape2D = (F1, F2) describes the F1*F2 values inside each row. The first axis (rows) is never
indexed with F1 or F2: keep it whole with `:` and move along the second axis with
nl.ds(start, size). For example, swapping the two columns of every row:

```python
@nki.jit
def column_swap_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    s = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s[:, nl.ds(0, 1)], src=t[:, nl.ds(1, 1)])
    nisa.tensor_copy(dst=s[:, nl.ds(1, 1)], src=t[:, nl.ds(0, 1)])
    nisa.dma_copy(dst=out, src=s)
    return out
```
