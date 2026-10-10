# nki-sched: write a schedule, not a kernel

You do not write NKI. You write a **schedule**: a short Python function that rewrites a naive loop
program step by step. Each call is checked; the final program is emitted as a verified NKI kernel.
A failed call raises an error that says what rule you broke; fix that call (and what depends on it).

```python
def schedule(s, hw):
    ...   # calls on s; use hw.pmax / hw.stationary_fmax / hw.moving_fmax, never the literals 128 / 512
```

`PSUM`, `SBUF` and `hw` are already in scope. No imports. Reply with one python block holding `schedule`.
Always take tile sizes from `hw`: the schedule is first replayed on a tiny fake chip (pmax=4,
fmax=4/8) and checked numerically after every call; a hardcoded 128 fails there.

## The program you start from (matmul `C[m,n] = cast(sum_k lhsT[k,m] * rhs[k,n])`)

Sizes K, M, N are symbolic. `lhsT` is `[K,M]` (contraction on axis 0), `rhs` is `[K,N]`, `C` is `[M,N]`.
Loop names are `stage.var`, and every handle is just such a name (a string):

```
alloc matmul : f32[M,N] @ HBM                    # accumulator stage
for matmul.init.m, matmul.init.n: matmul[m,n] = 0      # init nest
for matmul.m: for matmul.n: for matmul.k: matmul[m,n] += lhsT[k,m] * rhs[k,n]    # update nest
for C.m: for C.n: C[m,n] = cast(matmul[m,n])     # output nest
```

## Primitives (all take loop/buffer names as strings)

| call | effect |
|---|---|
| `o, i = s.split(loop, factor, names=("o","i"), perfect=True)` | `loop -> o*factor + i`; `perfect=True` asserts extent % factor == 0 (needed for symbolic M,N,K). Returns the new names; the old name stops existing |
| `s.reorder(a, b, c, ...)` | permute a perfectly nested chain of loops, outermost first. Legal for reductions (`matmul.k` may move freely); refused if it reverses a dependence |
| `s.compute_at(stage, at=loop)` | move stage `matmul` (alloc + init + update nests) inside loop `at`, shrunk to the region consumed there |
| `s.set_memory(buf, PSUM or SBUF)` | PSUM: f32, axis 0 <= `hw.pmax`, free size <= one bank (`hw.moving_fmax` f32). SBUF: axis 0 <= `hw.pmax` |
| `s.stage_in(tensor, at=loop, mem=SBUF, name="x_sb")` | copy the window of `tensor` read inside `at` into a new buffer at the top of `at` (HBM->SBUF DMA); reads are redirected |
| `s.stage_out(tensor, at=loop, mem=SBUF, name="c_sb")` | writes to `tensor` inside `at` go to a new buffer, DMA'd out after `at` |
| `s.replace(loop, "ns.tensor.matmul")` | tensorize: the perfect 3-loop nest rooted at `loop` (k, m, n in any order) around one `+=` becomes one hardware matmul. Needs: dst in PSUM, both operands in SBUF laid out `[k,m]` and `[k,n]`, k <= `hw.pmax`, m <= `hw.stationary_fmax`, n <= `hw.moving_fmax` |
| `s.fold_init(buf)` | drop the zero-init nest of a PSUM accumulator; the first matmul overwrites |
| `s.hoist(buf, to=loop_or_None)` | move the fill of a `stage_in` buffer out to loop `to` (`None` = top of kernel). If loops between grow the window the buffer grows to hold it all (K tiles fold into axis 1); loops that do not index the source just reuse the data. Fewer HBM reloads |
| `s.fold(buf)` | re-layout an on-chip temp whose axis 0 is `T*128` rows as `[128, T, ...]` (row `128*t+r` -> `[r, t]`). Needed for a block of several partition tiles, e.g. a 512-row PSUM accumulator. Every access to axis 0 must be `128*q + r`: split the loop that indexes it by `hw.pmax` first |
| `s.set_dma(buf, dge="hwdge", engine="sync")` | how DMAs that fill/drain `buf` get descriptors. `hwdge` runs on the `sync` or `scalar` queue and keeps GpSimd free (default `swdge` burns the GpSimd engine); give different buffers different queues to run them in parallel |
| `s.shard(loop, cores)` | SPMD over NeuronCores: `loop -> blk*cores + core`, `core = nl.program_id(0)`, kernel launched as `kernel[cores](...)` (emitted as `LNC = cores`). The loop's iterations must be independent |
| `s.mark(loop, "affine"/"sequential"/"static")` | optional assertion about loop kind |
| `s.loops(stage)` -> names, `s.show()` -> current program text | inspection only |

The cast-out nest `C_sb[m,n] = cast(matmul[m,n])` between PSUM and SBUF is turned into a vector copy
automatically; you do not tensorize it.

## Worked example: tiled matmul (nkibench level 4)

```python
def schedule(s, hw):
    mo, mi = s.split("C.m", hw.stationary_fmax, names=("mo", "mi"), perfect=True)
    no, ni = s.split("C.n", hw.moving_fmax, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mi, ni)
    s.compute_at("matmul", at=no)                       # one output tile at a time
    ko, ki = s.split("matmul.k", hw.pmax, names=("ko", "ki"), perfect=True)
    s.reorder(ko, ki, "matmul.m", "matmul.n")              # k outermost: accumulate across ko
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at=ko, mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at=ko, mem=SBUF, name="rhs_sb")
    s.stage_out("C", at=no, mem=SBUF, name="C_sb")
    s.replace(ki, "ns.tensor.matmul")
    s.fold_init("matmul")
```

Order matters: `compute_at` before splitting `matmul.k`; `set_memory` before `replace`; `stage_in` after
`compute_at` (so the window is a tile); `replace` before `fold_init`.

## Optimising

The checker also counts HBM bytes against the floor (each input and the output moved once).
Work out how many times each operand is loaded: a load inside a loop that does not index it is
repeated on every iteration of that loop, and the cost grows with the operand's size. Loop order
decides which loads get repeated, and `hoist` moves a load out past loops (`to=None` = out of all of
them, SBUF capacity permitting). Use `s.show()` to see where each `dma_copy` ends up. After a failure,
the `s.show()` output in the report is the program as of the last successful call: read the loop and
buffer names from it.

## Errors

A `ScheduleError` names the primitive and the violated rule; the report also gives the program state
before the failing call and the calls that succeeded. Names that were split no longer exist: use
the returned `o, i` names.

## When nothing fails but the checker is not satisfied

A schedule can apply cleanly and still miss the HBM-traffic ceiling. Then there is no error to fix:
the schedule itself needs to change. Keep what is correct (tiling, staging, tensorize), take the
reported traffic ratio and the emitted program, find which loads are repeated most, and restructure
loop order and hoisting to cut them. Re-applying the same hoists in a different place is not a new
idea; if a change leaves the ratio unchanged, the loads you moved were not the expensive ones.

## Fast matmul (see `examples/fast_mm.py`)

For large bf16 matmuls the winning shape is a `BM x BN` output block held in PSUM (4 x 2 tiles of 128 x 512 = 8 PSUM banks) while K streams by in groups of `KT` tiles:
split `C.m` into (block, tile, 128) and `C.n` into (block, BN), `compute_at("matmul", at=no)`, split the
accumulator loops (init and update) down to tiles, `fold("matmul")`, then `set_memory(PSUM)`,
`stage_in` at the k-tile loop and `hoist` to the k-group loop, `stage_out("C", at=<m tile loop>)`,
`replace`, `fold_init`. Then `set_dma` (hardware DGE) and `shard` the outermost block loop over 2 cores.
Measured on trn2, 4096^3 bf16: tiled l4 7.5 ms (18 TFLOP/s) -> blocked 1.84 ms (75 TFLOP/s, 95% of one
core's peak) -> sharded over 2 cores ~0.98 ms (140 TFLOP/s).
