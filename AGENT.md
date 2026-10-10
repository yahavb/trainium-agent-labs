# nki-sched: write a schedule, not a kernel

You do not write NKI. You write a **schedule**: a short Python function that rewrites a naive loop
program step by step. Each call is checked; the final program is emitted as a verified NKI kernel.
A failed call raises an error that says what rule you broke. Fix that call and nothing else.

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
A load inside a loop that does not index it is re-read every iteration. To cut traffic: reorder the
output-tile loops so the loop that does *not* index an operand is innermost, then `hoist` that
operand's buffer to the loop outside it. Hoisting `to=None` keeps an operand fully resident in SBUF
(capacity-limited). After a failure, the `s.show()` output in the report is the program as of the last
successful call: read the loop and buffer names from it.

## Errors

A `ScheduleError` names the primitive and the violated rule; the report also gives the program state
before the failing call and the calls that succeeded. Names that were split no longer exist: use
the returned `o, i` names. Change only the failing call (and what depends on it).
