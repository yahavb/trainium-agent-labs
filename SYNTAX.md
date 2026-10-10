# nki-sched — schedule language draft (v0)

Status: **design draft, nothing here is implemented.** Names are provisional. Everything marked `[verify]` depends on an NKI capability that was not confirmed against the installed package (nki 0.6.0, see PLAN §3).

## 0. Two program levels (and their names)

| level | what it is | instruction spelling | where it lives |
|---|---|---|---|
| **ns** (nki-sched IR) | the explicit loop program the schedule rewrites; engine-explicit | `nc.<engine>.<inst>(...)`, e.g. `nc.tensor.matmul`, `nc.vector.tensor_copy`, `nc.sync.dma_copy` | what `print(k.ir)` shows; what `replace`/`stage_*` create |
| **NKI source** (emitted) | Python that `@nki.jit` compiles | `nisa.<inst>(...)`, `nl.<...>` (real NKI names, e.g. `nisa.nc_matmul`, `nisa.tensor_copy(..., engine=...)`) | `k.source`; what `nkibench` checks |

The engine is part of the ns-level call, not an annotation: `nc.tensor` (PE array), `nc.vector`, `nc.scalar`, `nc.gpsimd`, `nc.sync` (DMA queues; an IR label — NKI has no such name, it maps to `dma_copy(..., engine=…)`/`dge_mode`, `[verify]`). The lowering ns → NKI source is a table lookup (PLAN §8). Why the real NKI names are kept at the lowest level: the emitted code must be exactly what `reference_level4.py` looks like. (`nl` is *not* used as the name of this level: it is already `nki.language`, and emitted code uses both `nl.*` and `nisa.*`.)

Design rule: *the syntax is Halide/TVM-flavoured names (`split`, `reorder`, `compute_at`, `stage_in`) on top of Exo-style semantics (each call is a checked rewrite of an explicit loop IR, references are cursors that forward).*

---

## 1. The reference implementation (torch)

The high-level spec is plain torch. It is both the **input to the compiler** (traced) and the **correctness oracle** (run eagerly).

```python
import torch, nki_sched as nks
from nki_sched import hw
from nki_sched.mem import HBM, SBUF, PSUM
from nki_sched import nc                      # the ns-level instruction namespace: nc.<engine>.<inst>

# Dims are symbolic in the IR; the example shapes are only used to trace and to run the oracle.
K, M, N = 512, 256, 1024

def matmul(lhsT: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:   # [K,M], [K,N]
    return (lhsT.T.float() @ rhs.float()).to(torch.bfloat16)          # -> [M,N]

prog = nks.trace(matmul,
                lhsT=nks.arg(("K", "M"), "bf16", example=(K, M)),
                rhs =nks.arg(("K", "N"), "bf16", example=(K, N)),
                name="nki_matmul_tiled_")      # kernel entry name expected by nkibench
print(prog)                                    # the naive IR below
```

Layout: the spec takes `lhsT` (K on axis 0), same as `nkibench`/NKI tutorial kernels, so the contraction axis is already the partition axis for both operands. Taking plain `A[M,K]` is supported later via a layout primitive (§5).

### Naive IR that `trace` produces (Halide-style: stages, breadth-first, "root" allocs)

```
stage acc   : f32[M,N]    = Σ_k lhsT[k,m] * rhs[k,n]      # reduction stage: init + update
stage C     : bf16[M,N]   = cast_bf16(acc[m,n])           # pointwise stage (the output)

# lowered to explicit loops (this is the program every schedule starts from)
alloc acc : f32[M,N] @ HBM
for acc.m in 0..M:           # init
  for acc.n in 0..N:
    acc[acc.m, acc.n] = 0
for acc.m in 0..M:           # update
  for acc.n in 0..N:
    for acc.k in 0..K:
      acc[acc.m, acc.n] += lhsT[acc.k, acc.m] * rhs[acc.k, acc.n]
for C.m in 0..M:
  for C.n in 0..N:
    C[C.m, C.n] = cast_bf16(acc[C.m, C.n])
```

Loops are auto-named `stage.var`; renamed by `split`/`fuse`. This IR is directly executable by the numpy interpreter — it is the first thing diffed against `matmul()` in torch.

---

## 2. References: how to point at loops, tensors, statements

Three complementary mechanisms; all produce **cursors** (opaque handles into the *current* program) and all cursors **forward** across rewrites (PLAN §6.3).

```python
@nks.schedule
def sched(s: nks.Sched):
    # (a) by auto-name — Halide/TVM style, the common case
    m = s.loop("C.m");   k = s.loop("acc.k")
    A = s.buf("acc");    st = s.stage("acc")          # tensor / stage references

    # (b) by return value — primitives hand back the new cursors
    mo, mi = s.split(m, 128)                          # old handle `m` now forwards to `mo` (outer)

    # (c) by pattern — Exo style, for things with no name
    mm_nest = s.find("for ki in _: _")                # first loop with that name
    first_upd = s.find("acc[_] += _")                 # a statement pattern
    second = s.find("for _ in _: _ #1")               # `#n` picks the (n+1)-th match
    ld = s.find_all("lhsT_sb[_] = lhsT[_]")           # many=True

    # navigation / gaps (Exo cursor API)
    s.parent(mm_nest); s.body(mo); s.next(first_upd); s.before(mo); s.after(mo)

    # inspection (never mutates)
    s.extent(mo); s.is_reduction(k); s.window(ld[0]); s.footprint(s.buf("lhsT_sb"))
```

Forwarding rules (precise semantics in PLAN §6.3): after `mo, mi = s.split(m, 128)`, the old handle `m` forwards to the outer loop `mo` (its index var is now `mo*128+mi`); using `m` where the split matters (e.g. a later `reorder`) is an error with a hint ("`m` was split; use `mo` or `mi`"), never silently ambiguous. Handles to deleted statements become `Invalid` and raise on use.

---

## 3. Primitive catalogue (draft)

Every primitive: `(preconditions checked) → rewrite → (cursor forwarding fn)`. Derived ops are library code built from these (Exo 2 philosophy), so the trusted core stays small.

### Core (trusted, each with its own legality check — PLAN §7)

| Primitive | Meaning | Legality |
|---|---|---|
| `split(loop, f, tail="cut"\|"guard"\|"perfect")` | `i → (io, ii)`, `i = io*f+ii` | `perfect`: extent ≡ 0 mod f proven; else tail handling per option |
| `reorder(l1, l2, ...)` | permute perfectly-nested loops | no dependence reversed; **reduction loops of an associative op may move freely** (Halide §3); fp-reassoc flagged |
| `fuse(l1, l2)` / `unroll(loop)` | as usual | const extent for unroll |
| `fission(loop, at=stmt)` / `fuse_stmts` | split/merge loop bodies | stmt-level commutativity |
| `compute_at(stage, at=loop)` | Halide `f.compute_at(g, y)` / TVM `s[f].compute_at(s[g], ax)`: move the producer's loop nests from the root to *inside* the consumer loop `at`; each iteration computes only the region the consumer needs there (interval analysis), and by default its storage shrinks/moves with it | producer region ⊆ computed region (Halide bounds inference) |
| `store_at(stage_or_buf, at=loop)` | allocation granularity | enclosing; live range covers uses |
| `stage_in(tensor, at=loop, mem=, name=, window=None)` | cache_read: copy window of `tensor` into new buffer in `mem` before `loop` body; rewrite uses. **The copy is created directly as the right ns instruction** from (src mem → dst mem): HBM→SBUF `nc.sync.dma_copy`, PSUM→SBUF `nc.vector.tensor_copy` (cast implied by dst dtype); no copy loop nest, no separate lowering step | window inferred (interval analysis) or checked ⊇ accesses; capacity check |
| `stage_out(tensor, at=loop, mem=, name=)` | cache_write/write-back (same implicit instruction choice) | same; `accum=True` form = zero/accumulate/add-back (Exo `stage_mem(accum=)`) |
| `set_memory(buf, HBM\|SBUF\|PSUM)` | memory space of an allocation | partition-dim rule (axis 0 ≤128 on SBUF/PSUM), PSUM ⇒ fp32 and ≤ bank size, accessed only by allowed instrs |
| `expand_dim(buf, size, idx)` / `lift_alloc` / `sink_alloc` | give a buffer a leading rotating dim; move alloc | indexing in bounds |
| `fold_init(stage)` | drop the zero-init nest; first matmul overwrites (`accumulate=None`) | init is a full-tile store of 0 immediately preceding the accumulate nest, target in PSUM |
| `replace(block, instr)` | **tensorize**: match block against instr's semantic body by unification; emit instr call | unifier succeeds ⇒ equivalence by construction |
| `mark(loop, AFFINE\|SEQUENTIAL\|STATIC)` | **optional assertion/override.** Loop kind is *inferred* from the dependence analysis at emission (→ `nl.affine_range / sequential_range / static_range`); `mark` checks an expectation (error if the analysis disagrees) or forces a more conservative kind | AFFINE: no loop-carried dep other than PSUM-accumulate reduction; STATIC ⇒ unroll |

### Derived (library, built on the core)

| Op | Built from | Intent |
|---|---|---|
| `tile(l1, l2, f1, f2)` | `split`×2 + `reorder` | |
| `hoist(buf, to=loop)` | re-`stage_in` at an outer loop with larger inferred window + delete old | the "hoist loads" optimization |
| `multibuffer(buf, depth, along=loop)` | `expand_dim` + index by `loop % depth` | rotating buffers (SBUF double-buffer, PSUM ping-pong) |
| `pipeline(loop, stages=[[...],[...]], lag=1)` | `fission` + `shift_loop` + peel prologue/epilogue + `multibuffer` | software pipelining; legality = WAR/RAW distance < depth |
| `overlap(stmtA, stmtB)` | `reorder_stmts` + engine annotation | assert independent & on different engines/queues, interleave in program order |
| `set_engine(stmt, nc.vector\|nc.scalar\|nc.gpsimd)` | move a statement whose instruction exists on several engines (e.g. `tensor_copy`) | instr must be legal on that engine; verified in 0.6.0: `tensor_copy`/`dma_copy` accept `engine=`, `activation` does not (fixed engine) |
| `transpose_input(arg, via="dma"\|"pe")` | insert layout stage | `A[M,K] → lhsT[K,M]` (`nisa.dma_transpose` / `nisa.nc_transpose` exist in 0.6.0) |

---

## 4. Worked schedules — the matmul ladder from `nkibench.py`

Each schedule is a Python function; constants (`BM`, `depth`…) are plain parameters, so a schedule is also a **tunable template** (TVM knobs).

### 4.1 Level 4 — tiled matmul (must lower to `reference_level4.py`'s structure)

```python
@nks.schedule
def l4(s: nks.Sched):
    # tile the output; mi/ni match PE-array limits: 128 stationary-free, 512 moving-free
    mo, mi = s.split("C.m", hw.GEMM_STATIONARY_FMAX)    # 128
    no, ni = s.split("C.n", hw.GEMM_MOVING_FMAX)        # 512
    s.reorder(mo, no, mi, ni)

    # one accumulator tile per output tile; bounds inferred: acc_t is [128,512]
    s.compute_at("acc", at=no)

    # contraction tiled by the partition size and made outermost in the update nest
    ko, ki = s.split("acc.k", hw.PMAX)                  # 128
    s.reorder(ko, ki, "acc.m", "acc.n")                 # legal: + is associative (reduction)

    # memory spaces
    s.set_memory("acc", PSUM)                           # f32, 128 partitions x 512 = one bank
    s.stage_in("lhsT", at=ko, mem=SBUF, name="lhsT_sb") # window inferred [ko*128:+128, mo*128:+128]
    s.stage_in("rhs",  at=ko, mem=SBUF, name="rhs_sb")  # window inferred [ko*128:+128, no*512:+512]
    s.stage_out("C",   at=no, mem=SBUF, name="C_sb")    # cast-out staging, then DMA to HBM

    # instruction selection
    s.fold_init("acc")
    s.replace(s.find("for ki in _: _"), nc.tensor.matmul)   # k=partition, m=stationary free, n=moving free

```

ns-level IR after the schedule (what `print(s.ir)` shows — and what the emitter consumes):

```
alloc C : bf16[M,N] @ HBM                              # kernel output (nl.shared_hbm)
for mo in affine(M/128):
  for no in affine(N/512):
    alloc acc : f32[128,512] @ PSUM
    for ko in affine(K/128):
      alloc lhsT_sb : bf16[128,128] @ SBUF
      alloc rhs_sb  : bf16[128,512] @ SBUF
      nc.sync.dma_copy(lhsT_sb, lhsT[ko*128:+128, mo*128:+128])
      nc.sync.dma_copy(rhs_sb,  rhs [ko*128:+128, no*512:+512])
      nc.tensor.matmul(dst=acc, stationary=lhsT_sb, moving=rhs_sb)      # accumulate=None
    alloc C_sb : bf16[128,512] @ SBUF
    nc.vector.tensor_copy(C_sb, acc)                                 # PSUM→SBUF + cast
    nc.sync.dma_copy(C[mo*128:+128, no*512:+512], C_sb)
```

The emitter lowers each `nc.<engine>.<inst>` to its NKI form (`nc.tensor.matmul` → `nisa.nc_matmul`, `nc.vector.tensor_copy` → `nisa.tensor_copy(..., engine=nisa.vector_engine)`, `nc.sync.dma_copy` → `nisa.dma_copy`). Emitted Python is the same shape as `reference_level4.py` (`nl.affine_range`, `nl.ndarray(..., buffer=nl.psum)`, `nisa.dma_copy(dst=, src=)`, `nisa.nc_matmul(dst=, stationary=, moving=)`, `nisa.tensor_copy`). **Acceptance test #1 is that this passes `nkibench.py --level 4 --check` and models ≈2.0× the HBM byte floor.**

### 4.2 Level 5 — hoist the big operand's loads out of the loop that doesn't index it (bar: ≤1.6× floor)

Hoisting a load out of `ko` alone saves nothing (each tile is still read once per output tile — L4 already does that). The saving comes from hoisting across `mo`/`no`: `rhs[:, no-strip]` does not depend on `mo`, so with `no` outermost it is loaded once per `no` instead of once per `(mo,no)`. `rhs` is the larger operand, so hoist that one (hoisting only `lhsT` strips gives 1.86× — fails the bar; see table).

```python
@nks.schedule
def l5(s):
    l4(s)
    s.reorder("no", "mo")                  # independent output-tile loops: any order is legal
    s.hoist("rhs_sb", to="no")             # window grows [128,512] -> [K,512] (interval analysis over ko, mo)
    s.fold_partition("rhs_sb", dim=0)      # K>128 can't live on the partition axis: buffer is [128, K/128, 512]
```

### 4.3 Level 6/7 — block M, N (and K) with several PSUM tiles live

This is a *re-placement*, not a reorder of the L4 nest: the accumulator is computed per **block** of `BM×BN` output tiles, and `acc`'s own loops are re-tiled with `kb` outermost. Everything is expressed with `split` + `compute_at` + `reorder` on loops that are perfect nests at that point (init is its own nest, so there is nothing between the loops).

```python
@nks.schedule
def l7(s, BM=None, BN=None, BK=None):
    M, N, K = s.size("M"), s.size("N"), s.size("K")          # symbolic: resolved when the kernel runs on a shape
    BM = BM or min(4, M // 128)                              # knobs are functions of the sizes
    BN = BN or min(2, N // 512)                              # (a fixed BM=4 would not divide M=128)
    BK = BK or min(4, K // 128)

    mb, mt = s.split("C.m", BM * 128);  nb, nt = s.split("C.n", BN * 512)
    s.reorder(mb, nb, mt, nt)
    s.compute_at("acc", at=nb)                               # acc block: [BM*128, BN*512] = BM*BN PSUM tiles
    s.set_memory("acc", PSUM); s.check_capacity()            # fails loudly if BM*BN tiles exceed PSUM

    am, an, ak = s.loops("acc")                              # acc.m, acc.n, acc.k restricted to this block
    kb, kt = s.split(ak, BK * 128)                           # K blocking: kb outermost, accumulate in PSUM across it
    ko, ki = s.split(kt, 128); mo, mi = s.split(am, 128); no, ni = s.split(an, 512)
    s.reorder(kb, ko, mo, no, ki, mi, ni)                    # legal: + is associative (reduction rule)

    s.stage_in("lhsT", at=kb, mem=SBUF, name="lhsT_sb")      # [BK*128, BM*128] folded to [128, BK, BM*128]
    s.stage_in("rhs",  at=kb, mem=SBUF, name="rhs_sb")       # [BK*128, BN*512] folded to [128, BK, BN*512]
    s.stage_out("C",   at=nb, mem=SBUF, name="C_sb")
    s.fold_init("acc")
    s.replace(s.find("for ki in _: _"), nc.tensor.matmul)
```

`l6` is the same schedule with `BK = K//128` (a single `kb` iteration — no K blocking). `l4` ≡ `BM=BN=1`, one `kb` per `ko`.

### 4.3.1 Predicted HBM traffic (static analysis; not yet run through nkibench)

Computed with the same accounting as `nkibench` (bytes moved by `dma_copy`, loads + the output store, against a floor of inputs + output), analytically from the schedule's staging windows. The model reproduces the L4 "2.0×" figure nkibench quotes for K=256, M=512, N=1024 — the sanity check that the accounting matches. Shapes are the four in `_MM_SHAPES`, ordered (K,M,N) = (128,128,512), (256,256,1024), (512,128,512), (256,512,1024):

| schedule | knobs | ratio vs floor on the 4 shapes | worst | bar |
|---|---|---|---|---|
| `l4` (≡ blocks of 1 tile) | — | 1.00 1.56 1.00 2.00 | 2.00 | none |
| `l5` hoist `rhs` strip (order no→mo) | — | 1.00 1.11 1.00 1.14 | **1.14** | ≤1.6 ✓ |
| hoist `lhsT` strip only (the *wrong* L5) | — | 1.00 1.44 1.00 1.86 | 1.86 | ≤1.6 ✗ |
| `l7`/`l6` strips blocked | BM=2, BN=2 | 1.00 1.00 1.00 1.29 | 1.29 | ≤1.25 ✗ |
| `l7`/`l6` strips blocked | BM=4, BN=1 | 1.00 1.11 1.00 1.14 | 1.14 | L6 ≤1.25 ✓, L7 ≤1.05 ✗ |
| `l7`/`l6` strips blocked (**defaults**) | BM=4, BN=2 | 1.00 1.00 1.00 1.00 | **1.00** | L6 ✓, L7 ≤1.05 ✓ |

Observations the plan must respect:
1. Blocked-strip traffic is `ceil(N/BN·512)·|lhsT| + ceil(M/BM·128)·|rhs| + |C|`; K-blocking (`BK`) does **not** change it when `acc` stays in PSUM across `kb` — so on the nkibench shapes L6 and L7 are indistinguishable by bytes once the blocks cover M and N. `BK` exists for SBUF *capacity* at larger K, not for these bars.
2. The defaults (BM=4, BN=2) keep 8 PSUM tiles live. Whether that fits depends on the PSUM bank count/size (`[verify]`, PLAN §3.2); if only fewer fit, L7's 1.05 bar fails on the largest shape (the BM=4, BN=1 row), and `check_capacity()` reports it — the static analysis catches this before any device run.
3. Both numbers in the "wrong" rows are what `s.hbm_bytes()` would print at schedule-authoring time.

### 4.5 Beyond the ladder — overlap (device-measured, not in the simulator)

```python
@nks.schedule
def l8(s, depth=2):
    l7(s)                                              # shape-derived BM,BN,BK defaults
    # SBUF double-buffering of loads against the matmul that consumes the previous block
    s.multibuffer("lhsT_sb", depth, along="kb"); s.multibuffer("rhs_sb", depth, along="kb")
    s.pipeline("kb", stages=[s.find_all("nc.sync.dma_copy(lhsT_sb, _)") + s.find_all("nc.sync.dma_copy(rhs_sb, _)"),
                             s.find_all("nc.tensor.matmul(_)")], lag=1)
    # PSUM ping-pong: evacuate tile t while TensorE accumulates t+1
    s.multibuffer("acc", 2, along="nb")
    s.pipeline("nb", stages=[[s.find("for mo in _: _")], [s.find("tensor_copy(C_sb, _)"),
                                                        s.find("dma_copy(C[_], C_sb)")]], lag=1)
    s.set_engine(s.find("nc.vector.tensor_copy(C_sb, _)"), nc.scalar)   # e.g. rebalance evacuation off the vector engine; tensor_copy accepts engine= (verified)
```

What `pipeline` *means on NKI 0.6.0*: no explicit sync API exists, so it is a **program-order + buffer-rotation transform**: the stage-0 work of iteration *i+lag* is emitted before the stage-1 work of iteration *i*, with `depth` rotating buffers so no WAR hazard exists; NKI's own dependence tracking then lets DMA/PE/Vector run concurrently. Whether we actually get overlap is an empirical question answered on device (PLAN §11, M5), not by the simulator.

---

## 5. Errors are part of the language

Lesson from this repo (`STATE.md`): the quality of the checker's message is the product. Every legality failure names the primitive, the violated rule, and the offending program locations:

```
ScheduleError in reorder(acc.k, acc.m) at l4.py:14
  cannot move 'acc.k' outside 'acc.m': 'acc.m' contains two statements in its body
  (the reduction nest and the cast nest 'C'), so the nest is not perfect.
  hint: compute_at("acc", at=...) first, or fission 'acc.m' so each nest is separate.

ScheduleError in reorder(mo, ko)
  would reverse dependence: acc_t[...] += ... (update, reads PSUM written in ko) vs tensor_copy(C_sb, acc) (reads after last ko)
  hint: the copy-out must stay outside the 'ko' loop.

ScheduleError in set_memory("acc", PSUM)
  acc has shape [256,512] f32: axis 0 (partition) = 256 > 128
  hint: split 'acc.m' by 128 and compute_at the inner tile.

ScheduleError in l6.check_capacity()
  PSUM: BM*BN = 8 live fp32 tiles of [128,512] need 8 banks; HardwareConfig.psum_banks = <probed value>   [hw numbers from the M0 probe, not assumed]
```

---

## 6. Debug/inspection surface

```python
k = nks.compile(prog, l7,                  # knobs default to functions of the shape
               
               check_each_step=True)      # after EVERY primitive: numpy-interpret IR vs torch eager on hostile inputs
print(k.ir)                               # scheduled IR           print(k.history[3])  # IR after 4th primitive
print(k.source)                           # emitted NKI python     k.schedule_log       # replayable primitive trace (JSON)
k.stats()                                 # static: hbm bytes, flops, arithmetic intensity, SBUF/PSUM footprint
k.verify_nkibench(level=7)                # runs nkibench check on the emitted file
```
