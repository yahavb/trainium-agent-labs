# nki-sched — a scheduling language on top of NKI

**Status: plan only. No implementation code exists yet.**
Companion docs: [`SYNTAX.md`](SYNTAX.md) (language draft with worked schedules), [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md) (what was read, what we take from each).

---

## 0. One-paragraph summary

We write the *algorithm* in torch (`lhsT.T @ rhs`), trace it to a small loop IR, and then author a **schedule** — a short Python program of checked rewrite primitives (`split`, `reorder`, `compute_at`, `stage_in`, `set_memory`, `replace`, `multibuffer`, `pipeline`, …) — that turns the naive loop nest into a tiled, SBUF/PSUM-staged, tensor-engine kernel. The result is emitted as NKI Python source and validated by (1) a numpy IR interpreter against eager torch after every primitive, (2) the existing `nkibench.py` simulator/traffic checks, (3) the device. **Architecture: Exo-style rewrite-based schedules over our own small Python IR, with Halide-style `compute_at`/bounds-inference and TVM-style memory scopes/tensorize; not MLIR (§9).** First target: reproduce the repo's level 4→7 matmul ladder from schedules; then double-buffering/pipelining; then attention.

Why this is worth building (beyond elegance): a schedule for the level-7 matmul is ~20 lines vs a ~100-line kernel, failures are *structured* (the primitive + rule that was violated), and the same schedule re-targets other shapes/dtypes by changing parameters. That matches this repo's finding that the checker's message quality drives agent success: an LLM that authors schedules gets legality errors by construction instead of silent numeric mismatches.

### Non-goals (v0)
Auto-scheduling/search (we only make schedules parameterisable), multi-core/LNC sharding, dynamic shapes, convs, training/backward, general torch graphs beyond a handful of ops.

---

## 1. Prior art → design decisions

(Detail and sources in `docs/PRIOR_ART.md`.)

| Question | Halide | TVM | Exo | **Decision** |
|---|---|---|---|---|
| What is a schedule? | args to a monolithic lowering | args to a monolithic lowering | sequence of rewrites of an explicit program | **Exo**: rewrites; every intermediate program is printable *and runnable* |
| Producer/consumer placement | `compute_at`/`store_at` + bounds inference | `compute_at`, `cache_read/write` | `stage_mem`, `fission`, + library `compute_at` (Exo 2) | **Halide front-end API, built from Exo-style core ops**; interval analysis infers windows |
| Hardware | fixed targets | memory scopes + tensorize (declared in tensor-expr lang) | memories/instrs/config as *user library*; `replace` by unification | **Exo+TVM**: NKI ISA & memories are a table in a library, no compiler special-casing |
| Latency hiding | n/a (sliding window, CPU) | virtual threads → dependence tokens | not in core | **No sync tokens in NKI 0.6.0** (verified): `pipeline`/`overlap` = reordering + buffer rotation; a token-injection pass slot is reserved |
| Safety | by construction (inferred bounds) | by construction of primitives, weak checking | effect analysis + SMT | **Layered**: primitive-local legality (affine/interval) + differential testing; z3 later |
| References | named vars/Funcs | stage objects & axes | pattern strings → **cursors with forwarding** | **Both**: `s.loop("C.m")`/`s.buf("acc")` for the common case; `s.find(pattern)` cursors for the rest; forwarding for all |

---

## 2. Scope and acceptance tests

**Matmul first.** We already have a graded ladder in this repo (`projects/02-kernel-agent/nkibench.py`), which gives us ready-made, objective acceptance tests with HBM-traffic bars:

| Milestone | Schedule must produce | Check |
|---|---|---|
| A4 | kernel equivalent in structure to `reference_level4.py` | `nkibench --level 4 --check` passes; traffic ≈ 2.0× floor |
| A5 | hoisted loads (`nki_matmul_hoist_load_`) | level 5 bar: ≤ 1.6× floor |
| A6 | M,N blocked (`..._block_free_dimension_`) | level 6 bar: ≤ 1.25× |
| A7 | M,N,K blocked (`..._fully_optimized_`) | level 7 bar: ≤ 1.05× |
| A8 | + SBUF double-buffer, PSUM ping-pong, pipelined | device latency ≤ A7 (**device-only**; the simulator cannot see overlap) |
| B | single-head attention (`nki_attention_`, nkibench level 8) | numerics + traffic; then multi-tile flash-style (stretch) |

Predicted traffic per worked schedule on all four shapes is in SYNTAX §4.3.1 (note: the naive-looking L5 and BM=BN=2 L6 choices would *miss* their bars; the static traffic analysis is how we catch that at authoring time).

Note the harness rejects kernels that call `matmul/dot/einsum` etc. and framework modules (torch/np) in the *emitted* code; our emitter must emit pure `nki`/`nisa`/`nl` and the entry names the harness expects (`nki_matmul_tiled_`, …).

---

## 3. NKI target model

### 3.1 Verified this session
* Installed on `seat-270`: `nki 0.6.0+31049202112`. `nki.isa` exports (relevant subset): `nc_matmul, nc_transpose, dma_copy, dma_transpose, tensor_copy, tensor_tensor, tensor_scalar, tensor_reduce, activation, activation_reduce, exponential, reciprocal, memset, iota, tensor_partition_reduce, core_barrier, tensor_engine, vector_engine, scalar_engine, gpsimd_engine, get_nc_version`. **No semaphore / dependence-token / explicit-wait API** appears; only `core_barrier` (cross-core).
* `nki.language` exports: `ndarray(shape, dtype, buffer=)`, buffers `sbuf, psum, hbm, shared_hbm, private_hbm`, loop constructs `affine_range, sequential_range, static_range, dynamic_range, fori_loop, while_loop, range`, `tile_size`, `program_id`, plus math (`softmax`, `rms_norm`, …).
* `nc_matmul(dst, stationary, moving, ..., accumulate=None, tile_position, tile_size, perf_mode)`: `dst = stationary.T @ moving`; operands in SBUF, `dst` in PSUM; both operands share partition size ≤128 (contraction); stationary free ≤128; moving free ≤512 on NC-v2/v3 (4096/8192 on v4); `dst` fp32 on v2/v3; `accumulate=None` ⇒ first write overwrites, later writes accumulate; mixing accumulate with non-matmul PSUM init is undefined on v2/v3.
* Tile caps used by `nkibench.py`: `pmax=128`, `gemm_stationary_fmax=128`, `gemm_moving_fmax=512`.
* The tutorial-style kernels in this repo use a new-style API: `nl.ndarray` allocs, `nisa.dma_copy(dst=,src=)`, `nisa.nc_matmul(dst=,stationary=,moving=)`, `nisa.tensor_copy`.

### 3.2 Not verified — must be checked on device before M2 (probe script in M0, risks in §12)
SBUF/PSUM capacities, PSUM bank count/size, DMA queue/engine selection kwargs, whether the compiler really overlaps across `affine_range` iterations with rotating buffers, NC-v3 (trn2) specifics. We encode all of these in a single `HardwareConfig` object (`nc_version`, caps, PSUM banks, SBUF bytes) filled from probes, so no number is hard-coded in primitives.

### 3.3 Consequences for the language
1. **Partition axis**: SBUF/PSUM buffers have axis 0 = partition ≤128. This is a *type-level legality rule* checked after every rewrite (`set_memory`, `stage_*`, `expand_dim`). Large logical extents on axis 0 must be folded (`[K,128] → [128,K/128,128]`; SYNTAX §4.2).
2. **Contraction on partition** for both matmul operands: a loop only matches `nc_matmul` if its reduction var indexes axis 0 of both SBUF operands — `replace` unification enforces it.
3. **PSUM** is an fp32 accumulator written only by matmul(/transposes); reads by Vector/Scalar engines (`tensor_copy` does the cast). `fold_init` models the first-write-overwrites semantics; the explicit-zero path is the fallback.
4. **Loop kinds** (`affine_range` vs `sequential_range` vs `static_range`) are *legality annotations with dependence meaning*, not syntax: `affine` asserts no loop-carried dependence other than PSUM-accumulate reduction.
5. **Overlap without sync API** ⇒ `pipeline`/`overlap` are semantics-preserving reorderings + rotating buffers; actual concurrency is up to NKI's compiler and must be *measured*.
6. **Layout**: NKI matmul wants `lhsT` (K-major). v0 interface: the torch spec takes `lhsT` (same as harness). `transpose_input` (via `dma_transpose` or PE `nc_transpose`) comes later as a derived layout op.

---

## 4. Architecture

```
 torch fn ──trace──► Algorithm IR ──naive lower──► Loop IR (v0) ──schedule primitives──► Loop IR (vN)
 (spec+oracle)       (stages, RDom)                (explicit loops,   (each = legality check +        │
        │                                           allocs @ HBM)     rewrite + cursor forward)       │
        │                                                                                              ▼
        │                                        hw legality pass ◄── memory/partition/PSUM rules   Emit NKI .py
        │                                                                                              │
        └────────────── eager torch (oracle) ──── diff ◄── numpy IR interpreter (every step) ◄────────┤
                                                    diff ◄── nki.simulate (nkibench) ◄────────────────┤
                                                    diff ◄── device (baremetal) + profile ◄───────────┘
```

Package layout (planned):

```
nki-sched/
  PLAN.md  SYNTAX.md  docs/PRIOR_ART.md
  nki_sched/
    frontend/   trace.py (torch.export → Algorithm IR), ops.py (op expansion rules)
    ir/         nodes.py (dataclasses), types.py, expr.py (affine exprs + simplifier),
                printer.py, parser.py (pattern strings), interp.py (numpy)
    analysis/   bounds.py (interval), deps.py (access sets, reduction-aware commute),
                footprint.py, traffic.py (hbm bytes/flops → reuses nkibench roofline)
    sched/      cursor.py (handles+forwarding), api.py (Sched), core/ (one file per primitive),
                derived/ (hoist, multibuffer, pipeline, tile, …), errors.py
    hw/         config.py (HardwareConfig), memories.py (HBM/SBUF/PSUM), nisa.py (instr table)
    emit/       nki_py.py (IR → NKI source), rules.py (post-emit self-check vs nkibench rules)
    verify/     diff.py (interp vs torch), hostile_inputs.py, nkibench_bridge.py
  tests/        per-primitive property tests, golden IR/source tests, ladder acceptance tests
  examples/     matmul_l4.py … matmul_l8.py, attention.py
```

Implementation language: Python 3.10+, `dataclasses` + hand-rolled pattern matching; deps: `torch`, `numpy`, (later) `z3-solver`/`islpy`. No LLVM build.

---

## 5. IR design

Two layers, because they serve different jobs.

### 5.1 Algorithm IR (the "what", Halide-flavoured)
```
Func  := name, dtype, vars[(name, extent)], body: PureExpr | Reduction(rdom, op=+,*,max,..., init, update_expr) | Cast/Pointwise
Prog  := inputs[Tensor], stages[Func] (DAG), output
```
`torch.export` → ATen graph → per-op expansion rule into `Func`s. For matmul: `aten.permute` folded into an index remap, `aten.mm` → reduction over an `RDom`, `aten.to`/`_to_copy` → cast stage. This is TVM's `te.compute` / Halide's `Func` — a few hundred lines, no general torch-graph ambitions.

### 5.2 Loop IR (the "how", Exo-flavoured) — the thing schedules rewrite

```
Proc    := name, args[Buffer], body: Block
Stmt    := Alloc(buf, shape, dtype, mem) | For(var, lo, hi, kind, body) | If(cond, body, orelse)
         | Assign(buf, idx, expr) | Reduce(buf, idx, expr, op) | Call(instr, args: Window[]) | Pass
Expr    := Const | Var | BinOp | Read(buf, idx) | Cast | WindowExpr(buf, [Point|Slice(lo,hi)])
Index   := affine over loop vars and size symbols (+ floordiv/mod by constants for multibuffer rotation)
Buffer  := name, shape[Expr], dtype, mem ∈ {HBM, SBUF, PSUM}
Instr   := name, semantic_body: Proc, emit: template, legal_mems: per-arg, shape_caps
```

Key properties:
* **Immutable, persistent tree with stable node ids** (structural sharing). A rewrite returns `(new_proc, forward_fn)`; this is what makes cursors work (§6.3) and makes `history[i]` free.
* **Explicit and executable**: the interpreter runs *any* version of the IR (including with `Call(instr)` nodes, by executing the instr's `semantic_body`). Correctness of intermediate states is therefore testable at every step, not only at the end.
* **Two instruction spellings.** The ns-level IR names instructions `nc.<engine>.<inst>` (engine is part of the call: `nc.tensor.matmul`, `nc.vector.tensor_copy`, `nc.sync.dma_copy`, `nc.scalar.activation`…). The final NKI source uses real NKI names (`nisa.nc_matmul`, …). Lowering ns → NKI source is a table lookup in the `Instr` entry (see §8).
* **Instruction table as data** (Exo/TVM): `nc.tensor.matmul` is an `Instr` with semantic body `for k,m,n: dst[m,n] += st[k,m]*mv[k,n]`, arg memories `(PSUM, SBUF, SBUF)`, caps `k≤128, m≤128, n≤512`, emit template `nisa.nc_matmul(dst=…, stationary=…, moving=…)`. Likewise `dma_copy`, `tensor_copy` (PSUM→SBUF + cast), later `nc_transpose`, `activation`, `tensor_reduce`, `exponential`.
* **Stage structure is retained as metadata** (which loops belong to which Func) so that Halide-style `compute_at` and name-based references (`"acc.k"`) remain available after lowering.
* Small affine simplifier (`expr.py`) to keep indices canonical so interval analysis and `replace` unification stay simple. Where affine isn't enough (tail guards), use explicit `If` + `cut`.

### 5.3 Analyses
* **Interval/bounds** (Halide): given a consumer region, infer producer region / staging window; used by `compute_at`, `stage_*`, `hoist`.
* **Dependence/commutation** (Exo-lite): read/write sets of statements as affine boxes, with a *reduction-aware* rule (updates with the same associative op to the same location commute). Conservative; reject when unsure.
* **Footprint** (SBUF/PSUM bytes per partition, from `HardwareConfig`) and **traffic** (HBM bytes, FLOPs, arithmetic intensity). Traffic is computed *statically from the IR* and cross-checked against nkibench's counted bytes — gives schedule-time assertions like `assert s.hbm_bytes() <= 1.25*floor`.

---

## 6. Schedule language (summary; full draft in SYNTAX.md)

### 6.1 Shape of the API
An imperative-looking facade (`s.split(...)`) over an immutable-IR history (so `s.history[i]`, replay, diff). A schedule is a Python function `def sched(s, **knobs)`; knobs make it a template.

### 6.2 Primitive set
Core (each independently checked): `split, reorder, fuse, unroll, fission, compute_at, store_at, stage_in, stage_out, set_memory, expand_dim, lift_alloc, sink_alloc, fold_init, replace, mark`. Derived (library code over core): `tile, hoist, multibuffer, pipeline, overlap, fold_partition, transpose_input, set_engine`. `stage_in`/`stage_out` emit the right ns copy instruction directly (no `lower_copies` step) and loop kinds are inferred at emission (`mark` only asserts). Table with legality rules in SYNTAX §3.

### 6.3 References and forwarding
* **Kinds**: loops (`"stage.var"` names, returned handles), buffers/stages (`s.buf`, `s.stage`), statements/blocks/gaps via patterns (`s.find("for ki in _: _ #1")`, `s.find("acc[_] += _")`, `many=True`), plus navigation (`parent/body/next/before/after`) and inspection (`extent, window, footprint, is_reduction`).
* **Handle = (stable node id, version)**. Every primitive returns `forward: id → id | Invalid | Split(outer,inner) | …`. Node ids of untouched nodes are preserved by the persistent tree; for restructured nodes the primitive registers an explicit mapping (e.g. `split`: `old_loop ↦ outer`, with inner reachable as `outer.inner`). Using a handle resolves it through the forwarding chain from its birth version to the current one; if the node was destroyed → clear `InvalidHandle` error naming the primitive that destroyed it. Ambiguous cases (loop that was split) error with a hint rather than guessing.
* Patterns use Exo's string syntax (`_` wildcard, `#n` selector, `;` sequences) so they compose with a future text dump of the IR.

### 6.4 Pipelining design (the NKI-specific part)
`pipeline(loop, stages, lag)` is *derived*: `fission` the loop body into stage groups → `shift_loop` later groups by `lag` iterations → peel prologue/epilogue (or guard) → require each buffer crossing a stage boundary to be `multibuffer`ed with `depth ≥ lag+1`. Because each step is a core rewrite with its own check, the composite is safe without a bespoke proof. The WAR/RAW distance condition is checked by the dependence analysis on the rotated indices. On NKI 0.6.0 this is where our semantics end: whether engines overlap is measured (§11, M5).

---

## 7. Correctness strategy ("correct assuming the transformations are correct" → make that assumption cheap)

1. **Primitive-local legality** (static, deterministic): each core primitive states and checks its precondition — e.g. `split` (divisibility or tail strategy), `reorder` (no dependence reversed; reduction exemption requires associativity; fp reassociation recorded and gated by tolerance), `stage_*` (window ⊇ accesses by interval analysis; capacity), `set_memory` (partition ≤128, PSUM fp32/bank), `replace` (unifier succeeds + instr caps), `mark(AFFINE)` (no loop-carried dep). Failure ⇒ structured `ScheduleError` (SYNTAX §5).
2. **Hardware legality pass** after every rewrite (cheap structural check): memory-space typing of every access (e.g. `nc_matmul` operands in SBUF, dst in PSUM), partition rule, footprint ≤ capacity.
3. **Differential testing at every step** (`check_each_step=True`): interpret the current IR on hostile inputs and compare to eager torch. Inputs follow the harness's lessons: non-divisible/prime/size-1 dims (where supported), large magnitudes, zeros, identical rows; tolerance stated per dtype (fp32-accumulate vs bf16 output ⇒ `rtol≈2e-2` like nkibench; tighter for fp32 pipelines).
4. **Soundness testing of the compiler itself**: property-based fuzzing — random small loop programs × random legal primitive sequences → interpreter equivalence. This is how the "assuming primitives are correct" premise gets exercised; bugs found here are compiler bugs, not user bugs.
5. **Emit-level checks**: emitted source is run through `nkibench` rule checker + simulator + traffic bar; device run compares to torch.
6. **Later (optional)**: z3/islpy for exact bounds & divisibility (replace interval analysis where it is too coarse), à la Exo.

Explicit caveat: numeric equivalence under reduction reordering is only up to fp reassociation; we track which rewrites reassociate and report them.

---

## 8. Lowering to NKI

Order of passes after scheduling:
1. **Normalize/simplify** (index canonicalization, drop trivial loops/ifs, constant fold).
2. **Hardware legality + footprint check** (final).
3. **Loop-kind inference** (implicit; `mark` only asserts/overrides): independent loops and PSUM-accumulate reduction loops → `nl.affine_range`; loops with carried deps → `nl.sequential_range`; `unroll`ed → `nl.static_range` (or python `range` expansion).
4. **Emission** (`emit/nki_py.py`), IR → Python source text via a small structured printer:
   | IR | NKI |
   |---|---|
   | `Alloc(mem=HBM, output)` | `nl.ndarray(shape, dtype=…, buffer=nl.shared_hbm)` |
   | `Alloc(mem=SBUF/PSUM)` | `nl.ndarray(shape, dtype=…, buffer=nl.sbuf / nl.psum)` |
   | `For(kind)` | `for i in nl.affine_range(n):` etc. |
   | `nc.sync.dma_copy(dst, src)` | `nisa.dma_copy(dst=…, src=…)` with window → slice syntax `x[a:b, c:d]` |
   | `nc.tensor.matmul(dst, stationary, moving)` | `nisa.nc_matmul(dst=…, stationary=…, moving=…)` (+ `accumulate=` when explicit) |
   | `nc.vector.tensor_copy(dst, src)` (or `nc.scalar/gpsimd`) | `nisa.tensor_copy(dst=…, src=…, engine=nisa.<engine>_engine)` (cast implied by dst dtype) |
   | Rotating buffer index `b[i % d]` | emitted as `b[i % d]` / or `d` separately named buffers via `static_range` unroll `[verify which form NKI accepts]` |
5. **Self-check**: re-parse emitted source with `ast`, run nkibench `check_rules` (no banned calls/framework modules, correct `@nki.jit` entry name), assert no tile exceeds caps.

Shape-genericity: the emitter keeps sizes symbolic: sizes come from `tensor.shape` at NKI trace time, trip counts like `M // (BM*128)` are plain Python ints in the emitted source, divisibility assumptions become `assert`s (mirroring `reference_level4.py`), and shape-dependent knob defaults are emitted as `min(...)` expressions.

Tail handling: NKI tile shapes are static, so non-divisible extents require `split(tail="cut")` (peeled static-shape remainder), never a dynamic-size tile. v0 accepts only divisible shapes (all level 4–7 shapes are), `cut` comes with M3.

Sync: none emitted (none exists in 0.6.0). A reserved pass `inject_sync` (TVM-style token insertion) stays a no-op unless a lower-level API appears.

---

## 9. MLIR vs. own IR

**Options**
* **A. Own small Python IR (recommended).**
* **B. MLIR**: torch-mlir/`torch.export`→`linalg`; schedule via the **Transform dialect** (`tile_using_for`, `fuse`, `promote`/bufferization, `scf` pipelining); custom `nki` dialect for memories/instructions; custom Python emitter.
* **C. Hybrid**: own IR now, keep concepts mappable to linalg/transform; optional torch-mlir frontend and/or MLIR export later.

**Why A**
1. *No NKI backend exists in MLIR.* The target is **Python source** using a Python DSL, not machine code. We would write a custom dialect (memory spaces, partition-axis types, NKI instrs) *and* a custom emitter — the parts of MLIR that actually save effort (LLVM/SPIR-V codegen, passes to machine code) are unused.
2. *The interesting content is the schedule semantics, which MLIR doesn't give us.* Transform-dialect handles are invalidated when consumed; we want Exo-style **forwarding** so schedule scripts stay readable. `replace`-by-unification, reduction-aware commutation, PSUM/partition legality, and pipelining-as-composite-of-core-ops all have to be built regardless.
3. *Iteration speed and hackability*: pure Python, `pytest`, a numpy interpreter, debuggable in a notebook, no LLVM build or Python-binding churn; the whole thing must run on a laptop or the Neuron instance without a toolchain.
4. *Scale*: matmul → attention needs ~tens of op/loop forms, not a general compiler. The cost of a bespoke IR is bounded (~few thousand lines); the cost of MLIR infra (build, dialect definition via ODS/TableGen, bindings) is not.
5. *Prior art agrees*: Exo and Halide are small custom IRs and ship real kernels.

**Why not "never MLIR"**: keep the door open by (i) mirroring linalg concepts (stages ≈ `linalg.generic` with indexing maps; memory spaces ≈ memref memory-space attr; our `compute_at/stage_in/split` ≈ `fuse/promote/tile_using_for`), (ii) keeping the IR printable/parsable so an MLIR exporter is a mechanical translator. **Revisit if**: we need to feed other backends, need polyhedral machinery MLIR already has, want to reuse upstream pipelining/bufferization passes, or the op set grows past a few dozen. torch-mlir remains the alternative *frontend* (`torch → linalg-on-tensors`) if `torch.export` expansion rules start to hurt.

(Caveat: MLIR Transform-dialect details here are from background knowledge, not re-read this session.)

---

## 10. Torch frontend

* **Spec = ordinary torch function** with example `nks.arg(shape, dtype)` inputs. Eager execution of the same function is the oracle for every verification step.
* **Tracing**: `torch.export.export(fn, example_inputs)` → ATen graph (fallback: `torch.fx.symbolic_trace` for the tiny op set). A registry maps ATen ops to Algorithm-IR constructors:
  * v0: `aten.t/permute` (fold into index remap), `aten.mm`/`matmul`, `aten._to_copy`/`to` (cast stage), elementwise add/mul/relu.
  * attention: `aten.softmax` expanded into `max`, `sub`, `exp`, `sum`, `div` stages (reductions with `max`/`+`), scale.
* **Shapes**: `trace` records concrete example shapes only to validate and to run the oracle. The IR keeps `M, N, K` **symbolic**, and the emitted kernel is *shape-generic* like `reference_level4.py`: it reads `K, M = lhsT.shape` at NKI trace time, computes trip counts in Python, and `assert`s the divisibility the schedule relied on (e.g. `M % (BM*128) == 0`). This is required because `nkibench --check` runs **one** kernel file over all four `_MM_SHAPES`. Schedule knobs are therefore functions of the sizes (`BM = min(4, M//128)`), also evaluated in-kernel, never fixed constants that would not divide the smaller shapes (e.g. a fixed `BM=2` fails at M=128, a fixed `BK=4` fails at K=128).
* **Layout decision (v0)**: spec takes `lhsT[K,M]`. Plain `A[M,K]` specs require a layout stage; provided later by `transpose_input` (`dma_transpose` or PE `nc_transpose` via identity — `nl.shared_identity_matrix` exists in 0.6.0).
* **Unsupported ops** fail at trace time with the op name (no silent fallback).

---

## 11. Roadmap, with verification at each step

| M | Deliverable | Done when |
|---|---|---|
| **M0** (≈ days) | Skeleton + `HardwareConfig`; probe script that fills it on `seat-270`; trace `lhsT.T@rhs` → Algorithm IR → naive Loop IR; numpy interpreter; printer | interpreter == eager torch on `_MM_SHAPES` + hostile shapes |
| **M1** | Persistent IR + cursors/forwarding; core loop prims `split/reorder/fuse/unroll`; dependence & reduction rule; `check_each_step`; fuzz harness | fuzz passes; reorder-of-reduction accepted, reorder-over-init rejected with good message |
| **M2** | `compute_at/store_at`, interval bounds, `stage_in/out`, `set_memory`, `fold_init`, instr table + `replace` (unification for the 3 instrs), hw legality pass, **NKI emitter**, `mark_defaults` | **A4**: emitted kernel passes `nkibench --level 4 --check`; traffic ≈ 2.0× |
| **M3** | `hoist`, `fold_partition`, `split(tail="cut")`, capacity checks, traffic model | **A5, A6, A7** pass bars (1.6 / 1.25 / 1.05) |
| **M4** | `multibuffer`, `pipeline`, `overlap`, `set_engine` | IR-level equivalence (interpreter); emitted kernels correct in simulator |
| **M5** | Device runs on `seat-270` (baremetal + profile); sweep knobs `BM,BN,BK,depth`; record which primitives actually yield overlap | **A8**: latency table; written finding on whether NKI 0.6.0 overlaps rotated buffers |
| **M6** | Attention: multi-stage `compute_at` (scores → softmax → PV), Vector/Scalar-engine instrs (`tensor_reduce`, `exponential`, `activation`, `tensor_scalar`), `nc_transpose` for Pᵀ; nkibench level 8 | numerics + "intermediate never leaves chip" traffic |
| **M7** (stretch) | Online-softmax / flash rewrite as a *verified algebraic rewrite* (it is not a pure loop transform — it changes the algorithm via rescaling), z3 for bounds, auto-tuner over knobs, text round-trip of IR / optional MLIR export | |

### Design implications of attention (so M1–M3 don't paint us into a corner)
Multiple stages with producer–consumer placement (`compute_at` generalises beyond one reduction), reductions with ops other than `+` (max), mixed engines (Tensor/Vector/Scalar/GpSimd) so `Call` nodes carry an **engine tag**, PSUM→SBUF→PSUM round-trips (S=QKᵀ evacuated, softmaxed, then P re-fed as a matmul operand after transpose), and numerics-sensitive rewrites (max-subtraction) that must be flagged as *not* simple reorderings.

---

## 12. Risks and open questions

1. **Overlap may not be controllable** on NKI 0.6.0 (no sync API). Mitigation: M5 measures it; the language still delivers legality + tiling + staging.
2. **Unverified hardware numbers** (§3.2). Mitigation: single `HardwareConfig`, probe script in M0; do not hard-code.
3. **Rotating-buffer indexing form** that NKI accepts (`buf[i % d]` with dynamic index vs unrolled named buffers) — unknown until tried; emitter has two strategies.
4. **`replace` unification** is the hardest single component (affine window inference). Plan: start with *template matching on the specific loop shapes* of the 3–4 instrs (matching up to loop order with symbolic affine solving only for window offsets), upgrade to SMT/z3 only if needed.
5. **Interval analysis too coarse** for tails/modular indices ⇒ restrict v0 to divisible shapes + `cut`.
6. **Scope creep to a general compiler**. Guardrail: ops added only when a ladder rung or attention needs them.
7. **Open question for the user**: is `lhsT`-in-the-spec acceptable for v0 (my default), or should `A[M,K]` + automatic transpose be in the first milestone? Also: Python-source emission only, or also keep an option for `nki.baremetal`/`neuron-profile` integration inside the tool (my default: later, as a thin wrapper around what `nkibench` layers 2–3 will do).
