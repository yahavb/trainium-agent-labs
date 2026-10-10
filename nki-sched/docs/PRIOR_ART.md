# Prior art digest — what we take from each system

Sources actually read for this plan (not recalled from memory):

| Source | How it was read |
|---|---|
| Halide, PLDI'13 (Ragan-Kelley et al.) | full PDF → text; read §3 (schedule model) and §4 (lowering, bounds inference) |
| TVM, OSDI'18 (Chen et al., arXiv 1802.04799) | full PDF → text; read §4.2 (memory scopes), §4.3 (tensorization), §4.4 (latency hiding / virtual threads) |
| Exo, PLDI'22 (Ikarashi et al.) | full PDF (par.nsf.gov copy); read §2 (examples), §3 (language, rewrites, `replace`), §5 (effect analysis, skimmed) |
| Exo repo (`exo-lang/exo`, shallow clone) | `src/exo/API_scheduling.py` (primitive list/docstrings), `docs/Cursors.md`, `docs/Design.md`, `stdlib/halide_scheduling_ops.py` (header) |
| Exo 2, ASPLOS'25 (arXiv 2411.07211) | abstract only (cursors, "AIR": Actions / Inspection / References) |
| NKI | installed package on `seat-270` (nki `0.6.0+31049202112`): `dir(nki.isa)`, `dir(nki.language)`; the online `nki.isa.nc_matmul` doc page; this repo's `reference_level{3,4}.py` and `nkibench.py`. The rest of the online NKI docs site returned 404 during this session, so architecture numbers (SBUF/PSUM sizes) are *not* verified — see PLAN §3. |

Not read: MLIR Transform-dialect docs (background knowledge only — see PLAN §9, flagged there).

---

## Halide — separate *what* from *how*; two orthogonal schedule axes

* **Algorithm** is a pure functional definition (`Func`s over `Var`s; reductions over an `RDom`). **Schedule** never changes semantics.
* Schedule = **domain order** (per function: split / reorder / serial|parallel|vectorize|unroll over its own iteration domain) + **call schedule** (per function: *where in the consumer's loop nest it is stored and computed* — `store_at` / `compute_at`; root = breadth-first, innermost = fully fused, store-root + compute-inner = sliding window).
* Reduction dimensions may only be reordered/parallelized if the update is associative (§3). We need exactly this rule for K loops.
* **Safe by construction** because loop bounds and allocation sizes are *inferred* by interval analysis from the output size, never user-written (§4.2). Axis-aligned boxes only; chosen over polyhedral for generality of expressions.
* Lowering is a monolithic, ordered pipeline (§4). The compiler makes no heuristic decisions: "we defer to the schedule".

**Take:** algorithm/schedule split; `compute_at`/`store_at` for producer–consumer placement (needed for `acc → C` now and attention later); interval-based bounds inference for staging windows and capacity checks; the associativity rule for reductions.
**Leave:** monolithic lowering (hard to inspect; every primitive is entangled with the lowerer).

## TVM — hardware-specific scopes, tensorize, explicit latency hiding

* Tensor-expression language + schedule primitives (`split`, `tile`, `reorder`, `fuse`, `bind`, `compute_at`, `cache_read/cache_write`, `tensorize`, virtual threads).
* **Memory scopes** (§4.2): a buffer is tagged with a scope and lowering/sync rules depend on it → our `set_memory(buf, SBUF|PSUM|HBM)`.
* **Tensorize** (§4.3): the hardware intrinsic is *declared with the same tensor-expression language* (behaviour + lowering rule), decoupling schedule from specific instructions → our NKI instruction table is written as loop-nest semantics. (Exo does the same and also automates the matching.)
* **Latency hiding** (§4.4): decoupled access–execute hardware needs fine-grained dependence tokens (`push_dep_to` / `pop_dep_from`, RAW and WAR). TVM lets the user write a *virtual-thread-parallel* program; the compiler injects tokens and interleaves into one instruction stream (their FPGA accelerator: peak compute utilization 70% → 88%).
  **Key difference for us:** NKI 0.6.0 exposes no such tokens (`dir(nki.isa)` has `core_barrier` and nothing like semaphores/`push_dep`). NKI's compiler does dependency tracking. So our `pipeline` / `overlap` primitives are *program transformations that expose overlap* (multi-buffering + reordering + engine assignment), not sync insertion. We keep a pass slot for TVM-style token injection in case a lower-level API appears.
* Auto-tuning: AutoTVM = hand-written templates with knobs + learned cost model → our schedules are Python functions with parameters (`BM, BN, BK, depth`), i.e. templates for free.

**Take:** memory scopes; intrinsics declared in the IR's own language; schedule-as-template; the idea virtual-thread → pipeline.
**Leave:** one-shot lowering from a stage graph; implicit scope inference.

## Exo — rewrite-based, small trusted core, hardware as a library

* **Scheduling = successive equivalence-preserving rewrites of an imperative program** (`p = divide_loop(p, ...)`); each primitive is checked independently, so "the correctness of each operator is independent of the correctness of each other operator" (§3.3). Contrast: Halide/TVM schedules are arguments to a monolithic lowering.
* Primitive set (from `API_scheduling.py`): `divide_loop(tail=guard|cut|cut_and_guard, perfect=)`, `reorder_loops`, `fission`, `fuse`, `unroll_loop`, `cut_loop`, `shift_loop`, `join_loops`, `remove_loop`, `lift_scope`, `reorder_stmts`, `stage_mem(win, name, accum=)`, `expand_dim`, `lift_alloc`, `sink_alloc`, `set_memory`, `bind_expr`, `reuse_buffer`, `delete_buffer`, `replace(block, instr)`, `inline`, `call_eqv`, `specialize`, `simplify`, … `stage_mem(accum=True)` = "zero a staging buffer, accumulate into it, add back" — the PSUM-accumulator pattern.
* **`replace` = instruction selection by unification** (§3.4): a hardware instruction is an Exo procedure whose *body is its semantics*; `replace` infers the call arguments (including affine window expressions, via SMT) such that inlining the instruction recovers the original code. → our `replace(block, nisa.nc_matmul)`.
* **Hardware as library** (§3.2): `Memory` classes, `@instr` procs (semantics + C template), `@config` state. The compiler has no hardware knowledge.
* **Safety**: effect analysis over a denotational semantics Σ→Σ with a ternary logic ("definitely / maybe / not" read or written); commutativity of statements, static bounds checks, equivalence via SMT (§5). Configuration state makes full precision undecidable, hence ternary.
* **Cursors** (Exo 2, `docs/Cursors.md`): first-class references to code: `p.find("for k in _: _ #1")`, `p.find_loop("k")`, `p.find_alloc_or_arg("A")`; navigation (`.next() .prev() .parent() .body() .before() .after()`); `StmtCursor / GapCursor / BlockCursor`. Each primitive returns a **forwarding function** so older cursors can be re-resolved in the rewritten program (`p.forward(c)`; invalid if the node was destroyed). Exo 2's thesis: Actions + Inspection + References let users write new scheduling ops (e.g. a Halide-style `compute_at`) as *library code* (`stdlib/halide_scheduling_ops.py`).

**Take:** this is the backbone — rewrite-based IR, small trusted primitive core, `replace` with unification, memories/instructions as library data, cursors with forwarding, derived ops in library code.
**Adapt:** Exo targets C with scalar-loop semantics; we additionally need (a) partition-dim legality, (b) PSUM accumulate semantics, (c) engine/async annotations, (d) Python-source emission. We skip Exo's SMT-heavy effect analysis at first (affine/interval checks + differential testing) and keep the door open for z3.

## MLIR Transform dialect (background knowledge; not re-read this session)

`linalg` ops + `transform.structured.{tile_using_for, fuse, pad, promote, vectorize}`, `transform.bufferization.*`, `scf` loop pipelining. Schedules are IR (a transform script) operating on SSA *handles* to payload ops, with handle-invalidation rules. It is the closest industrial analogue of "a scheduling language over IR". Its handle model (consumed handles are invalidated) is weaker than Exo's forwarding for our purposes, and it has no NKI backend. See PLAN §9 for the build-vs-adopt decision.
