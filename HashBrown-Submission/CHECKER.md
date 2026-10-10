# CHECKER.md — what `nkibench.py` + `augment.py` accept, reject, and why

Deliverable for the Hack the Chip 2026 kernel-hardening project. This describes the checker
as it exists after Phases 1-4 of `HARDENING.md`, which has the full history, measurements and
reasoning behind every number here — this file is the condensed reference.

The checker runs three layers, in order, on every candidate kernel:

1. **Static rules** (`check_rules`) — milliseconds, no execution.
2. **Numerics** (`simulate_and_count` + `describe_mismatch` + `check_inputs_untouched`) — runs
   the kernel in `nki.simulate` against a NumPy reference.
3. **Traffic bar** (`check_traffic_bar`, levels 5-7 only) — counts HBM bytes moved.

A fourth, optional layer (**augmentation**, `augment.py`) replaces or extends the fixed test
shapes/values with synthesized ones, behind the `--augment` flag or an explicit `tiers=` set
passed to `nkibench.verify()`.

---

## 1. Static rules (`check_rules`, `nkibench.py:396-454`)

A source-level AST scan. Runs before anything is imported or executed, so a kernel that would
crash the simulator never gets the chance to.

| rule | accepts | rejects | why |
|---|---|---|---|
| **banned framework calls** | NKI's own primitives (`nl.sum`, `nisa.nc_matmul`, ...) — never flagged, regardless of name, because banning by bare name would also reject the CORRECT kernel | a call whose head is a known framework module (`np`, `torch`, `jnp`, `jax`, `F`, `nn`) or whose head equals its own leaf name, AND whose leaf is in that level's own `banned` set (e.g. `{"mean","average",...}` for level 1, `{"matmul","dot","einsum",...}` for levels 3/4/8) | the level is specifically about computing that operation IN the kernel; handing it to a framework function is the single most direct cheat, and it is per-level because `nl.sum` doing a real reduction is correct NKI, not a cheat |
| **`.T` on an entry-point argument** | `.T` on anything else (a local variable, an intermediate tile) | `some_arg.T` where `some_arg` is a parameter of the level's entry function | transposes the input on the HOST before the kernel ever runs — the kernel is supposed to do that work itself (relevant to level 2) |
| **the `@` operator** | nothing containing `@` as a binary matmul op (there is no legitimate use of it in NKI kernel code at any level) | any `ast.BinOp` with `ast.MatMult` | does the entire matmul in one Python expression on the host, bypassing `nisa.nc_matmul` entirely |
| **oversized literal partition dim** | `nl.ndarray((128, ...))` or anything whose first-dimension literal is ≤128, or any size that isn't a literal constant (can't be statically checked) | `nl.ndarray((N, ...))` where `N` is a literal integer `> 128` (`PMAX`) | a tile's partition axis has a hard hardware ceiling; this catches the OBVIOUS, literal violation only — see "known gaps" below for what it misses |
| **entry point exists** | a function matching the level's declared `entry` name | no function with that name anywhere in the file | that's the only name the harness looks for — there is no fallback |
| **entry point is `@nki.jit`-decorated** | any decorator whose unparsed text contains `"jit"` | the entry function existing but undecorated | an undecorated function runs as plain Python, not a compiled device kernel |

**Known gaps, logged honestly (Phase 1 findings):** this is narrower than the official
challenge's generic Stage-A rule list. It does NOT statically enforce: the 512-column
stationary/moving free-dim limit (only the 128-row partition limit has a literal check); that
the kernel contains an explicit loop over tiles at all (no banned call ≠ a tiled kernel); or a
general ban on fancy indexing / boolean-mask indexing / `np.einsum` / broadcasting tricks
(only banned where a level's own `banned` set happens to include that name). These were
judged not worth over-fitting a regex-style static scanner for; the numerics layer below
catches most of what slips through, because a kernel that cheats on tiling or indexing
produces a wrong answer, which the next layer is built to catch precisely.

---

## 2. Numerics (`describe_mismatch`, `nkibench.py:459-529`, plus `check_inputs_untouched`)

Runs the kernel in `nki.simulate` on real NumPy-generated inputs, compares against the
level's reference function.

### 2a. Tolerance — measured, not guessed

```python
TOL_BY_DTYPE = {"float32": 1e-4, "bfloat16": 3e-2}
```

**float32 (MEASURED).** Every honest reference kernel (levels 1-4, every declared shape
including level 3's second one) was run against its NumPy reference, and the worst observed
relative error — `max(|got - want|) / RMS(want)`, the same metric the checker itself uses —
was recorded:

| level | worst relative error across its shapes |
|---|---|
| 1 (avgpool) | 2.44e-7 |
| 2 (transpose) | 0.0 (pure data movement, no arithmetic) |
| 3 (matmul, single tile) | 0.0 |
| 4 (matmul, tiled) | 3.103e-6 (shape K=256 M=512 N=1024, the most accumulation steps) |

**Max over everything honest: 3.103e-6.** `1e-4` sits ~32x above that noise floor (room for
legitimate float32 roundoff on shapes we haven't tested yet) and ~10x BELOW the error a
`+1e-3` uniform offset introduces on this data (`cheats/c6_almost_right.py`; its measured
worst error at this tolerance was 0.00099). That 10x gap is deliberate: a tolerance set
exactly at the noise floor would be fragile; one set an order of magnitude below the known
cheat's error leaves margin while still catching it.

**bfloat16 (PROVISIONAL, NOT measured).** No current test shape uses bfloat16 — every level's
input generator casts to `float32` (see `eval_set.md`). `3e-2` is a placeholder, deliberately
LOOSER than float32 (bf16 has ~3 decimal digits of precision, machine epsilon ≈7.8e-3), never
tighter — rule 10. Replace this the day a bfloat16 test shape exists; it is not load-bearing
today because nothing is ever checked against it.

`describe_mismatch(got, want, tol=None)` resolves `tol` from `got`'s own dtype automatically
when the caller doesn't pass one explicitly (`tol_for_dtype`), so both `agent.grade()` and
`nkibench.verify()`/`--check` get the dtype-aware default without either call site needing to
know about `TOL_BY_DTYPE`.

### 2b. What's checked, in order, and the message each produces

| check | accepts | rejects | message |
|---|---|---|---|
| **shape** | output shape == reference shape | anything else | `WRONG SHAPE: returned X, reference is Y. Check the output-size arithmetic, not the values.` |
| **finiteness** | no NaN/Inf anywhere | any NaN or Inf | `NON-FINITE OUTPUT: N NaN and M Inf, first at (...)`. In practice this is almost always an unwritten SBUF/PSUM tile — `nki.simulate` fills NEW allocations with NaN (confirmed empirically for both `shared_hbm` outputs, cheat C1, and plain `sbuf` tiles, cheat C9's fallback branch), not zeros and not leftover memory |
| **almost-all-zero** | — | ≥90% of output is zero while the reference isn't | `OUTPUT IS X% ZEROS ... results never reached the output tensor` — named separately from a generic mismatch because it's a distinct, very common failure mode (copied out only the last tile, or never copied at all) and sending the model to "re-check your arithmetic" would be actively misleading |
| **partially-zero** | — | 25-90% zero in a contiguous block, reference has none there | names the exact index block, same reasoning |
| **tolerance** | `worst_relative_error <= tol` (dtype-aware, see 2a) | otherwise | `NUMERICAL MISMATCH: worst error E of the output's RMS (R), tolerance T` + the worst index, expected/got, % of elements outside tolerance, and — if the worst index falls in the final partial partition/free tile — an explicit note that the ragged edge is the likely cause, not the core arithmetic |

### 2c. Input protection (`check_inputs_untouched`, `nkibench.py:641-653`)

**Accepts:** a kernel that allocates a fresh `nl.ndarray(..., buffer=nl.shared_hbm)` and
writes its result there. **Rejects:** a kernel that writes its result back into (or otherwise
mutates) any `np.ndarray` argument it was given, even if the returned VALUES are numerically
correct. **Why:** the caller still owns that buffer in a real graph; a kernel that aliases its
input only "works" because this harness happens to compute the reference before calling the
kernel — STATE.md already recorded this exact hole once ("a kernel wrote into its input and
passed"). This check is run from BOTH `agent.grade()` and `nkibench.verify()`/`--check` as of
Phase 3 — it used to be `agent.grade()`-only, which is why `cheats/c5_input_tamper.py` is in
the cheat set: it's the direct regression test for that fix.

---

## 3. Traffic bar (`check_traffic_bar`, `nkibench.py:615-638`, levels 5-7 only)

Not a correctness check — levels 5-7 share level 4's reference and shapes exactly, so
correctness alone can't tell them apart. **Accepts:** HBM traffic ≤ `max_waste × floor` (floor
= read every input once, write the output once). **Rejects:** more than that, with a message
naming how far over and the standard fix (hoist loads / block M,N / block K too, per level).
Thresholds (1.6x, 1.25x, 1.05x) come from where the shipped tiled kernel (level 4's own
reference) measures on its largest shape (2.00x the floor) — each level must beat the one
below it. **Known gap:** this check is wired into `verify()` but was never ported into
`agent.grade()`'s own logic path for levels 5-7 specifically (levels 5-7 aren't exercised by
the agent today — `agent.py --all` only runs levels 1-4).

---

## 4. Augmentation (`augment.py`, opt-in via `--augment` or an explicit `tiers=` set)

Off by default — rule 7 — so the original fixed-shape, seed=0 checker stays exactly
reproducible. See `results/eval_set.md` for the full, literal list of every shape/value/seed
each tier actually uses.

| tier | accepts | rejects | why |
|---|---|---|---|
| **A1** fresh seed | the SAME declared shapes, any random instantiation of them | nothing by itself — it's a value change, not a new check | catches a kernel that memorized/overfit the exact seed=0 values. We could not build a working example of this (see Findings log, C10) but the tier is essentially free (same test count, just a different seed), so it stays on by default |
| **A2a** extra evenly-dividing shapes | one additional shape per level, still inside its declared class, still evenly divisible | a kernel that only handles the ORIGINAL declared shapes | catches "memorized the test set" cheats with more than one entry (`cheats/c9_memorise_all_shapes.py`) — measured: this is where C9 is first caught |
| **A2b** ragged shapes | a prime dimension, a dimension of exactly 1, still inside the declared class | a kernel that assumes a size divides some tile constant evenly | catches "assume divisible" cheats (`cheats/c11_assume_divisible.py`) — measured: this is where C11 is first caught, and NOT at A2a, confirming the two tiers catch genuinely different things. **Level 4 has no A2b shapes at all** — its own declared class requires exact tile multiples (asserted in `reference_level4.py` itself), so no ragged shape is even legal input there; this is a scope boundary, not a gap |
| **A3** hostile values: large magnitude (±1e4), large mean (1e4 + noise), zeros, negatives, **tiny (1e-6 to 1e-3)**, one identical row | friendly standard-normal values passing the SAME tolerance | the SAME shape, with values engineered to expose overflow, cancellation, or a fixed bias | **tiny, not large, is what catches a fixed absolute offset** (`cheats/c6_almost_right.py`): a `+1e-3` bias is a tiny relative error against RMS≈1e4 data and a huge one against RMS≈1e-4 data. Empirically confirmed at the OLD (pre-Phase-3) tolerance: of 64 checks, exactly the 8 "tiny" variants failed, zero "large_magnitude" ones did |
| **A4** output-sensitivity: run twice on different inputs, same shape | an output that changes when the input does | an output that's identical (incl. all-NaN, via `equal_nan=True`) across two different random inputs | catches "ignores its input" cheats (do-nothing, constant-output) by a mechanism that doesn't depend on what uninitialized memory happens to contain — the harness doesn't own the output buffer (every kernel allocates its own `shared_hbm` tensor), so pre-filling with NaN and checking for leftovers isn't an option; this is the only one of the two originally-proposed A4 designs that applies here |

### Recommended default tier set

Measured cost-effectiveness (`results/cost.csv` + `results/cheats_after.csv`, HARDENING.md
Findings log): **A2a and A2b are the two tiers with a demonstrated catch on the current cheat
set, and they cost almost nothing** (+0.007s and -0.003s/kernel respectively, within noise of
A0 itself). A1 is similarly cheap (same shapes, same test count). A3 is the single most
expensive tier (+1.2s/kernel — 6 hostile variants per shape) and A4 moderate (+0.3s/kernel);
neither catches anything NEW on the current cheat set (A3's original target, C6, is already
closed by the Phase 3 tolerance fix; A4's targets are already caught by `describe_mismatch`'s
NaN/zero heuristics).

**Recommendation: run A0 + A1 + A2a + A2b on every agent round** (cheap, with demonstrated or
principled value). **Reserve A3 + A4 for a periodic or pre-submission full audit** — A4 for
its mechanism-independent robustness (worth the cost when it matters), and A3 particularly
once the ladder grows a level with a genuine magnitude-dependent numerical trap (softmax,
layernorm — see `HARDENING.md`'s C12 discussion), where its cost becomes clearly justified
rather than the most expensive tier with nothing to show for it today.

---

## 5. What this checker still cannot do

- **No real device timing.** Everything here runs in `nki.simulate` on a CPU; the roofline
  arithmetic (bytes, flops, arithmetic intensity) is derived, not measured on hardware.
- **No bfloat16 ground truth.** The `3e-2` tolerance is a placeholder.
- **No static tiling/indexing-style enforcement** beyond the literal partition-dimension
  check — see section 1's "known gaps."
- **No augmentation for level 8** (attention) — `augment.extra_shapes()` returns `[]` for any
  level it doesn't have an explicit case for, including 8; it was never extended because no
  cheat was built against it.
- **No C10/C12-style cheats could be built** against the current four levels — see
  `HARDENING.md`'s Phase 4 Findings log for the concrete technical reasons (an NKI API
  limitation; a float32 dynamic-range argument). This checker has not been tested against a
  genuine precision-cliff cheat because one couldn't be constructed for these operations.
