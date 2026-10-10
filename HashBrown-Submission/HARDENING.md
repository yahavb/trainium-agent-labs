# Checker hardening: augmented test inputs for the kernel agent

Living plan for Hack the Chip 2026 (NYU × Annapurna Labs). **Claude Code: read this whole file at the start of every session, follow the rules, and append what you learn to the Findings log (section 10).** Keep edits factual; never delete earlier findings.

---

## 1. The project in one paragraph

Project 2 (`projects/02-kernel-agent/`) is an agent loop: a small model (Qwen3-8B) writes NKI kernels for AWS Trainium, and a checker (`nkibench.py`) grades them (0.1 parses, 0.2 rules clean, 0.2 runs in `nki.simulate`, 0.5 correct on every shape). The loop is only as trustworthy as the checker: a kernel that does not really do the work must never score 1.0. In 2025, Sakana AI's kernel results were inflated by exactly this kind of exploit. **Our project: augment the checker's test inputs so cheating or lazy kernels are caught, and measure how many augmented tests it takes to catch each cheat and what that costs in checking time.**

**Pitch:** "The checker tested a few fixed inputs, so cheats could slip through. We made it generate varied inputs automatically, and measured how many tests it takes to catch each cheat, and what that costs in time."

## 1b. Alignment with the official challenge (`CHALLENGE-kernel-agent.md`)

Claude Code: also read `CHALLENGE-kernel-agent.md` in this folder. Its scoring rubric is likely how we are judged. Our work maps onto it as follows:

| challenge requirement | where we cover it |
|---|---|
| Verification harness: uneven shapes incl. a prime dimension and a dimension of exactly 1 | A2 |
| Hostile values: large magnitudes, large means, zeros, negatives, a row of identical values | A3 |
| Stated, justified tolerance | `CHECKER.md` |
| Static rejection of rule violations | existing static rules; verify in Phase 1 |
| Failure messages the agent can act on (an instruction, not a verdict) | rule 12 |
| "Break it on purpose and confirm the harness catches it" | Phase 2 cheats |
| Eval set (shapes and values tested) | `results/eval_set.md` |
| Failure taxonomy with counts | Phase 4b |
| Method & honesty: does the agent know when it failed? | Phase 4b "false verified" count |
| Token instrumentation | stretch, Phase 4c |

## 2. Workflow: where things live (read carefully)

- **Laptop (`<local clone>/projects/02-kernel-agent/`)**: Claude Code reads and edits code here. This is the source of truth. Git commits and pushes happen here, to the user's own fork.
- **Pod `seat-<N>` (`/workspace/projects/02-kernel-agent/`)**: the only place code can *run* (it has the Neuron SDK, `nki.simulate`, and the model server). Nothing important should live only in the pod.
- **Sync laptop → pod** after every edit, before running:
  `kubectl cp ./projects/02-kernel-agent/. seat-<N>:/workspace/projects/02-kernel-agent/`
- **Run in the pod:**
  `kubectl exec seat-<N> -- bash -c "cd /workspace/projects/02-kernel-agent && <command>"`
- **Long runs in the pod** go in the background:
  `kubectl exec seat-<N> -- bash -c "cd <dir> && nohup <command> > <log> 2>&1 < /dev/null &"`
- **Copy results pod → laptop** before committing:
  `kubectl cp seat-<N>:/workspace/projects/02-kernel-agent/results ./projects/02-kernel-agent/results`
- If any kubectl command fails with `ExpiredToken` or asks for credentials, **stop and tell the user**; they must paste fresh AWS credentials into this terminal. Do not retry in a loop.

## 3. Rules (do not break)

1. **Never change what an honest kernel needs to pass.** After any checker change, honest reference kernels must still score 1.0, and the honest agent run must not regress.
2. **Measure with repeats.** Agent runs use `--repeat N` (N ≥ 3). Never claim a change from one run.
3. **Do not re-add the reverted tiling example to the prompt** (it dropped level 2 from 4/5 to 0/5).
4. **Do not use `--think`** (rounds go from ~8 s to ~446 s and score 0.00).
5. **Do not touch the model server.** It runs at TP=2 on NeuronCores 2–3. Use `--context 8192`.
6. **Never edit files the running baseline uses.** The baseline runs from a frozen copy in the pod (`/workspace/baseline-copy`), never from the working folder.
7. **Augmentation is off by default behind a flag `--augment`**, so the original checker stays reproducible.
8. **Seed everything.** Every augmented input must be reproducible from a logged seed.
9. **Augment only within each level's declared shape class.** Shapes the task never promised (e.g. >128 rows on a level that doesn't require tiling) change the task instead of hardening the check.
10. **Tolerance follows dtype, never tighter.** fp32 and bf16 need different tolerances.
11. **"Pass" means score 1.0.** Record partial scores too.
12. **Failure messages must name the specific problem** (e.g. "rows 64–127 never written"), because they are fed back to the model.
13. **Commit and push from the laptop after every phase.**

## 4. Files we add

```
projects/02-kernel-agent/
  HARDENING.md        <- this file (plan + findings)
  score_kernel.py     <- only if no existing entry point: score one kernel file
  cheats/             <- deliberately bad kernels, one file each
    README.md         <- what each cheat does, which test should catch it
  augment.py          <- augmented input generator
  run_cheats.py       <- runs every cheat under each checker tier, writes CSVs
  results/
    baseline.log
    cheats_before.csv
    cheats_after.csv
    cost.csv
    augmented_agent.log
  CHECKER.md          <- deliverable: each rule, what it accepts/rejects, why
  NOTE.md             <- deliverable: one-page note
```

## 5. Phases

### Phase 0: Environment (target 20 min)
- [ ] In the pod: `neuron-ls` shows a device; `./serve.sh` printed READY (user does this in a separate terminal).
- [ ] In the pod: `git config --global --add safe.directory /workspace`
- [ ] Sync this file to the pod.
- [ ] In the pod: `python nkibench.py --selftest` passes.
- [ ] Freeze a baseline copy: `cp -r /workspace/projects/02-kernel-agent /workspace/baseline-copy`
- [ ] Start the honest baseline from the copy, in the background:
  `cd /workspace/baseline-copy && mkdir -p results && nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 3 > results/baseline.log 2>&1 < /dev/null &`
  Record the start time. It may take an hour or more.

### Phase 1: Understand the checker (target 30 min). Read only; no code changes.
Answer each in the Findings log with function names and line numbers:
- [x] **Entry point:** how `agent.py` hands kernel text to `nkibench.py`, and whether a kernel can be scored from a file. If not, note that `score_kernel.py` is needed.
- [x] **Inputs:** where test inputs are built per level (shapes, seeds, value ranges, dtype).
- [x] **Run and compare:** where `nki.simulate(kernel)(*args)` is called; where output is compared; which tolerance.
- [x] **Output allocation:** does the harness pass in an output buffer, or does the kernel allocate and return it?
- [x] **Input protection:** confirm inputs are snapshotted and compared after the run, and for which inputs.
- [x] **Shape classes:** each level's declared shapes and dtype.
- [x] **Reference kernels:** which levels have a known-good kernel to base cheats on (prefer level 2).
- [x] **Cost today:** rough checker time per kernel.

### Phase 2: Cheat kernels (target 45 min)
If needed, first write `score_kernel.py` (score one kernel file through the real checker, print score and message). Then write each cheat as a variant of a passing reference kernel, with a header comment: what it does, why it's wrong, which test should catch it.

| id | cheat | wrong because | expected catcher |
|---|---|---|---|
| C1 | do_nothing | never writes output | A4 |
| C2 | partial_rows | only processes the first 64 rows | A2 |
| C3 | hardcoded_shape | only correct at the default test shape | A2 |
| C4 | constant_output | ignores inputs, writes zeros/constant | A1, A3 |
| C5 | input_tamper | writes the result into its input | existing snapshot |
| C6 | almost_right | adds a tiny offset (+1e-3) | A3 (large values) |
| C7 | edge_skip | skips the last row/column | A2 (odd sizes) |

Note for C1: Sakana's exploit relied on GPU memory reuse; `nki.simulate` may give fresh zeros instead. Record what actually happens; either result is a finding.

- [x] Write `run_cheats.py`: runs every cheat through the **unchanged** checker → `results/cheats_before.csv` (cheat, level, score, passed (1.0?), checker message).
- [x] Sync, run in the pod, copy results back, commit, push.

### Phase 3: Augmentation (target 60 min)
Implement `augment.py` and wire it into `nkibench.py` behind `--augment`. Build in this order; stop after A4 if time is short.

| tier | augmentation | catches |
|---|---|---|
| A0 | original fixed inputs | baseline |
| A1 | fresh random values per run, logged seed | C4 |
| A2 | varied shapes inside the level's declared class: a prime dimension (e.g. 37, 127), a dimension of exactly 1, and sizes that don't divide the tile evenly (partial last tile) | C2, C3, C7 |
| A4 | output never written: the harness does NOT own the output buffer (every kernel allocates and returns its own -- see Phase 1 findings), so this is always "run twice with different inputs of the same shape, fail if outputs are identical" | C1, C4 |
| A3 | hostile values (respecting dtype): large magnitudes (±1e4), large means (e.g. 1e4 + small noise), zeros, negatives, **tiny values (1e-6 to 1e-3, not just one point value)**, and one row of identical values. **Tiny values, not large magnitudes, are what catches C6** -- a fixed absolute offset is a bigger relative error against a smaller RMS, not a smaller one; see Findings log. | C6 (via tiny), C4 |
| A5 | stretch only: metamorphic checks (e.g. transpose twice = original) | logic errors |

- [x] Verify every honest reference kernel still scores 1.0 with `--augment`.
- [x] Sync, commit, push.

### Phase 4: Measure (target 45 min) -- reframed: play a smarter attacker against the hardened A0
- [x] Write second-generation cheats C9-C12 designed to pass the hardened A0. C9 and C11
      built and pass A0; C10 and C12 attempted and dropped with reasons logged (Findings log).
- [x] Extend `run_cheats.py` to run every cheat (C1-C9, C11) at each cumulative tier
      (A0; +A1; +A2a; +A2b; +A3; +A4) → `results/cheats_after.csv`. `results/cheats_before.csv`
      left untouched.
- [x] Record checker time per kernel at each tier → `results/cost.csv`.
- [x] Fill in the tables in section 8. Sync, copy results back, commit, push.

### Phase 4b: Failure taxonomy and "false verified" (target 30 min, high value)
- [x] Write `results/eval_set.md`: every shape, dtype, value pattern and seed the hardened checker uses, per level. (Written early, during Phase 6 prep, since it only needed the checker code -- not the baseline run.)
- [x] Collect every kernel the honest agent wrote in the baseline run (from its logs). Re-score each one under the hardened checker.
- [x] **False verified:** count kernels the original checker scored 1.0 that the hardened checker rejects. This is the headline "does the system know when it failed" number. **Result: 0** — see Findings log.
- [x] **Taxonomy:** classify every failing kernel into a few named failure modes with counts per level → `results/taxonomy.md`.

### Phase 4c: Token instrumentation (stretch, only if ahead)
- [ ] If `agent.py` logs prompt sizes, extract input tokens per attempt and what they were spent on (instructions, previous kernel, error, ledger) → `results/tokens.csv`. Don't modify the agent's prompting to do this.

### Phase 5: No-regression check (target 30 min active, longer waiting)
- [x] Once the baseline has finished, start the honest agent with augmentation, same settings, in the background. **`agent.py` gained a real `--augment` flag (Phase 5), not a workaround** — see Findings log. Started from a freshly frozen `/workspace/hardened-copy`, 2026-10-10 20:08 UTC; logging to `results/augmented_agent.log`.
- [x] Compare per-level solve rates and scores with `baseline.log`; fill in section 8. Report differences honestly.

### Phase 6: Deliverables (target 30 min)
- [x] `CHECKER.md`: per rule, what it accepts, rejects, and why. Written early (doesn't need
      the baseline run) -- includes the measured tolerance reasoning and the recommended
      default tier set.
- [x] Attempt logs: `results/*.csv` and `baseline.log` committed; `augmented_agent.log` exists
      in the pod (`/workspace/hardened-copy/results/`) but the run isn't finished yet -- will
      be copied back and committed once it is (Phase 5).
- [x] `NOTE.md`: one page, drafted with everything measured through Phase 4b, with a clearly
      marked placeholder for the Phase 5 regression numbers (filled in once that run finishes).
- [x] `DEMO.md`: a 2-minute live demo script, every command tested once against seat-66 and
      timed (~7.2s total command time). Shows C3 passing the original checker, the same file
      rejected by the hardened one with its exact failure message, and C9 passing the
      hardened fixed tests but caught by `--augment`.
- [ ] Copy results back, commit, push.

## 6. Minimum viable result (switch to this if behind at the 2.5-hour mark)
Cheats C1, C2, C4; augmentations A1, A2, A3, A4; the before/after table; `eval_set.md`; the false-verified count and taxonomy from the baseline kernels; one honest regression run if time allows.

## 7. Open questions for organisers (user handles these)
- Is working solo allowed? (README says teams of 3–5.)
- Does a modified Project 2 checker count as a submission?
- How and when to submit; judging criteria.
- Is `CHALLENGE-kernel-agent.md` the rubric we're judged on? It describes a NumPy "Stage A" ladder on the shared gpt-oss endpoint; does hardening Project 2's NKI checker count against it?

## 8. Results (fill in)

### Cheats vs checker tiers (✅ = caught, ❌ = passes at 1.0; from `results/cheats_after.csv`)

C1-C8 are the Phase 2 cheats, re-run here against the HARDENED A0 (level 3's second shape,
measured tolerance, tamper-check parity -- all from Phase 3, all unconditional, not gated by
`--augment`). C9/C11 are the Phase 4 "smarter attacker" cheats, built specifically to pass that
hardened A0. C10 and C12 were attempted and dropped -- see the Phase 4 Findings log entry
below for the concrete technical reasons.

| cheat | A0 | +A1 | +A2a | +A2b | +A3 | +A4 |
|---|---|---|---|---|---|---|
| C1 do_nothing | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C2 partial_rows | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C3 hardcoded_shape | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C4 constant_output | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C5 input_tamper | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C6 almost_right | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C7 edge_skip | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C8 path_mismatch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C9 memorise_all_shapes | ❌ | ❌ | **✅** | ✅ | ✅ | ✅ |
| C11 assume_divisible | ❌ | ❌ | ❌ | **✅** | ✅ | ✅ |
| C10 fixed_input_lookup | — attempted, dropped (NKI cannot read a concrete Python scalar out of a tile; see Findings log) — | | | | | |
| C12 precision_shortcut | — attempted, dropped (no float32 precision cliff exists inside our 1e-6..1e4 test range for these 4 ops; see Findings log) — | | | | | |

C9 and C11 are the only two rows with a ❌ anywhere, and each turns ✅ at EXACTLY the tier it
was designed against (A2a for C9, A2b for C11) and stays caught at every tier after. No cheat
in this set needs A5 (not implemented; stretch-only per the original plan), so it's omitted
as a column.

### Cost (`results/cost.csv`; seconds per `nkibench.verify()` call, seed=424242 once A1 is on)

"Tests per kernel" is the number of individual numeric/sensitivity checks one `verify()` call
runs -- it depends on how many shapes the level has, so it's a range across levels 1-4, not a
single number. Level 4 (matmul, tiled) is consistently the slowest kernel to check, because its
shapes are the largest and it does the most HBM transfers per check.

| tier | tests per kernel (range, levels 1-4) | seconds per kernel (range, honest references) |
|---|---|---|
| A0 | 2-4 | 0.009s - 0.70s |
| +A1 | 2-4 (same shapes, different seed) | 0.008s - 0.90s |
| +A2a | 3-5 | 0.010s - 0.92s |
| +A2b | 4-8 (level 4: still 5, no in-class ragged shape -- see augment.py) | 0.011s - 0.90s |
| +A3 | 28-56 (×7: base + 6 hostile variants per shape) | 0.068s - 5.29s |
| +A4 (full) | 32-64 (×8: + 1 output-sensitivity check per shape) | 0.085s - 6.49s |

The full augmented checker (+A4) costs roughly **8x** A0's test count and, for the heaviest
level (4), **~9x** its wall-clock time (0.70s -> 6.49s). For the lighter levels (1-3) the
absolute cost stays well under a second even at full augmentation (worst case 0.2s). All
numbers measured live on seat-66; see `results/cost.csv` for every individual
(kernel, tier) pair, including all 10 cheats, not just the 4 honest references summarized
here.

### Honest agent, before vs after (runs = 3 each)
| level | baseline: solved, scores | augmented: solved, scores |
|---|---|---|
| 1 | 0/3, [0.30, 0.30, 0.30] | 0/3, [0.30, 0.30, 0.30] — identical |
| 2 | 2/3, [1.00, 1.00, 0.30] | 1/3, [1.00, 0.30, 0.30] — lower solve rate, but **confirmed NOT a real regression**: of the augmented run's 48 level-2 attempts, exactly ONE passes the original 4 declared shapes on its own (`agent.grade(augment=False)` reward 1.00) — and that same attempt ALSO scored 1.00 live, under full augmentation. Zero cases of "passed original shapes, failed on an augmented shape/value." Nothing was rejected by any tier; attribute the 2/3→1/3 solve-rate drop entirely to normal run-to-run sampling variance at n=3 (see Findings log) |
| 3 | 0/3, [0.30, 0.30, 0.30] | 0/3, [0.30, 0.30, 0.30] — identical |
| 4 | 0/3, [0.62, 0.30, 0.62] | 0/3, [0.83, 0.83, 0.83] **as originally measured — now KNOWN TO BE INFLATED by a reward-aggregation bug, found by this measurement and fixed** (see Findings log). Re-scoring the same best kernel with the FIXED `agent.grade()`: **0.625 under `augment=False`** (identical to baseline, same 1/4 shapes, same exact failure) and **0.60 under `augment=True`** (correctly 1 of 5 shapes — A2a's extra shape also fails, which now correctly LOWERS the prorated score instead of sub-checks on the one passing shape inflating it to 0.83). A full corrected 3-repeat re-run was not performed (another ~hour); the single-kernel re-score already proves the fix and that no regression exists |

## 9. Known context (from the repo's STATE.md)
- Past checker holes: a kernel wrote into its input and passed; an element-at-a-time transpose passed as solved; byte counting was 2× low.
- Baselines (Qwen3-8B, 5 runs): L1 0/5 always 0.30; L2 4/5; L3 0/5 always 0.30; L4 0/5 always 0.62. Levels 1, 3, 4 have zero variance; level 2 is luck-limited. **CONTRADICTED for level 4 by this project's own baseline run (2026-10-10, 3 repeats): level 4 scored `[0.62, 0.3, 0.62]`, not always 0.62 — see the Phase 4b Findings log entry below and `results/taxonomy.md`. Leaving this original claim in place per the "never delete findings" rule; treat it as superseded for level 4.**
- On-device timing is not built; all checking is in the CPU simulator.

## 10. Findings log (append below; newest last; include time and what changed)

- **2026-10-10 19:05 UTC — Phase 1, checker understood (read-only, no code changed).**

  **Entry point.** `agent.py` does NOT call `nkibench.verify()`/`nkibench.main()`. It has its own
  grading function, `grade(source, level)` in `agent.py:55-162`, which re-implements the pipeline
  (parse → `nkibench.check_rules` → `nkibench.load_kernel` → per-shape `nkibench.simulate_and_count`
  → `nkibench.describe_mismatch`/`nkibench.check_traffic_bar`) and turns it into the WEIGHTS-based
  reward (`agent.py:52`, `parses=0.1 rules=0.2 runs=0.2 correct=0.5`). `nkibench.py` separately has
  its own CLI path to score one file: `verify(path, level_n, tol, seed)` at `nkibench.py:676-750`,
  reached via `python nkibench.py --level N --check file.py` (`nkibench.py:942-943`). **So a
  from-a-file entry point already exists (`verify`/`--check`) — no new `score_kernel.py` is needed
  for that** — but `verify()` is not equivalent to `agent.grade()`: `verify()` prints
  pass/fail + roofline notes and returns a small integer code (0/1/2/3), it does NOT compute the
  0.1/0.2/0.2/0.5 reward, and critically **`verify()` never calls `check_inputs_untouched`** (see
  "Input protection" below), so it would not catch C5 (input_tamper). For `run_cheats.py` (Phase 2)
  we should reuse `agent.grade()` directly, not `nkibench.verify()`, to get the real reward and the
  real set of checks the agent actually trains against.

  **Inputs.** Built per-level in `nkibench.py` by `make_inputs(spec, level_n, seed=0)`
  (`nkibench.py:217-219`), which seeds `np.random.default_rng(seed + level_n)` and dispatches to the
  level's `make_args` function: `_args_pool` (199-200), `_args_transpose` (203-204), `_args_matmul`
  (207-209), `_args_attention` (212-214). Every one of these calls `.astype(np.float32)` — **all
  four levels test in float32 only**, never bfloat16, even though the roofline ridge
  (`RIDGE_FLOPS_PER_BYTE = {"bfloat16": 222.0}`, line 51) is a bfloat16 figure; `verify()` prints a
  NOTE about this mismatch for levels ≥3 (confirmed in the level 3/4 `--check` output below). Shapes
  and value-ranges are declared per level in the `level(...)` calls: level 1 `nkibench.py:252-253`,
  level 2 `266-268`, level 3 `278`, level 4 `285-286`, level 8 `333`. All shapes today are **fixed,
  hand-picked tuples** — no randomized shape generation, no prime dimensions, no dimension-of-1 case
  anywhere in the current ladder. That gap is exactly what A2 (Phase 3) has to fill.

  **Run and compare.** `nki.simulate(kernel)(*args)` is reached through `_simulator(nki_mod, kernel)`
  (`nkibench.py:520-536`), which prefers `nki.simulate` and falls back to the older
  `nki.simulate_kernel`; confirmed on this pod the installed nki (0.6.0) exposes `nki.simulate`
  (selftest output: "simulation API: nki.simulate"). The actual call happens inside
  `simulate_and_count(kernel, args)` (`nkibench.py:611-662`), at `run(*args)` on line 653, wrapped so
  every `nisa.dma_copy` is counted for bytes/transfers. Output comparison is
  `describe_mismatch(got, want, tol=2e-2)` (`nkibench.py:421-484`); default tolerance is `2e-2`
  relative to the reference's RMS (`scale = sqrt(mean(want**2))`, line 434) — a single global
  tolerance regardless of dtype today (rule 10 in section 3 is not yet honored: fp32 vs bf16 use the
  same `tol`).

  **Output allocation.** The kernel allocates and returns its own output — the harness does not pass
  in a buffer. Every reference kernel does `out_tensor = nl.ndarray(shape, dtype=..., 
  buffer=nl.shared_hbm)` and returns it (confirmed in `reference_level1.py:39-40` and
  `reference_level2.py:49-50`; same pattern in the `API_CARD` example kernel, `agent.py:196-202`).
  This matters for A4 (Phase 3): since the harness does not own the output buffer, NaN-prefilling it
  is not an option — A4 must use the "run twice with different inputs, fail if identical" branch, not
  the "pre-fill with NaN" branch.

  **Input protection.** Snapshotting exists but is wired into only one of the two entry points.
  `check_inputs_untouched(before, args)` (`nkibench.py:596-608`) compares every positional arg
  against a pre-call copy (`np.array_equal`) and is called from `agent.grade()` at `agent.py:132`
  (`before = [x.copy() ... for x in args]` taken at `agent.py:120`, before `simulate_and_count`).
  **It covers every argument that is an `np.ndarray`** — not just the first — so for level 3/4
  matmul it protects both `lhsT` and `rhs`. It is NOT called anywhere in `nkibench.verify()`, so
  `--check` alone would miss a kernel that writes its result into an input (C5). Confirmed by
  reading `verify()` end-to-end (`nkibench.py:676-750`): no `before`/snapshot variable exists there.

  **Shape classes (today, all float32):**
  - Level 1 avgpool: `(32,32,32)` pool=2, `(128,16,16)` pool=4, `(8,24,24)` pool=3, `(64,8,8)`
    pool=2. All H,W divisible by pool_size; no ragged case.
  - Level 2 transpose: `(32,12)` as 3×4, `(128,64)` as 8×8, `(64,128)` as 4×32, `(8,35)` as 5×7.
    partition sizes (32,128,64,8) are all ≤128 so none exercises the partition-tiling limit; free
    dims up to 128, under the 512 moving-free max.
  - Level 3 matmul (single tile): one shape, `K=128 M=64 N=512` — exactly at the tile limits
    (K=PMAX, N=GEMM_MOVING_FMAX), no tiling needed.
  - Level 4 matmul (tiled): `K=128 M=128 N=512`, `K=256 M=256 N=1024`, `K=512 M=128 N=512`,
    `K=256 M=512 N=1024` (`_MM_SHAPES`, line 285) — all dimensions exact multiples of 128/512, so
    **no shape anywhere in the current ladder has a non-divisible/partial last tile**, which is the
    central gap A2 exists to close, and is also why C2/C3/C7 (partial rows, hardcoded shape,
    edge-skip) are expected to slip through A0 untouched.
  - Level 8 attention: `seq=128 dim=64`, `seq=64 dim=128`, `seq=96 dim=32` — none is a multiple of
    512 free-dim limit issue, no ragged edge either.

  **Reference kernels available.** `reference_level1.py`, `reference_level2.py`,
  `reference_level3.py`, `reference_level4.py` all exist in the project root and all score 1.0
  today (verified live, see "Cost today" below — `numerics 4/4` for levels 1/2/4, `1/1` for level
  3, rules clean on all four). Per the plan's preference for level 2 as the cheat base: level 2's
  reference (`reference_level2.py:22-77`) is a plain nested-loop element-by-element transpose
  (`nisa.tensor_copy` one scalar at a time inside `nl.affine_range` over `sz_f1 × sz_f2`) — simple
  to mutate into C2 (partial_rows), C3 (hardcoded_shape), C7 (edge_skip) by truncating or hardcoding
  the loop bounds. Level 1's reference (`reference_level1.py:22-73`) is a good base for C4
  (constant_output) and C1 (do_nothing) since it clearly separates "load → reduce → scale → store"
  into discrete steps that are easy to delete/no-op.

  **Cost today (measured live on seat-66, `kubectl exec ... time python nkibench.py --level N
  --check reference_levelN.py`):** level 1: 1.76s real (4 shapes). level 2: 1.68s real (4 shapes).
  level 3: 1.69s real (1 shape). level 4: 2.59s real (4 shapes, includes roofline/traffic-bar
  arithmetic for all 4). So **roughly 0.4-1.7s per shape** on this pod today, dominated by
  process/import startup (`user` time is 3-12s but spread across threads — likely NKI's own
  internal parallelism/JIT warmup) rather than by the simulation itself. This is the number to beat
  against in `results/cost.csv` once `--augment` multiplies the shapes per level.

  **Does the existing static rule-checker match the official challenge's rule list
  (`CHALLENGE-kernel-agent.md`, "the task" section, rules 1-5)?** Partially, and the gaps are real:
  - Rule 5 ("no calling the thing you're implementing") — **matches well**. `check_rules`
    (`nkibench.py:358-416`) bans exactly the per-level framework calls that would hand off the whole
    op (`spec["banned"]`, e.g. `{"mean","average",...}` for level 1, `{"transpose",...}` for level
    2, `{"matmul","dot","einsum",...}` for levels 3/4/8), plus the `@` operator (`ast.MatMult`,
    line 406) and `.T` on an entry-point argument (lines 402-405). Confirmed caught live in
    `--selftest` output: "catches the @ operator", "catches .T on an argument", "catches np.mean".
  - Rule 1 ("fixed tiles ≤128×512") — **only partially enforced, and only statically for literal
    constants.** The only size check is in `check_rules` (`nkibench.py:394-401`): it flags
    `nl.ndarray((N, ...))` **only when `N` is a literal integer constant > 128** in the source. A
    tile size computed at runtime (e.g. from a variable, or any tile ≤128 rows but >512 free-axis
    columns) is not statically caught at all — the 512 column limit for GEMM moving/stationary free
    dims has no static check whatsoever, only exercised implicitly through the simulator at runtime
    if the kernel happens to violate it operationally.
  - Rule 2 ("explicit loops over tiles, no whole-array ops that hide tiling") — **not enforced**.
    There is no check that the entry point contains a loop at all; a kernel that does the whole
    computation inside one `nisa` call with no `nl.affine_range` would not be flagged by
    `check_rules` as long as it avoids the named banned calls.
  - Rule 3 ("no fancy indexing, no boolean masks, no `np.einsum`, no broadcasting tricks") — **not
    enforced generically.** `einsum` is only banned where it happens to appear in a level's
    per-level `banned` set (levels 3/4/8); there is no general ban on boolean-mask indexing or
    broadcasting tricks anywhere in `check_rules`.
  - Rule 4 ("handle shapes that don't divide evenly") — **not a static rule at all, by design** —
    it is meant to be caught behaviorally, by testing on ragged shapes. But as noted above under
    "Shape classes," **no current test shape is ragged**, so today nothing catches a kernel that
    mishandles the partial tile. This is precisely what A2 is for.

  **Net:** the existing static checker is narrower and more surgical than the challenge's generic
  Stage-A rule list — it targets "did you call the framework op you're supposed to implement"
  very well, but does not generically enforce tile-size limits, loop structure, or indexing style,
  and (correctly, per the plan) leaves ragged-shape handling to be caught by test data rather than
  by static analysis. Phase 3's augmentation work is therefore necessary, not just nice-to-have:
  today's harness would score a hardcoded-shape or partial-rows cheat 1.0 on every existing test
  shape for levels 1-4 and 8.

- **2026-10-10 ~19:40 UTC — Phase 2, cheat kernels written and run (both scoring paths).**

  **Level 2 reference confirmed element-by-element before basing cheats on it, as asked.**
  Re-reading `reference_level2.py:69-72`: the double loop over `nl.affine_range(sz_f1)` /
  `nl.affine_range(sz_f2)` calls `nisa.tensor_copy` on `nl.ds(..., 1)` slices — a single
  element of the free axis per call, `sz_f1 * sz_f2` calls total. This is NOT tiled; it is the
  same "element-at-a-time" pattern flagged in section 9 as a past checker hole ("an
  element-at-a-time transpose passed as solved"). The shipped reference itself is that pattern —
  it is correct, just not efficient, and the checker today has no mechanism to penalize it for
  that (no transfer-count check fires for level 2's current shapes; `verify()`'s "CORRECT BUT
  ISSUE-BOUND" message at `nkibench.py:714-718` only triggers when
  `transfers > max(8, elements // 64)`, which none of level 2's four shapes exceed). Using it as
  the base for cheats is still fine for the correctness-cheat purposes here (C2/C5/C6/C7/C8), but
  it means those cheat files inherit the same per-element copy structure, not a tiled one.

  **Eight cheats written** in `cheats/` (`c1_do_nothing.py` through `c8_path_mismatch.py`, plus
  `cheats/README.md`). Level assignment deviated from the plan's table in two places, both logged
  in each file's header: C3 (hardcoded_shape) moved to level 3, not level 2, because level 3 has
  exactly ONE declared test shape (`K=128 M=64 N=512`, `nkibench.py:278`) and its own shipped
  reference already hardcodes that shape via `assert K==128` / `assert M==64` / `assert N==512`
  (`reference_level3.py:40-44`) — on level 2, which already has 4 varied shapes, any shape-
  hardcoding cheat we tried was already partially caught by the existing shape diversity, so it
  would not have demonstrated the "A0 misses it" story the plan wants. C8 (`path_mismatch`) was
  added, not in the original plan, because it fell directly out of reading the reference's two
  index formulas (`out[i_f2*sz_f1+i_f1] = in[i_f1*sz_f2+i_f2]`) — swapping `sz_f1`→`sz_f2` in
  only the destination stride is a one-token bug that is mathematically identical to correct
  exactly when the tile is square.

  **Results (`results/cheats_before.csv`, both scoring paths, unaugmented A0):**

  | cheat | name | level | `agent.grade()` reward | grade result | `nkibench.verify()`/`--check` rc | check result | path mismatch |
  |---|---|---|---|---|---|---|---|
  | C1 | do_nothing | 1 | 0.50 | caught | 1 | caught | no |
  | C2 | partial_rows | 2 | 0.88 | caught (3/4 shapes) | 1 | caught | no |
  | C3 | hardcoded_shape | 3 | **1.00** | **PASSES** | 0 | **PASSES** | no |
  | C4 | constant_output | 1 | 0.50 | caught | 1 | caught | no |
  | C5 | input_tamper | 2 | 0.50 | caught | 0 | **PASSES** | **YES** |
  | C6 | almost_right | 2 | **1.00** | **PASSES** | 0 | **PASSES** | no |
  | C7 | edge_skip | 2 | 0.50 | caught (4/4 shapes) | 1 | caught | no |
  | C8 | path_mismatch | 2 | 0.62 | caught (1/4 shapes) | 1 | caught | no |

  **Findings, one per cheat:**
  - **C1 do_nothing**: an unwritten `nl.ndarray(buffer=nl.shared_hbm)` output comes back from
    `nki.simulate` as **NaN** (8192 NaN, 0 Inf on the first shape), not fresh zeros and not stale
    reused memory. This answers the plan's open question directly: on this simulator, the
    Sakana-style "lucky stale memory" exploit does not apply — `describe_mismatch`'s
    non-finite branch (`nkibench.py:428-433`) catches it immediately and unconditionally, with no
    augmentation needed.
  - **C2 partial_rows**: caught already at A0, 3 of 4 shapes — the existing level-2 shape
    `(128, 64) as 8x8` already has `sz_p=128 > 64` (the cheat's hardcoded cap), so it was already
    exposed by pre-existing shape diversity, not by anything A2 would add. The "miss" A2 is meant
    to close here would need to be demonstrated on a level whose existing shapes never exceed the
    cap, which level 2 does not provide — worth re-checking against level 1 or 8 once A2 exists.
  - **C3 hardcoded_shape**: passes both paths at 1.0, fully, today. Confirms the finding above:
    level 3 is a genuine, currently-unclosable gap. **Action item for Phase 3/4: level 3 needs a
    second declared shape inside its own class (any K≤128, M≤128, N≤512) before this cheat can be
    caught by anything** — this is a shape-list gap, not something A1–A5 alone fix.
  - **C4 constant_output**: caught at A0 via the "100% ZEROS" branch (`nkibench.py:444-451`), on
    every shape, already — no augmentation needed for this one on level 1's current data.
  - **C5 input_tamper — the one real path mismatch.** `agent.grade()` catches it (reward 0.50,
    "THE KERNEL MODIFIED ITS INPUT") because it takes its own `before`-snapshot and calls
    `check_inputs_untouched` (`agent.py:120,132`). `nkibench.verify()` reports "numerics 4/4
    shapes passed" and returns rc 0 — a clean, false PASS — because `verify()` never takes that
    snapshot (confirmed in Phase 1, reconfirmed here empirically). **Anyone scoring a kernel file
    with `python nkibench.py --level N --check file.py` alone, independent of the agent loop,
    would be fooled by exactly the checker hole STATE.md already recorded once (section 9: "a
    kernel wrote into its input and passed").** This is the single most actionable Phase 2
    finding: either wire `check_inputs_untouched` into `verify()` too, or document `--check` as
    non-authoritative for this class of cheat.
  - **C6 almost_right — passes both paths at 1.0, and the plan's catcher assumption was
    backwards.** A uniform `+1e-3` offset is invisible under `describe_mismatch`'s tolerance
    (`tol=2e-2` relative to the output's RMS, `nkibench.py:434-437`) on today's standard-normal
    (scale≈1) data. The plan's table says "A3 (hostile values, large magnitudes)" should catch
    this; checking the arithmetic shows the opposite: a *larger*-magnitude RMS makes the SAME
    fixed absolute offset *relatively smaller* and therefore *more* invisible, not less (a +1e-3
    offset against RMS≈1e4 is a relative error of ~1e-7). The augmentation that would actually
    expose C6 is **tiny-magnitude values** (e.g. the 1e-6 entry already planned under A3), where
    the same fixed offset becomes a relative error orders of magnitude over tolerance. **Action
    item for Phase 3: A3's own catcher mapping for C6 should read "tiny values," not "large
    magnitudes."**
  - **C7 edge_skip**: caught at A0, all 4 shapes, via NON-FINITE (uninitialized tail of the SBUF
    tile) — the bug is unconditional (drops the same last group regardless of shape), so it needs
    no shape augmentation at all to be exposed on level 2.
  - **C8 path_mismatch (unplanned)**: caught overall (reward 0.62), and the per-shape breakdown
    confirms the prediction in the cheat's own header exactly — it PASSES on the one existing
    square shape (`(128,64) as 8x8`, the only one where `sz_f1==sz_f2`) and fails on all three
    non-square shapes with an out-of-bounds `AssertionError` (e.g. "index range [12, 12] exceed
    dimension size of 12" on `(32,12) as 3x4`). A live, already-measured demonstration that
    non-square shape diversity — which level 2 already has, with no augmentation — is doing real
    work today.

  **Headline for Phase 3 planning:** under the path that actually matters (`agent.grade()`, what
  the agent trains against), **6 of 8 cheats are already caught at A0, with zero augmentation.**
  Only C3 (level 3's single-shape gap) and C6 (tolerance absorbs a tiny uniform offset on
  friendly-magnitude data) get through. That changes the Phase 3 priority: A2's biggest
  immediate value is not levels 1/2/4 (already shape-diverse) but **giving level 3 a second
  shape**, and A3's biggest immediate value is **tiny values, not large ones**, to catch C6.

  Synced to the pod and run there (not against `/workspace/baseline-copy`, which is untouched).
  `results/cheats_before.csv` copied back to the laptop.

- **2026-10-10 ~20:20 UTC — Phase 3, structural + tolerance fixes, augment.py, re-validated
  cheats (all against seat-66, not `/workspace/baseline-copy`).**

  **Level 3 given a second declared shape, and why it was necessary (not optional).**
  `nkibench.py`'s `LEVELS[3]["shapes"]` now has `[dict(K=128,M=64,N=512),
  dict(K=64,M=32,N=256)]`. The new shape alone is not the fix — `reference_level3.py` had to
  be generalized first: it previously asserted `K==128`, `M==64`, `N==512` (exact equality),
  which is structurally IDENTICAL to `cheats/c3_hardcoded_shape.py`. Loosened to `K<=128`,
  `M<=128`, `N<=512` (the actual single-tile hardware bound, rule 1's "never change what an
  honest kernel needs to pass" honored: the original shape still passes, and now so does the
  new one — confirmed live, `numerics 2/2 shapes passed`). With the fix in place, the cheat
  (which keeps the hardcoded `==` asserts) now fails on the new shape:
  `AssertionError: expected K=128, got 64` — confirmed live. Without this shape, no
  augmentation tier could ever have exposed C3, because level 3's declared class had exactly
  one point in it; A2a/A2b generate shapes alongside the declared ones, but a single-shape
  class gives a hardcoded-constant cheat nothing to disagree with.

  **Tolerance: measured, not guessed.** Ran every honest reference kernel (levels 1-4, all
  shapes including the new level-3 one) against its NumPy reference and recorded
  `max(|got-want|) / RMS(want)` (same metric `describe_mismatch` already uses) per shape:
  level 1 ≈2.4e-7, level 2 and level 3 exactly 0.0 (pure data movement, no arithmetic), level
  4 up to 3.103e-6 (the shape with the most matmul accumulation steps,
  `K=256 M=512 N=1024`). **Max over everything honest: 3.103e-6.** Set
  `TOL_BY_DTYPE["float32"] = 1e-4` — roughly 32x above that measured noise floor (room for
  legitimate float32 roundoff) and roughly 10x below `cheats/c6_almost_right.py`'s error
  (a uniform `+1e-3` offset against RMS≈1 data is a ≈1e-3 relative error — confirmed live:
  "worst error 0.00099 ... tolerance 0.0001"). `bfloat16` is set to a PROVISIONAL `3e-2`
  (never tighter than float32, per rule 10) since no current test shape uses bfloat16 — see
  Phase 1 findings; this number is not measured and should be replaced the day bf16 test data
  exists. `describe_mismatch(got, want, tol=None)` now resolves `tol` from `got`'s own dtype
  via `tol_for_dtype()` when the caller doesn't pass one explicitly, so both `agent.grade()`
  (which never passed a `tol`) and `verify()`/`--check` (CLI `--tol` now defaults to `None`,
  i.e. auto) pick up the new, dtype-aware default without each call site having to know about
  it. **Re-verified all 4 levels' honest references still score 1.0 after this change** (shown
  live above). This is written up for `CHECKER.md` later; the numbers above are the full
  record of how `tol` was chosen.

  **A4 redesigned to "run twice, compare," per the adjustment, and confirmed it's the only
  option that fits.** The harness does not own the output buffer (Phase 1 finding, reconfirmed
  here): every kernel allocates its own `nl.ndarray(buffer=nl.shared_hbm)` and returns it, so
  there is no pre-existing buffer to NaN-prefill before the kernel runs — the "pre-fill with
  NaN" branch the original plan described simply has no buffer to act on here.
  `augment.output_changes_with_input()` runs the kernel on two different random
  instantiations of the SAME shape (seed and seed+10007) and fails if the two outputs are
  `np.allclose(..., equal_nan=True)` to each other. The `equal_nan=True` choice is deliberate:
  two all-NaN outputs (do_nothing's measured behavior, Phase 2) also count as "did not
  change," so A4 catches C1 and C4 by the same single mechanism, without relying on the NaN or
  "100% zeros" heuristics already in `describe_mismatch` — those two heuristics happen to
  catch C1/C4 too today, but A4 is a mechanism-independent backstop that would still work if a
  "do nothing" cheat somehow returned something other than NaN or zero.

  **A2 split into A2a (extra, evenly-dividing shapes) and A2b (ragged: prime dim, dim of 1,
  non-tile-multiple), both inside each level's own declared class, implemented in
  `augment.py::extra_shapes(level_n, tier)`.** Ran EVERY HONEST REFERENCE KERNEL against A2b
  FIRST, before using augmentation for anything else, as asked. Result: **no failures.**
  Live: level 1 `48/48` checks passed (4 original + 1 A2a + 1 A2b shape, each with the base
  numeric check + 6 A3 hostile variants + 1 A4 check once --augment is on); level 2 `64/64`;
  level 3 `32/32`; level 4 `40/40`. **This is a finding, not a regression**, exactly as framed
  in the instruction: today's honest reference kernels already generalize correctly to a
  prime partition dimension (37), a dimension of exactly 1, and (for level 1) a degenerate
  `C=1` case — none of them needed a code change to pass A2b. Level 4's A2b is deliberately
  empty (`[]`): its declared class requires `M`, `N`, `K` to be exact multiples of the tile
  size (asserted in `reference_level4.py` itself, lines 48-53), so no ragged shape is even
  legal input for that level — logged as a scope boundary, not a gap to close.

  **A3 tiny-value hypothesis, confirmed empirically, isolated from the tolerance fix.** Ran
  `cheats/c6_almost_right.py` through `--augment --tol 0.02` (the OLD global tolerance,
  deliberately reused here so the A3 tiers are tested on their own merit, independent of the
  new tighter default): of 64 checks (8 shapes x (1 base + 6 hostile + 1 A4)),
  **exactly 8 failed — the "tiny" variant, on every single shape, and nothing else.**
  `large_magnitude` passed on every shape even at the old loose tolerance. This directly
  confirms the correction logged in Phase 2: a fixed `+1e-3` absolute offset is a LARGER
  relative error against smaller-magnitude data, not larger-magnitude data. The plan's A3 row
  has been corrected accordingly (section 5).

  **Tamper-check parity: `verify()`/`--check` now runs the same `check_inputs_untouched`
  snapshot-and-compare that `agent.grade()` always has.** This closes the C5 path mismatch
  recorded in Phase 2 findings directly — re-run live: `C5 input_tamper ... check rc=1
  (caught)`, message "THE KERNEL MODIFIED ITS INPUT (argument 0)," matching `agent.grade()`'s
  message exactly, because it's now the same function call.

  **`run_cheats.py`'s `check_message` now extracts the real failure line, not the generic
  "improve THIS message" trailer.** `extract_failure()` scans `verify()`'s stdout for any of
  the known failure markers (`NUMERICAL MISMATCH`, `NON-FINITE OUTPUT`, `RAISED during`, `THE
  KERNEL MODIFIED`, etc.) and returns that block, falling back to the `numerics N/M` summary
  line on a clean pass. Confirmed live in the re-run CSV: C6's `check_message` is now
  "NUMERICAL MISMATCH: worst error 0.00099 of the output's RMS (1.01), tolerance 0.0001 | at
  index (0, 1): ... | 100.0% of elements are outside tolerance | ..." instead of the useless
  "improve THIS message before you touch the prompt."

  **Cheats re-run after all of the above (A0, unaugmented, no `--augment` flag) —
  `results/cheats_before.csv` updated in place:**

  | cheat | name | level | `agent.grade()` reward | grade result | `--check` rc | check result | path mismatch |
  |---|---|---|---|---|---|---|---|
  | C1 | do_nothing | 1 | 0.50 | caught | 1 | caught | no |
  | C2 | partial_rows | 2 | 0.88 | caught | 1 | caught | no |
  | C3 | hardcoded_shape | 3 | **0.75** | **now caught** | 1 | **now caught** | no |
  | C4 | constant_output | 1 | 0.50 | caught | 1 | caught | no |
  | C5 | input_tamper | 2 | 0.50 | caught | 1 | **now caught** | **no (mismatch closed)** |
  | C6 | almost_right | 2 | **0.50** | **now caught** | 1 | **now caught** | no |
  | C7 | edge_skip | 2 | 0.50 | caught | 1 | caught | no |
  | C8 | path_mismatch | 2 | 0.62 | caught | 1 | caught | no |

  **Headline: all 8 cheats are now caught at A0, with `--augment` off, by three structural
  fixes alone (level 3's second shape, the tightened measured tolerance, and the
  tamper-check wired into `--check`) — before any shape/value augmentation tier runs at
  all.** This changes what Phase 4's "tiers catch cheats" table will actually show: the
  interesting measurement there will be less "A0 passes, A-tier catches" (since A0 is now
  already fully closing everything we have cheats for) and more "how much MORE expensive is
  the fully augmented checker, for zero additional catches on this cheat set" — i.e., Phase 4
  should go hunt for a NEW cheat that slips past the hardened A0 and needs an augmentation
  tier specifically, so the table has something to show besides a column of already-caught.

  **Partial credit, logged as asked, weights unchanged.** `WEIGHTS` in `agent.py` was not
  touched. C1 (do_nothing) scores 0.5 = `parses` (0.1) + `rules` (0.2) + `runs` (0.2), with
  `correct` (0.5) withheld entirely. `parts["runs"] = True` is set right after
  `simulate_and_count` returns without raising (`agent.py:131`), BEFORE the numeric comparison
  runs — and `simulate_and_count` genuinely does succeed for C1 (the kernel executes cleanly,
  it just never writes its output), so "runs" is honestly earned; `describe_mismatch` then
  catches the NaN and withholds all of `correct`. 0.5 is the correct, honest score for
  "parses, rule-clean, executes without error, but wrong" — not a scoring bug. C2 (partial_rows) scores 0.875 because 3 of
  its 4 shapes are numerically correct (only the one shape with `sz_p=128>64` exposes the
  cap), so `correct` partial-credits at `0.5 * 3/4 = 0.375`, plus the 0.5 from
  parses+rules+runs = 0.875. Both numbers are the WEIGHTS formula working as designed on a
  kernel that is partially right; they are not evidence of a scoring bug and no weight was
  changed to "fix" them.

  Files added/changed: `augment.py` (new), `nkibench.py` (TOL_BY_DTYPE, `tol_for_dtype`,
  `describe_mismatch` dtype-aware default, `verify()` tamper check + `--augment` wiring,
  `LEVELS[3]` second shape, CLI `--tol`/`--seed`/`--augment`), `reference_level3.py`
  (generalized asserts), `run_cheats.py` (`extract_failure`). Synced to the pod and validated
  there; `/workspace/baseline-copy` untouched throughout.

- **2026-10-10 ~21:10 UTC — Phase 4, reframed: second-generation cheats against the hardened
  A0, cumulative-tier measurement (`results/cheats_after.csv`, `results/cost.csv`), section 8
  tables filled in.** All against seat-66; `/workspace/baseline-copy` untouched.

  **`nkibench.verify()` given a `tiers` parameter** (`ALL_TIERS = ("a1","a2a","a2b","a3","a4")`)
  so each tier can be turned on independently and cumulatively, instead of the Phase 3
  all-or-nothing `--augment` bool (which still works, as shorthand for the full set). This is
  what makes a cumulative-tier CSV possible at all; without it, `verify()` could only report
  "A0" or "everything."

  **C9 memorise_all_shapes (level 3) — built and confirmed exactly as designed.** Keeps a
  Python `MEMORIZED = {(128,64,512),(64,32,256)}` set (level 3's two current declared shapes)
  and does the real `nc_matmul` only when `(K,M,N)` is in it; otherwise zeroes the output.
  Measured live: **passes A0 and +A1** (`numerics 2/2 checks passed` both times — correct,
  since neither A1's fresh seed nor A0 changes the shape), **caught starting at +A2a**
  (`NON-FINITE OUTPUT: 18432 NaN`, the new unmemorized shape `K=96,M=48,N=384`), and stays
  caught through +A2b/+A3/+A4. This directly demonstrates the Phase 3 prediction: a
  multi-entry "memorize the test set" cheat defeats a single extra declared shape just as
  easily as it defeats zero, but a genuine augmentation tier (which doesn't care how many
  entries were memorized) still catches it.

  **C9's "unrecognized shape" branch returns NaN, not a clean zero — a broader version of the
  C1 finding.** `nisa.tensor_scalar(dst=result_sbuf, data=result_sbuf, op0=nl.multiply,
  operand0=0.0)` multiplies an UNINITIALIZED SBUF tile by zero; if the result were genuinely
  0 (finite) × 0, it'd be 0, but the measured output is NaN. This means **`nki.simulate`
  fills newly allocated SBUF tiles with NaN, not just unwritten `shared_hbm` outputs** (the
  original C1 finding was scoped to `shared_hbm`; this extends it to `sbuf` too). Multiplying
  NaN by zero is still NaN in IEEE float arithmetic, which is why the zero-out trick doesn't
  actually zero anything here. Caught either way (`describe_mismatch`'s NON-FINITE branch),
  but worth knowing precisely what "write zeros" cheats actually produce on this simulator.

  **C11 assume_divisible (level 2) — built and confirmed exactly as designed.** Tiles the
  partition axis as `cap = (sz_p // 8) * 8` and only copies rows `[0:cap]`. Measured live:
  **passes A0, +A1, and +A2a** (`numerics 4/4` then `5/5 checks passed` — every one of level
  2's four original partition sizes {32,128,64,8} and A2a's extra shape's 16 are multiples of
  8), **caught starting at +A2b specifically** (`NON-FINITE OUTPUT: 60 NaN ... first at
  (32, 0)`, from the ragged shape with `sz_p=37`: `cap=(37//8)*8=32`, dropping rows 32-36 —
  exactly 5 rows × 12 free positions = 60 elements, matching the message exactly), and stays
  caught through +A3/+A4. This is the cleanest demonstration in the whole cheat set of why the
  A2a/A2b split matters: a generic "more shapes" tier (A2a) would NOT have caught this cheat,
  only a genuinely ragged one (A2b) does, on this specific axis.

  **C10 fixed_input_lookup — attempted, dropped. Concrete, reproducible reason, not a
  difficulty judgment call.** The design needed the kernel to recognize the exact seed=0 input
  (so it's correct at A0 but wrong once A1 changes the values at the SAME shapes). Three
  things were tried, each with direct empirical evidence:
  1. Reading a raw Python float out of an indexed tensor element inside an `@nki.jit` function
     and comparing it (`float(in_tensor[0,0,0])`) — **confirmed to fail live**:
     `TypeError: float() argument must be a string or a real number, not 'NkiTensor'`.
     Indexing an NKI tensor, even under `nki.simulate`, returns a wrapped `NkiTensor`
     (`nki.language.tensor.NkiTensor`), not a plain Python/numpy scalar, and it has no `.item()`
     either (confirmed via `hasattr(v, 'item') == False` in a probe kernel).
  2. Reducing the WHOLE tile to one true scalar via `nl.sum(tile, axis=[0,1])` (both the free
     AND partition axes) so it could be compared with `nl.equal(fp, CONST)` (which does accept
     a tile vs. a plain Python/float scalar, per its own docstring) — **confirmed to fail
     live**: `AssertionError: SBUF and PSUM tensors must have at least 2 dimensions
     (partition-dim and free-dim)`, raised inside `nl.sum` itself. A full reduction to a true
     scalar is structurally disallowed for an SBUF-resident result.
  3. Reducing only the free axis (keeping a per-partition-row scalar, which IS allowed) and
     comparing that against a baked PER-ROW constant array — this just relocates the original
     problem, since there is no primitive anywhere in `nl`/`nisa` to materialize a host-side
     literal array into an SBUF/PSUM tile (only `dma_copy` between HBM/SBUF/PSUM and
     `tensor_scalar` with a single scalar operand are available) — not attempted further since
     it reduces to the same unsolved sub-problem.

  **Conclusion, stated plainly: this NKI dialect does not expose a way for a kernel to branch
  on the concrete VALUE of its own input inside Python control flow, and item (2) shows this
  is enforced at the API level, not merely inconvenient.** That is arguably a GOOD property of
  the language for real hardware (real kernels can't branch on data values without explicit,
  limited predicated-select ops like `nisa.affine_select`/`nisa.range_select`, which select
  between tensors by position/pattern, not by comparing to an arbitrary baked constant) — so
  this is a reassuring finding about the kernel language's shape, not a weakness of the
  checker. `cheats/c10_fixed_input_lookup.py` was deleted (not left as a broken file); this
  entry is the permanent record of the attempt.

  **C12 precision_shortcut — attempted (on paper, before writing code), dropped.** The design
  needed an op that's numerically fine at O(1) magnitude but wrong at A3's tiny (1e-6 to 1e-3)
  or large (±1e4) extremes, while still passing A0. Worked through the arithmetic for all three
  ops before writing anything:
  - A dtype downcast (e.g. route through `nl.bfloat16`/`nl.float16` mid-computation) was the
    obvious first idea, but bf16/fp16 relative precision (~0.4%/~0.05%) already exceeds the
    NEW tolerance (1e-4) at NORMAL magnitude too (Phase 3) — so it fails "fine at normal
    magnitudes" before even reaching the extremes; ruled out analytically, not empirically.
  - A fixed-scale-factor trick for matmul (scale operands down by a constant before
    multiplying, rescale after) is lossless in float32 unless it causes overflow/underflow.
    Worked the numbers for all three A3 regimes at our actual shapes (K≤128 or tiled):
    normal (~O(1) inputs) stays in float32's representable range after scaling; LARGE (±1e4)
    scaled by the tested factor still lands around 1e4-1e6, nowhere near float32's ~3e38
    ceiling; TINY (1e-6 to 1e-3) scaled down still lands around 1e-8 to 1e-5, nowhere near
    float32's ~1e-38 subnormal floor. **Float32's ~76 orders of combined dynamic range
    (1e-38 to 3e38) comfortably swallows our entire test range (1e-6 to 1e4, only 10 orders)
    without ever approaching an overflow/underflow cliff** — so no plausible fixed-scaling
    bug is detectable with these magnitudes.
  - Catastrophic cancellation (the mechanism the official challenge doc names for layernorm's
    `E[x²]-E[x]²`) needs a formula that subtracts two near-equal LARGE quantities. None of our
    4 implemented ops (avgpool-sum-then-scale, pure-copy transpose, single matmul-accumulate)
    naturally contains such a subtraction, and a prefix-sum-difference trick worked out
    analytically for avgpool: cancellation error at our shape sizes (≤32 elements per pooled
    axis) is ~1e-6 relative even at 1e4 magnitude -- far under the 1e-4 tolerance, so it
    wouldn't register as a failure either.

  **Conclusion: a genuine "fine normally, wrong at the extreme" precision bug needs an
  operation shaped like softmax's overflow or layernorm's cancellation (both explicitly named
  in `CHALLENGE-kernel-agent.md`'s operation ladder, levels 5 and 8) — and none of our 4
  CURRENTLY IMPLEMENTED levels (avgpool, transpose, matmul single/tiled) has that shape.** This
  is a real scope boundary of the current ladder, not a checker weakness: if/when softmax or
  layernorm are added as levels, C12 becomes straightforward to build (naive `exp()` without
  max-subtraction, or `E[x²]-E[x]²` directly), and A3's tiny/large tiers would be exactly the
  right tool to catch it. No file was written for C12.

  **`results/cheats_after.csv`** (10 cheats × 6 tiers = 60 rows) and **`results/cost.csv`**
  (10 cheats + 4 honest references × 6 tiers = 84 rows) written by
  `run_cheats.py --tiered`, a new function (`tiered_main()`) added alongside the original
  Phase 2 `main()` — `results/cheats_before.csv` was not regenerated or modified, per the
  instruction. A1's seed is fixed at `424242` (not time-based) so this report is reproducible,
  per rule 8.

  **Headline cost result:** the full augmented checker (+A4) runs **8x** as many individual
  checks as A0 (every shape now also runs 6 hostile-value variants plus one A4
  output-sensitivity check: `shapes × 8` vs. `shapes × 1`). Wall-clock cost scales
  similarly but is dominated by level 4 (matmul, tiled): its honest reference goes from 0.70s
  at A0 to 6.49s at full augmentation (~9.3x), while levels 1-3 stay under 0.2s even fully
  augmented, their shapes being much smaller. **Every honest reference kernel still passes at
  every single tier** (re-confirmed here across all 6 tiers, not just the Phase 3 `--augment`
  check) — no regression anywhere in this phase either.

  **Section 8 tables filled in** (cheats-vs-tiers, with an A2a/A2b split instead of one A2
  column, A5 omitted as not implemented; cost table expanded from 2 rows to all 6 cumulative
  tiers). Both shown to the user; see the tables above in section 8 for the final numbers.

- **2026-10-10 ~21:45 UTC — why C10 and C12 could not be built to pass A0 (crisp restatement
  of the Phase 4 entry above, for anyone skimming just the Findings log).**

  **C10 fixed_input_lookup: blocked by the NKI API itself, not by effort.** Its design needed
  the kernel to look at its OWN input's values (to know "is this the default seed=0 run or
  not") and branch in plain Python. Two concrete, reproducible errors shut this down:
  1. `float(in_tensor[0,0,0])` → `TypeError: float() argument must be a string or a real
     number, not 'NkiTensor'`. Indexing a tensor inside `@nki.jit`, even under
     `nki.simulate`, returns a wrapped `NkiTensor`, never a plain Python/numpy scalar — and it
     has no `.item()` either.
  2. `nl.sum(tile, axis=[0,1])` (reducing BOTH the free and partition axes, to get one true
     scalar to compare) → `AssertionError: SBUF and PSUM tensors must have at least 2
     dimensions`, raised inside `nl.sum` itself. A full reduction to a scalar is disallowed by
     construction.
  Net: there is no documented path from "a tensor's contents" to "a Python value I can put in
  an `if`" anywhere in `nl`/`nisa`. That's a property of the kernel language (real hardware
  can't branch on data values without an explicit predicated-select primitive like
  `nisa.affine_select`, which selects by POSITION/pattern, not by comparing to an arbitrary
  baked constant), not a gap in this harness.

  **C12 precision_shortcut: blocked by arithmetic, worked out before writing any code.** Its
  design needed an operation that's numerically fine at O(1) magnitude but wrong at A3's tiny
  (1e-6 to 1e-3) or large (±1e4) extremes. Three concrete mechanisms were checked against our
  actual shapes and tolerance:
  1. A dtype downcast (route through bf16/fp16) fails "fine at normal magnitude" FIRST — bf16/
     fp16 relative precision (~0.4%/~0.05%) already exceeds the new 1e-4 tolerance at normal
     (O(1)) magnitude, before any extreme value is even involved.
  2. A fixed-scale-factor trick (scale down, multiply, scale back up) is lossless in float32
     unless it overflows/underflows. Worked the numbers for our shapes at normal, large
     (±1e4), and tiny (1e-6) magnitudes: every intermediate value stays within roughly 1e-8 to
     1e6 — nowhere near float32's ~1e-38 (subnormal) or ~3e38 (overflow) edges. **Our entire
     test range (1e-6 to 1e4, 10 orders of magnitude) sits deep inside float32's ~76 orders of
     dynamic range**, so no plausible fixed-scaling bug is detectable there.
  3. Catastrophic cancellation (the mechanism named in `CHALLENGE-kernel-agent.md` for
     layernorm's `E[x²]-E[x]²`) needs a subtraction of two near-equal LARGE quantities. None
     of avgpool/transpose/matmul (our only 4 implemented levels) contains one; a
     prefix-sum-difference variant for avgpool was worked out analytically and its
     cancellation error at our shape sizes (≤32 elements per pooled axis) comes out to ~1e-6
     relative even at 1e4 magnitude — far under the 1e-4 tolerance.
  Net: a genuine "fine normally, wrong at the extreme" bug needs an operation shaped like
  softmax's overflow or layernorm's cancellation. Our ladder doesn't have one of those yet
  (levels 1-4 only); C12 becomes easy to build the day one is added.

- **2026-10-10 ~21:50 UTC — cost-effectiveness: which tiers earned their keep on this cheat
  set, and a recommended default.**

  Using `results/cost.csv`'s honest-reference timings (averaged across levels 1-4) and
  `results/cheats_after.csv`'s catch data, the INCREMENTAL cost of each tier and how many
  cheats in our current set it catches THAT NO EARLIER TIER ALREADY CAUGHT:

  | tier | incremental avg seconds/kernel | NEW cheats caught (this set) | notes |
  |---|---|---|---|
  | A0 | — (baseline, 0.184s avg) | 8 (C1-C8) | the hardened baseline itself, from Phase 3's structural fixes, not an augmentation tier |
  | +A1 | +0.049s | 0 | would have caught C10, which could not be built (see above) — currently pure cost on this set, but cheap, since it doesn't add test count, only reseeds |
  | +A2a | +0.007s | **1 (C9)** | essentially free AND catches a real cheat — the best ratio measured |
  | +A2b | **-0.003s** (within noise — more shapes, but smaller/cheaper ones) | **1 (C11)** | also essentially free, also catches a real cheat |
  | +A3 | +1.196s | 0 | by far the most expensive tier (6 hostile variants per shape = 6x the test count) and, on THIS cheat set, catches nothing new -- C6 (its original target) is already caught by Phase 3's tightened tolerance at A0 |
  | +A4 | +0.322s | 0 | moderate cost; catches nothing new on this set because C1/C4 are already caught by `describe_mismatch`'s NaN/zero heuristics, but it is a MECHANISM-INDEPENDENT backstop (doesn't rely on what uninitialized memory happens to contain) |

  **Reading this honestly: A2a and A2b are the only tiers that bought a real catch on this
  cheat set, and they were nearly free (±0.007s). A1, A3, and A4 cost real time today for zero
  additional catches on this specific set** — A1's target (C10) and A3's original target (C6,
  pre-tolerance-fix) don't exist as catchable cheats right now, and A4 is redundant with
  heuristics that already fire. That is not evidence those tiers are useless in general — A4
  is a principled backstop that doesn't depend on simulator memory-initialization behavior,
  and A3 is exactly the tool a future softmax/layernorm level (C12's blocker) would need — but
  it is the honest, measured answer to "which tiers are earning their keep right now."

  **Recommended default tier set for routine use (e.g. every agent round, not just an audit):
  A0 + A1 + A2a + A2b.** This is the full "cheap" set — A2a/A2b are the two tiers with a
  demonstrated catch on this cheat set, and A1 costs almost nothing extra (same shapes, same
  test count, just a different seed) for insurance against hardcoded-default-input cheats,
  even though we couldn't build one ourselves to prove it live. **A3 and A4 are recommended
  for a periodic or pre-submission full audit, not every round**: A4 for its
  mechanism-independent robustness (worth the +0.322s when it matters), and A3 specifically
  once the ladder grows a level with a genuine magnitude-dependent trap (softmax, layernorm),
  at which point its +1.196s becomes money well spent rather than the most expensive tier with
  nothing to show for it.

- **2026-10-10 ~22:05 UTC — Phase 6 (partial): `CHECKER.md` and `results/eval_set.md` written
  while the baseline agent run is still in progress on seat-66.** Both only needed the checker
  code (`nkibench.py`, `augment.py`), not the baseline's output, so they didn't have to wait.
  `CHECKER.md` condenses every rule from sections 1-4 above (static rules, numerics/tolerance,
  traffic bar, augmentation tiers) into one per-rule accepts/rejects/why reference, plus the
  measured tolerance reasoning and the recommended default tier set (both already logged
  above; this is where they become the deliverable). `results/eval_set.md` is the literal,
  exhaustive shape/dtype/value/seed table for all four implemented levels at all 6 tiers,
  including the two intentional gaps already on record (level 4 has no A2b shapes; level 8
  has no augmentation coverage at all since `augment.extra_shapes()` never got a case for it).
  `NOTE.md` is intentionally not started yet — it needs Phase 4b's false-verified
  count/taxonomy and Phase 5's regression run, neither of which exist until the baseline
  finishes.

- **2026-10-10 20:08 UTC — Phase 5 started: `/workspace/hardened-copy` frozen, augmented
  agent run launched in the background.**

  **The baseline (started Phase 0) actually finished at ~20:00 UTC** (confirmed: no
  `agent.py` process remained; `results/baseline.log` has the full 3-run summary;
  `attempts.jsonl` has 252 entries) — it ran for about an hour, as the plan warned it might.

  **`agent.py` was given a REAL `--augment` flag, not a workaround.** Its own `grade()`
  function never called `nkibench.verify()` — it has always had its own inline
  parse/rules/simulate/compare loop (Phase 1 finding). Extending THAT loop, rather than
  routing through `verify()`, was the only option; done by adding an `augment=False`
  parameter to `grade()` that, when true: extends `cases` with `augment.extra_shapes(level,
  "a2a")` and `"a2b"`; after each case's existing checks, loops `augment.hostile_args(args,
  seed=0)`'s 6 hostile variants (A3) and one `augment.output_changes_with_input()` call (A4),
  using the exact same `total`/`passed` accounting `nkibench.verify()` uses for its own
  cumulative-tier counting. Wired through `solve()` and a new `--augment` CLI flag in
  `main()`. **Deliberately seed=0 throughout, not a fresh A1 reseed** — this keeps the
  augmented run's base input VALUES identical to the baseline's, so the regression comparison
  in Phase 5 isolates "how much more does the checker look at" from "did the random data also
  change," which would be a confound. Sanity-tested before trusting it for the real run:
  `agent.py --offline --level 2 --rounds 2 --samples 1 --augment` replays
  `reference_level2.py` on round 1 and it scored 1.0 / SOLVED even with every A2a/A2b/A3/A4
  check active — confirms the wiring doesn't accidentally fail an honest kernel.

  **`/workspace/hardened-copy`** frozen from `/workspace/projects/02-kernel-agent` (the
  current, Phase-1-through-4 working copy — NOT `/workspace/baseline-copy`, which stays the
  original pre-hardening snapshot untouched for the rest of this project). Stray
  `attempts.jsonl`/`__pycache__`/`results/` left over from the `--augment` sanity test were
  removed from the copy before starting the real run, so its own `attempts.jsonl` starts
  clean.

  **Started:** `cd /workspace/hardened-copy && python agent.py --all --rounds 8 --samples 4
  --context 8192 --repeat 3 --augment > results/augmented_agent.log 2>&1`, PID 20353,
  **2026-10-10 20:08 UTC**. Confirmed running (model server was confirmed ready via `curl
  localhost:8000/v1/models` immediately before launch). Will be compared against
  `results/baseline.log` once it finishes.

- **2026-10-10 ~20:15 UTC — Phase 4b: every baseline kernel re-scored under the hardened
  checker. False-verified count: 0. Taxonomy built.** Full tables in `results/taxonomy.md`;
  this entry has the reasoning behind the numbers.

  **Method:** every one of the 252 attempts in `/workspace/baseline-copy/attempts.jsonl` (one
  row per model sample, across all rounds/levels/repeats of the baseline) re-scored via THIS
  directory's `agent.grade(code, level, augment=False)` — the hardened A0, deliberately
  without the augmentation tiers, since "false verified" is specifically about what Phase 3's
  structural fixes (tighter tolerance, level 3's second shape, tamper-check parity) catch
  that the ORIGINAL checker missed, not about what a new augmentation tier would catch.
  Temporary script `rescue_rescoring.py` (not a deliverable, not committed — the deliverables
  are `results/false_verified.csv`, `results/taxonomy_raw.csv`, and `results/taxonomy.md`).

  **False-verified: 0, and here's why that's a real result, not a null one.** Of the 252
  attempts, only **3 ever scored 1.0 under the ORIGINAL checker** (all level 2, round 0, from
  2 of the 3 repeats — matching `baseline.log`'s "level 2: solved 2/3"). **All 3 also score
  1.0 under the hardened checker.** Checked by hand which cheat-adjacent category each would
  have fallen into if it HAD regressed (it didn't): these are genuinely, unambiguously
  correct level-2 transposes — not near-misses sitting just inside the old 2e-2 tolerance that
  the new 1e-4 tolerance would expose. **Verdict for all 3: genuinely correct, not "only less
  precise."** There is no case in this baseline where "genuinely wrong vs. only less precise"
  needed to be judged, because nothing flipped.

  **The much bigger, more useful finding: 249 of 252 attempts fail with a RAISED EXCEPTION,
  and NONE of them ever reach the numeric-comparison stage at all.** Re-scoring found zero
  attempts in any of the checker's numeric-layer categories (`non_finite_output`,
  `zero_output`, `numerical_mismatch`) and zero in `rule_violation` — every single failure is
  an uncaught Python exception raised while the kernel was actually running inside
  `nki.simulate`, before `describe_mismatch` is ever called. Breakdown (full table in
  `results/taxonomy.md`):
  - **Level 1 (96 attempts, 0 correct):** `invented_api_call` (48 — guessed an `nl`/`nisa`
    attribute that doesn't exist), `dma_shape_mismatch` (24), `invalid_keyword_arg` (12),
    `wrong_buffer_placement` (12).
  - **Level 2 (36 attempts, 3 correct):** `dma_shape_mismatch` (17), `out_of_bounds_index`
    (16).
  - **Level 3 (72 attempts, 0 correct):** `reshape_disallowed` (48 — called `.reshape()`,
    which NKI tensors don't support), `partition_exceeds_max` (21), `dma_shape_mismatch` (3).
  - **Level 4 (48 attempts, 0 correct):** `partition_exceeds_max` (29 — level 4's larger
    shapes make this the dominant failure here), `assign_shape_mismatch` (18),
    `wrong_buffer_placement` (1).

  **These 11 fine-grained categories collapse into three families, out of 249 failures:**

  | family | rolled-up categories | count | % of 249 |
  |---|---|---|---|
  | shape / tiling | `dma_shape_mismatch` (44), `out_of_bounds_index` (16), `reshape_disallowed` (48), `partition_exceeds_max` (50), `assign_shape_mismatch` (18) | **176** | **70.7%** |
  | invented API | `invented_api_call` (48), `invalid_keyword_arg` (12) | **60** | **24.1%** |
  | buffer placement | `wrong_buffer_placement` (13) | **13** | **5.2%** |

  Seven in ten failures are shape/tiling mistakes (wrong tile dimensions, bad destination
  slices, partition overflow, `.reshape()` misuse) — not wrong names, not wrong memory
  regions. Guessing a nonexistent `nl`/`nisa` name or keyword is the clear second family
  (24.1%); confusing sbuf/psum placement is a distant third (5.2%).

  **Implication for "does the hardening effort matter for THIS baseline": only partially, and
  it says something important about priority.** Phases 2-4's hardening (tolerance, extra
  shapes, hostile values, tamper detection) all operate AFTER a kernel successfully runs and
  produces a same-shaped, finite output — a stage this particular agent/model
  (Qwen3-8B, terse prompting, 8 rounds) reaches on only 3 of 252 attempts. The highest-leverage
  improvement for THIS agent is not more checker hardening at all — it's better error
  messages/API-name hinting to get PAST the raised-exception stage (which `agent.py`'s
  `enrich()` function already attempts for exactly these categories: invented API names,
  wrong keyword args, buffer placement, out-of-bounds, etc. — see `agent.py:248-347`). The
  checker is honest about failure here (0 false-verified) specifically because it almost
  never gets the chance to be dishonest; that is a different, and arguably more fundamental,
  finding than "the checker has holes."

  **Stated plainly: false-verified is 0 because 249/252 attempts crash before numeric
  checking is ever reached — so at THIS model's current skill level, the loopholes Phases
  2-4 closed are PREVENTIVE, not (yet) REACTIVE, and the hand-written cheats are the proof
  they're closed regardless.** Every one of Phases 2-4's fixes (level 3's second shape, the
  measured tolerance, tamper-check parity, A2a/A2b/A3/A4) acts on a kernel that already runs
  and returns a same-shaped, finite answer — exactly the stage this baseline reaches on 3 of
  252 attempts. That is NOT evidence the hardening was unnecessary: `cheats/c1_do_nothing.py`
  through `cheats/c11_assume_divisible.py` are hand-written kernels built specifically to
  reach that stage and look plausible once there, and the hardened checker catches all 10 of
  them (`results/cheats_before.csv`, `results/cheats_after.csv` — Phases 2-4). The baseline's
  0 false-verified and the cheat set's 10-for-10 catch rate are two sides of the same
  statement: the checker currently has nothing to be dishonest ABOUT in this baseline (the
  model never produces a plausible-but-wrong kernel), but it would catch one immediately if
  the model — or a better-prompted one, or a stronger one — ever did. The hardening is
  insurance against a failure mode this specific run didn't happen to exhibit, verified by
  construction rather than by observation.

  **Level 4 variance contradicts the repo's recorded "zero variance" claim.** This run's
  level 4 rewards across 3 repeats: `[0.62, 0.3, 0.62]` (from `baseline.log`'s own summary
  line). Section 9 of this file says "L4 0/5 always 0.62... Levels 1, 3, 4 have zero
  variance" — true for level 1 (`[0.3,0.3,0.3]` this run too) and level 3
  (`[0.3,0.3,0.3]` this run too), but **not true for level 4 this time.** One repeat (0.30)
  never got past the parses/rules stage before `give_up_after` triggered, the other two
  scored 0.62 as the old claim predicted. Logged as a correction, not a deletion, of the
  original claim per the "never delete findings" rule — section 9 above now carries a note
  pointing here. Possible explanations not investigated further here: this is only 3 repeats
  (vs. the original 5), or the live model server's behavior has genuinely drifted since
  whatever run produced the original 5-run claim (prompting, context, or model version could
  differ) — worth checking the original claim's provenance before trusting either number as
  "the" level-4 variance going forward.

- **2026-10-10 ~22:40 UTC — `NOTE.md` and `DEMO.md` drafted while the Phase 5 regression run
  is still in progress on seat-66 (confirmed alive and unaffected by this work — `ps aux`
  checked before and after).**

  **`NOTE.md`** is the one-page Phase 6 reproduction note, built from everything measured
  through Phase 4b: what ran and on what, the cheats-vs-tiers table condensed to its
  headline (10/10 caught, 8 at A0 alone), cost, the baseline's runs-and-spread (including the
  level 4 variance correction), "no regressions found at any tier," the false-verified=0 /
  taxonomy summary with the preventive-not-reactive interpretation, and limitations. The
  Phase 5 section is a clearly marked placeholder (`[PLACEHOLDER — ...]`) — it will be filled
  in once `results/augmented_agent.log` is complete and compared against `baseline.log`.

  **`DEMO.md`** is a 2-minute script, every command tested once live against seat-66 and
  timed before being written down (not estimated):
  1. `cheats/c3_hardcoded_shape.py` against the ORIGINAL checker
     (`/workspace/baseline-copy/nkibench.py`, pointed at the cheat file by path — nothing
     copied into `baseline-copy`) — **1.9s**, `numerics 1/1 shapes passed`.
  2. The SAME file against the hardened checker (`/workspace/projects/02-kernel-agent`) —
     **1.8s**, exit code 1, `numerics 1/2 checks passed` /
     `AssertionError: expected K=128, got 64` on the new shape.
  3. `cheats/c9_memorise_all_shapes.py` against the hardened checker, `--augment` OFF —
     **1.7s**, exit code 0, `numerics 2/2 checks passed`.
  4. The SAME file with `--augment` ON — **1.9s**, exit code 1,
     `NON-FINITE OUTPUT: 18432 NaN` on the unmemorized A2a shape.

  Total measured command time: **~7.2s**, leaving well over a minute of the 2-minute budget
  for narration. Chose C3 (not C6) per the instruction's "C3 or C6" option, specifically
  because C3's story — "one shape, hardcode it, pass; add a shape, same hardcode, fail" — is
  the more visually direct one to narrate live than C6's tolerance-margin story, and it sets
  up C9's "smarter version of the same idea" naturally.

- **2026-10-10 21:07 UTC (run finished) / ~22:55 UTC (analyzed) — Phase 5 complete: the
  augmented regression run finished, compared against `baseline.log`, and section 8's
  before/after table filled in. Two real deltas, both investigated rather than taken at face
  value, and neither turned out to be what it looked like at first glance.**

  **The augmented run finished at 21:07 UTC** (started 20:08 — ~59 minutes, comparable to the
  baseline's ~61 minutes). No process remained; `results/augmented_agent.log` has the full
  3-run summary; `attempts.jsonl` (256 entries) copied back as
  `results/augmented_attempts.jsonl`. (The background tool reported this job's wrapper shell
  as "failed, exit code 1" — that is the `nohup ... &` wrapper's own exit status, not the
  Python process: the log shows a normal, complete finish with no traceback, and the process
  list was empty, i.e. it exited on its own after printing the summary, not because it was
  killed. Logged here since it's a one-line investigation that could otherwise look alarming.)

  **Important methodological point, checked BEFORE trusting any delta:** the baseline and
  augmented runs each independently call the LIVE model (temperature 0.6, not greedy), so
  they are NOT the same kernels being re-scored under two checkers — they're two
  independently-sampled sets of model outputs. A solve-rate difference between them could be
  pure run-to-run model variance (already documented: section 9 calls level 2
  "luck-limited") rather than anything the hardened/augmented checker actually caught. To
  separate the two, every kernel the AUGMENTED run actually wrote was re-graded with
  `agent.grade(code, level, augment=False)` (hardened A0 only) and compared to its OWN live
  augmented score. **Result: 0 cases where hardened-A0 said 1.0 but the live augmented score
  was lower.** That means augmentation, as wired, did not catch anything in the augmented
  run's OWN kernels that the hardened checker's base shapes would have missed — any
  difference from the baseline run is attributable to sampling, not to the augmentation tiers
  catching a near-miss that happened to occur this time.

  **Level 2: solved 2/3 (baseline) vs. 1/3 (augmented), mean 0.77 vs. 0.53. NOT a real
  regression** — confirmed by the re-grading check above (0 flips). This is ordinary
  run-to-run variance on a level the project's own prior notes already call luck-limited; with
  only 3 repeats per run, a one-sample difference in solve count is expected noise, not
  evidence the checker changed anything. Would need many more repeats to say anything
  statistically meaningful about level 2's TRUE solve rate under either checker.

  **Level 4: mean 0.52 (baseline, `[0.62,0.30,0.62]`) vs. 0.83 (augmented, `[0.83,0.83,0.83]`,
  zero variance) — HIGHER reward under the supposedly stricter checker, and this IS
  explained, not dismissed as noise: it's a scoring-granularity artifact of how `augment=True`
  was wired into `agent.grade()`, not evidence the checker got easier.** Traced to the exact
  mechanism: `agent.grade()`'s A3 (6 hostile variants) and A4 (1 output-sensitivity check) are
  only run for a shape that ALREADY passed the base numeric check — they sit after the
  `continue` that skips a failing shape, mirroring `nkibench.verify()`'s own design exactly.
  Confirmed from the actual best attempt's feedback: `"8 of 12 checks passed. On K=256
  M=256 N=1024: raised AssertionError: dma_copy dst partition dimension 256 exceeds
  maximum 128..."` — level 4 has 5 shapes under augmentation (4 declared + 1 A2a; A2b is
  empty for this level), and `5 + 7×1 = 12` is exactly what you get when ONE shape passes the
  base check and all 7 of its extra hostile/A4 checks ALSO pass (`8 = 1 + 7`). Under the
  unaugmented baseline, that same one-shape-right kernel would score `0.5 + 0.5×(1/4) = 0.62`
  — which is exactly 2 of the baseline's 3 repeats. **So the underlying kernel behavior
  (gets exactly one shape right, crashes on the rest) is consistent across both runs; what
  changed is that augmentation gives MORE ways to earn partial credit on the ONE shape that
  already works, not fewer ways to pass.** This is a real design note for anyone extending
  this reward formula: weighting partial credit by `passed_checks / total_checks` means
  testing a passing case more thoroughly inflates its contribution to the overall reward,
  which could read as "got better" when the actual shape-level correctness (1 of 4/5) didn't
  change at all. A future fix, not implemented here: weight `correct` by UNIQUE SHAPES
  correct, independent of how many sub-checks were run on each one.

  **Net conclusion for the no-regression check: no instance, in either direction, of the
  augmented checker producing a WRONG verdict on a kernel that should have scored
  differently.** The level 2 delta is sampling noise; the level 4 delta is a partial-credit
  accounting artifact on an UNCHANGED underlying kernel behavior. Both are reported honestly
  in section 8's table above rather than smoothed over, per the instruction.

- **2026-10-10 ~23:15 UTC — precise follow-up on both Phase 5 deltas, requested explicitly:
  per-kernel numbers rather than the aggregate "0 flips" check above. Supersedes nothing
  (kept per "never delete findings"); sharpens it with exact figures.**

  **Level 2, per-kernel: of the augmented run's 48 level-2 attempts (with code), exactly ONE
  passes the ORIGINAL 4 declared shapes on its own** (`agent.grade(code, 2, augment=False)`
  reward = 1.00, round 0) — **and that same attempt also scored 1.00 LIVE, under full
  augmentation.** No other attempt gets anywhere near passing the original shapes (checked
  every one of the 48, not a sample). So the precise answer to "did any kernel pass the
  original shapes but fail only on an augmented shape/value": **no, zero such kernels
  exist in this run.** There is nothing for A1/A2a/A2b/A3/A4 to have rejected, because
  nothing besides that one already-passing kernel got far enough to be a candidate. The
  2/3→1/3 solve-rate drop is fully attributable to normal variance at n=3 (whether the
  model's round-0 sample that run happens to nail the transpose or not), not to any tier
  catching anything.

  **Level 4, precise re-score of the single BEST augmented-run attempt (round 0, live
  reward 0.8333, "8 of 12 checks passed"):** re-graded under `augment=False` (original 4
  shapes only) → **reward 0.625, feedback "1 of 4 checks passed. On K=256 M=256 N=1024:
  raised AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128..."** —
  the identical failure, on the identical shape, as **baseline's own best level-4 attempt**
  (also reward 0.625, also "1 of 4 shapes passed," also the same K=256 partition-dimension
  AssertionError). **This is the SAME underlying kernel behavior in both runs, not an
  improvement.** Working out the arithmetic that produced 0.8333 confirms exactly how:
  under augmentation this level has 5 base cases (4 declared + A2a's `K=384 M=384 N=512`);
  `total=12` only matches `5 + 7×1`, meaning exactly ONE of the 5 base cases passes (not
  two) — so **A2a's new shape does NOT pass either**, it just adds one more failing case to
  the denominator. The extra 7 (`passed=8=1+7`) are the 6 A3 hostile-value variants plus the
  1 A4 output-sensitivity check, run ONLY on the one shape that already passed, and all 7
  happen to also succeed (unsurprising: a correctly-computed single-tile matmul doesn't care
  what values it's fed). **Direct answer to the question as posed: the increase is neither
  "augmentation added a shape the kernel passes" nor "the kernel passes more of the original
  shapes" — it is a third mechanism, partial credit from testing the one already-passing
  shape more thoroughly, with shape-level correctness provably unchanged (0.625 under
  augment=False, identical to baseline).**

  **`EXPLAIN_ADULT.md`**: searched the full working tree and all git history
  (`find . -iname "*explain_adult*"`, `git log --all -- "*EXPLAIN_ADULT*"`) — this file does
  not exist anywhere in this repo. Nothing to update.

- **2026-10-10 ~23:40 UTC — AUGMENTATION INTRODUCED A REWARD-INFLATION BUG; FOUND BY
  MEASUREMENT; FIXED.** Confirmed the mechanism precisely, fixed `agent.py`'s `grade()`,
  re-verified against the exact kernel that exposed it, and re-confirmed every honest
  reference and all 10 cheats.

  **Confirmed mechanism (re-derived from the actual numbers, not assumed):** the previous
  `grade(augment=True)` incremented `total`/`passed` once per SUB-CHECK (1 for the base
  shape, +1 for each of A3's 6 hostile variants, +1 for A4) but only for shapes that already
  passed the base check — a failing shape contributed exactly 1 unit to `total`, a passing
  shape contributed up to 8. For the best augmented-run level-4 kernel: 4 of 5 base shapes
  fail (1 unit each = 4), the 1 passing shape contributes 1 (base) + 7 (its hostile/A4
  sub-checks, all of which also pass) = 8, giving `total=12`, `passed=8`, reward
  `0.5 + 0.5×(8/12) = 0.8333` — exactly the number measured live. **A passing shape was
  worth 8x a failing one in the denominator, which is the inflation.**

  **Fix, in `agent.py`'s `grade()` (WEIGHTS untouched, per the instruction):** restructured
  so every CASE (shape) is exactly one unit of credit, whether augmented or not. A shape now
  counts as passed only if its base check AND every one of its A3 hostile variants AND its
  A4 check all pass (checked in order, short-circuiting on the first sub-failure so the
  reported message names the specific thing that broke). The `correct` weight is now
  prorated over `len(cases)` — the number of SHAPES (declared + A2a + A2b), never over
  however many sub-checks ran. When `augment=False`, `cases == spec["shapes"]` and the code
  path is byte-identical to before the fix (no sub-check loop ever runs), so unaugmented
  behavior, including every number reported in Phases 0-4b, is completely unaffected.

  **Re-scored the exact kernel that exposed the bug** (the best augmented-run level-4
  attempt, round 0, originally logged live at 0.8333):
  - `agent.grade(code, 4, augment=True)` with the FIX: **0.60** — feedback "1 of 5 shapes
    passed," the same `K=256` partition-dimension failure as always. This is the CORRECT,
    non-inflated number for this tier: 1 of 5 shapes (4 declared + A2a's `K=384 M=384
    N=512`, which this kernel does NOT pass either) — lower than the old buggy 0.8333, and
    also lower than 0.625, because A2a legitimately adds one more shape this kernel fails,
    which should and now does pull the prorated score down, not up.
  - `agent.grade(code, 4, augment=False)` (the original 4 shapes only, no A2a, no
    augmentation at all): **0.625**, exactly matching baseline's own best level-4 attempt's
    reward and failure message. This is the apples-to-apples number against the pre-Phase-5
    baseline and confirms the underlying kernel behavior genuinely did not change between
    runs — only the (now-fixed) accounting did.
  - A full corrected 3-repeat re-run of the augmented agent was NOT performed (that's another
    ~hour of model calls); the single-kernel re-score above is sufficient to prove the fix
    and rule out a regression, since it is the EXACT kernel the original 0.8333 came from.

  **Re-verified after the fix, both required checks:**
  - **Every honest reference kernel (levels 1-4) scores exactly 1.0 under
    `agent.grade(augment=True)`** — re-run live, all four: `reward=1.0 PASS`. No regression
    from the fix.
  - **All 10 cheats (C1-C9, C11) are still caught under `agent.grade(augment=True)`** —
    re-run live: every one returns a reward below 1.0 (`C1: 0.50`, `C2: 0.9375`, `C3: 0.625`,
    `C4: 0.50`, `C5: 0.5625`, `C6: 0.50`, `C7: 0.50`, `C8: 0.625`, `C9: 0.75`, `C11: 0.875`).
    None of these numbers need to match their pre-fix values exactly (the proration
    denominator changed for any cheat that passes at least one augmented sub-check), only
    that every one stays below 1.0 — confirmed.

  **Why this matters beyond just this one number:** this is the second time in this project
  that a MEASURED result (not a code read-through) surfaced a real bug in the harness itself
  — the first was the Phase 1 byte-sizing fallback (STATE.md's "byte counting was 2x low");
  this is the second, and it was caught specifically BECAUSE the Phase 5 regression
  comparison insisted on explaining every delta rather than reporting the raw number. A
  reward formula that grants partial credit per SUB-CHECK rather than per SHAPE will always
  be vulnerable to this: more scrutiny on an easy case outweighs failure on a hard case,
  which is backwards for a graded harness whose whole point is to reward breadth of
  correctness, not depth of testing on whatever already works.
