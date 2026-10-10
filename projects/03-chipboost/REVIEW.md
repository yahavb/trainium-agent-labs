# CHIPBOOST: bottlenecks and what to change

*Review of all four branches, Oct 10 2026, ~13:00. Read from `origin/referee-timing`, `origin/kernels-search`,
`origin/redteam-agent` and the local dashboard work. Nothing was run on a chip.*

**Labels:** **[V]** verified by reading the code. **[est.]** an estimate, not a measurement.
Line numbers are from the branch named.

## TL;DR

1. **The loops don't exist yet.** `agent.py` (P3) and `search.py` (P2) aren't on any branch, and they gate 15:00.
2. **Turn off held-out shapes inside the loop.** Each check compiles about 10 kernels, and the referee's
   message names the held-out shape, which leaks it to the model. Run held-out only on each arm's best kernel
   at the end. This roughly halves the cost of each evaluation.
3. **Matmul only.** About 100 attempts per arm per seat fit before 18:30 [est.]. RMSNorm is a stretch goal.
   Drop SwiGLU.
4. **Three contract breaks to fix before merging:**
   - `run.py` calls a `check()` signature that doesn't exist.
   - The referee never returns `no_gain`.
   - Infrastructure errors are logged as `wrong` kernels.

---

## Everyone: critical path and budget

### Write the loop now (P3 `agent.py`, P2 `search.py`) [V: neither exists]

Minimal `projects/03-chipboost/agent.py`:
- Import `ask`, `ask_parallel` and `extract_code` from `02-kernel-agent/agent.py`. Drop its `grade()`, level
  ladder and `first_prompt`.
- Each round, generate every sample first, then grade each one with
  `speedcheck.check_isolated(file, op, heldout=False)`. One loop per seat, so referees don't fight over cores.
- Before appending each line to `attempts.jsonl`, fill in the fields the referee doesn't return: `seat`,
  `arm`, `run_id`, `attempt_no`, `round`, `code`, `prompt`, `response`, `prompt_tokens`. Then run
  `schema.validate`.
  - `ask()` (line 433) throws away `usage.prompt_tokens`, so make it return that.
  - The timeout and crash records from `check_isolated` have no `code_hash` or `sim_ok`; fill those in too.
  - The dashboard skips any line missing `kernel`, `arm`, `run_id`, `attempt_no` or `verdict`.
- **Arms:**
  - **referee:** the current kernel plus `instruction_given`.
  - **model_alone:** "make it faster" plus `time_us_median` only, with no referee message or instruction.
  - **random_search:** `search.py`. Stub it until P2 delivers.
- **One shared `--budget`** (number of speedcheck calls) for `agent.py` and `search.py`. Random search has only
  62 valid tile settings, so without a fixed budget it simply tries them all in about 45 minutes [est.].
- Keep `--think` off. Keep prompts to the kernel plus one line, under about 1.5k tokens. Long prompts and
  thinking mode were measured to wreck results in project 2.

### Cost of one evaluation [V count, est. time]

`check(heldout=True)` is the default. It compiles 2 dev + 6 held-out + 2 baseline = **about 10 kernels** at
2–2.6 s each. STATUS measured about 32 s per check with 6 compiles; expect about 40–45 s now [est.]. The timing
itself takes under a second.

| Change | Saves | Effort |
|---|---|---|
| Loops call `heldout=False`; held-out runs once on each arm's best kernel | ~4–6 compiles per check | 15 min |
| Cache the baseline compile per shape (`speedcheck.py:284-288` recompiles it on every call) | ~5 s per check [est.] | 30 min |
| Compile all shapes in parallel before running on the chip | several-fold on compiles [est.; compiler may not be safe to run in parallel] | 45 min, optional |

### Scope [est.]

15:00–18:30 is 210 minutes. A model attempt is about 60–120 s of generation plus 15–45 s of referee, so about
2 minutes. That is roughly 100 attempts per arm per seat.

**Plan for:** matmul, 3 repeats of about 25–30 evaluations per arm, one seat per arm.
**Stretch:** RMSNorm.
**Cut:** SwiGLU (level 12 has no kernel; leave the registration in place).

### Cores (corrects older docs) [V: STATUS.md, timing.py]

- vLLM holds **cores 0–1** on the seat pods, not 2–3. Kernels are timed on core 2, with core 3 as fallback.
- Measured interference from vLLM is **0.0%**, so TEAM.md's rule "time only while vLLM is idle" is out of date.
- **At most 2 referee processes per seat.** A third fails in `_pick_core` (see P1 #3).

---

## P1: `referee-timing` (`speedcheck.py`, `timing.py`)

1. **Make `check()` match the contract** [V]. The real signature is
   `check(path, op="matmul", baseline=None, heldout=True, rounds=3, verbose=False)` (`speedcheck.py:206`).
   TEAM.md:89 and P3's `run.py` both use `shapes="dev"|"heldout"`. Either add a `shapes=` alias or update TEAM.md
   to the real signature and tell P3. *10 min.*
2. **Return `no_gain`** [V]. At `speedcheck.py:297`, `verdict = "faster" if speedup >= threshold else "slower"`.
   The threshold is at least 1.05, so a correct 1.03× kernel is logged as `slower`. Use:
   `faster` if `s >= thr`, `slower` if `s <= 1/thr`, otherwise `no_gain`. Call `timing.noise_threshold`
   instead of repeating it inline. *5 min.*
3. **Don't score infrastructure failures as wrong kernels** [V]. `check_isolated` turns timeouts and crashes into
   `verdict="wrong"` (`:319-324`), and the chip stage catches every exception (`:262-265`), including
   "no free NeuronCore" from `timing.py`. Raise on infrastructure errors and retry in the loop, or prefix the
   message with `INFRA:` so those lines are excluded. *20 min.*
4. **Stop leaking held-out shapes** [V]. The failure message names the shape (`:263-273`), but `shapes.py:18`
   says they are never shown to the agent. The docstring (lines 12–13) also runs held-out *before* timing, the
   reverse of the README order. Run timing first, and run held-out only when there's a `faster` candidate.
5. **Byte count for converting DMAs** [V]. `simulate_count_all` (`:110-115`) counts the source side. P2 fixed
   the same bug in nkibench (commit 3163951) by counting `min(src, dst)`. Without the fix, a kernel whose final
   fp32→bf16 copy matches the expert's reads "1.29× the floor". That is over the 1.15 trigger at `:149`, so it is
   told "Same tiles reloaded", which is wrong. *10 min.*
6. **Pick the instruction from the chip result** [V]. `one_instruction` (`:141`) uses only the simulator's byte
   count, from the last simulated shape only (`:243`). STATUS finding 1 says redundant DMAs cost nothing on the
   chip. Choose the instruction from the timing (for example achieved TFLOP/s against the expert), and send the
   byte hint only when the candidate is slower. *30 min.*
7. **Smaller issues** [V]:
   - `time_us_iqr = t_cand * iqr` (`:304`) uses the worst IQR of both kernels and all shapes; use the
     candidate's own.
   - `_mismatch` (`:195`) loosens every op's tolerance to at least 5e-2.
   - Dev chip shapes never get hostile values.
   - The counter's `unmeasured` value is never shown; put it in the message.
   - `rounds` defaults to 3 here but 5 in `time_ab`; pick one.
8. **Cut** [V]:
   - `timing.load_kernel` duplicates `nkibench.load_kernel`.
   - `timing._matmul_inputs` duplicates `speedcheck._matmul_inputs`.
   - Delete the built-in `MATMUL` spec now that `shapes.py` overrides it, or at least assert that the
     required keys are present after `OPS.update`.

## P2: `kernels-search` (`shapes.py`, `kernels/`, nkibench)

1. **Measure the expert on the chip now** [V: STATUS has no expert timing]. Run
   `speedcheck.py --op matmul --check kernels/matmul_expert.py --no-heldout`. It takes about a minute.
   - If the expert isn't clearly faster than the start kernel (under ~1.3×), there is no headroom to show, so
     change the target.
   - It is also the ceiling bar on the dashboard. Save the JSON output.
2. **Strip hints from the start kernels before prompting** [V]. The docstrings in `rmsnorm_start.py:4-13` and
   `matmul_start.py:12-13` tell the model what to fix. That contaminates the model-alone arm and adds tokens:
   matmul_start is about 640, rmsnorm_start about 840, the expert about 1,700. Ship bare copies for prompts,
   or have `agent.py` strip comments. *20 min.*
3. **The RMSNorm and copy timing shapes are too small** [est.]. 256×4096 moves 4 MiB, against a measured
   launch cost of about 17 µs, so launch overhead hides any speedup. Move 2048×4096 into `time_shapes`
   (`shapes.py:184`, `:198`). *1 line each.*
4. **Score weighting** [V]. The score sums both timing shapes (`speedcheck.py:286-295`), so gate/up dominates
   and q_proj barely counts. Fine if intended; say so in the note.
5. **Cut** [V]:
   - `matmul_expert_aws.py` (141 lines) differs from the expert only in the accumulator's dtype, which P3's
     lower-precision cheat already covers. Remove it from `probe_nki.py` and the `check_kernels` plan too.
   - The matmul input builder and reference exist three times (`speedcheck.py:54-83`, `shapes.py:82-94`,
     nkibench `_args_*`). Keep `shapes.py`.
   - Fix the `matmul_expert.py` docstring (~lines 14–18): it says M=512 and blocks 4×2×8; the real values are
     M=256 and 2×2×8.
6. **Fine as is** [V]:
   - The nkibench changes don't break project 2: the tolerance for levels 1–8 is unchanged and `check_rules`
     passes all four reference kernels.
   - Simulator cost is small: only dev shapes are simulated.
   - The start kernel (62 lines, one triple loop) is a good fit for an 8B model.

## P3: `redteam-agent` (`redteam/`, loop)

1. **`run.py` crashes against the real referee** [V]. `run.py:87,90` calls `sc.check(..., shapes=...)`, which
   raises TypeError. Replace `judge_speedcheck` with a single `speedcheck.check_isolated(path, "matmul")`, one
   fresh process per cheat. Work out the stage from the record:
   - `rules`
   - `wrong` with `sim_ok` False: the simulator caught it
   - `wrong` with `sim_ok` True: the chip caught it
   - `heldout_fail`
   - otherwise: timing

   *30 min.*
2. **Wire the loop** (see "Write the loop now" above). This is the critical path.
3. **Cheats that don't test what they claim** [V by reading]:

   | Cheat | Problem | Fix |
   |---|---|---|
   | `cheat_cached` | Each check loads a fresh module and runs each shape once, so the cache never fires and the kernel is honestly correct | Drop it, or keep it as a control that should pass |
   | `cheat_unhooked_dma` | Passes: speedcheck has no byte-floor check | Port the floor check from `stage12.py:154-165` into speedcheck (with P1), or reclassify as "misleads the hint, not the time" |
   | `cheat_heldout_only` | Hard-codes nkibench level-4 shapes, so it fails as `wrong` and never reaches held-out | Use the dev and timing shapes from P2's `shapes.py` |
   | `cheat_noise`, `cheat_compile_time` | Both are copies of the reference; they count as caught only because `no_gain` is never returned | Keep one as the timer canary after P1 #2 |
   | `cheat_special_inputs` | Misnamed: it skips K tiles; it doesn't shortcut on special inputs | Rename it |

   Caught correctly: zeros, numpy, numpy_obfuscated, writes_input, and probably low_precision.
4. **Retire `stage12.py`** [V]. Its 195 lines repeat speedcheck stage 2, with different shapes (level 4, fp32)
   and a different tolerance, so its results don't carry over. Keep only the byte-floor and hazard checks, moved
   into speedcheck. *15 min.*
5. **Give the red-team output a contract** [V]. `run.py:68,101` writes `caught` as text in six forms ("yes",
   "NO", "PASS", "PENDING", "FALSE ALARM", "not run") and has no `stage` field (`where` holds the verdict).
   Proposed row: `cheat: str, expected_stage: str, caught: bool, verdict: str, stage: str, message: str,
   source: str`. Put the honest kernel in as a row whose expected result is a pass. Tell P4. *20 min.*
6. **Low priority:** the 11 cheat files are about 95% copied from `reference_level4.py`. They could become one
   table of text replacements, but separate files are easier for judges to read. Do this only after #2.

## P4: `dashboard`

1. **Red-team panel shows missed cheats as caught** [V]. The template tests `r.caught` for truthiness
   (`template.html:408,413`), and the string `"NO"` is truthy. Add an adapter in `build.py` that reads
   `redteam/redteam_results.json` until P3 #5 lands.
2. **Read the real sources instead of waiting for a `results.json` nobody writes:**
   - **Red team:** P3's file (above).
   - **Expert ceiling:** P2 #1's saved speedcheck JSON. It is a sum over both timing shapes, so any floor must
     be summed over the same shapes.
   - **Physics floor for matmul:** it is a projection, not a chip or simulator number. Label it that way or
     drop it.
3. **Held-out panel has no data source** [V]. The referee stops at the first failure and never times held-out
   shapes per shape. Change the panel to `heldout_fail` counts per arm from `attempts.jsonl`, plus the
   end-of-run held-out check on each best kernel.
4. **Page size** [est.]. Storing every prompt and response could make `index.html` several MB. Load the code
   diff lazily, or include code only. No more visual polish until real attempts exist.

---

## Merge notes [V]

- `kernels-search` already contains `referee-timing` and master's schema. **Merge `kernels-search`; skip
  `referee-timing`.**
- `referee-timing`'s `schema.py` is older than master's: it lacks `no_gain` and the `code`, `prompt` and
  `response` fields. **Master's schema wins.** The other branches and the dashboard depend on those fields.
- `redteam-agent` and `kernels-search` share no files. Only `kernels-search` changes `nkibench.py`.
- `redteam-agent` tracks a `CLAUDE.md`. Anyone with an untracked local `CLAUDE.md` must move it aside before
  merging. It also has the cores backwards (it says vLLM is on 2–3) and the stale `shapes=` API; fix it in the
  merge.
- Update TEAM.md after the merge: the referee API, the vLLM-idle rule (item 6), and the cores.
