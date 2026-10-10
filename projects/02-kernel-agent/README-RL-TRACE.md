# Verifier-grounded RL experiment (v6)

Additive experiment for `projects/02-kernel-agent`. It reuses `agent.py` (prompts and grader) and
`nkibench.py` (verifier) **unchanged**. All new logic is in the files below.

## Files (replace whole files; nothing else changes)

| File | Status | What it is |
|------|--------|-----------|
| `rl_trace_agent.py` | replaced (v5 -> v6) | the controller |
| `summarize_rl_trace.py` | replaced | per-sample pass@k with intervals, arm table, confound warnings |
| `test_rl_trace_agent.py` | replaced | 64 offline tests, no Neuron SDK and no model needed |
| `README-RL-TRACE.md` | replaced | this file |
| `kernel_lint.py` | **new** | static check for hard-coded tile sizes |
| `holdout.py` | **new** | runs a kernel on shapes the checker never showed it |
| `run_matrix.py` | **new** | arms x levels, interleaved, resumable, with one summary |

## What the v5 A/B said, and what it did not

`plain --hints none` vs `reflect --hints api`, level 2, 5 episodes each, `--no-seed --same-temp`.

**The headline was weaker than it looked.**

- Reflect verified 5/5 and plain 3/5, but **every reflect episode verified in round 0 and stopped**, so
  the repair loop, the restart logic and the lint-in-repair path were never exercised in that arm.
- Per sample in round 0 (no feedback, so independent draws): plain 3/20 = 15%, reflect 7/20 = 35%.
  Wilson 95% intervals are [0.05, 0.36] and [0.18, 0.57]; Fisher exact p = 0.27, and samples inside an
  episode share a prompt, so even that is optimistic. Suggestive, not shown.
- **The arms differed in more than `--mode` and `--hints`.** `plain` is `agent.py`'s prompt only.
  `reflect` also adds the rules card, per-sample variant hints, duplicate re-asks, lint, the probe and the
  located analysis. A gap between them cannot be credited to the API card. v6 lets you ablate each piece
  (`--no-variants`, `--no-analysis`, `--hints`) and the summary prints what every arm contained.
- `solved_attempts=14/88` counted cached grades (58/88 rows) and byte-identical samples (48/88 rows) as
  separate wins.

**What the plain arm showed about failure.**

| Observation | Cause | v6 |
|---|---|---|
| 58% `tile_limits`, nearly all on shape (32, 12): `got src=384, dst=512 / 16384 / 128`, or "index range [0, 127] exceed dimension size of 32" | the kernel allocates or loops to a fixed size instead of the tensor's own | `kernel_lint` + shape rule + a rewritten message (below) |
| Episodes 2 and 4: 5 rounds of the identical kernel, flagged `STUCK-RESTART`, each re-sent the same prompt | in `--mode plain`, `build_prompt` returned `agent.py`'s repair prompt and **ignored `stuck`**. The "restart" restarted nothing. | plain now restarts with a fresh first prompt (`--no-plain-restart` restores v5) |
| The repair prompt contains the failing code, so the model copies it; temperature 0.9 did not help | low-entropy copying; diversity has to come from the prompt, not the sampler | restart = fresh prompt + grounded analysis + one hint tier up |
| Rounds 2 to 5 of those episodes: zero new information | `--patience 5`, and stuck needed 3 identical failures | stuck = 2 identical failures; episode ends after `--max-restarts 2` unchanged restarts |
| Group statistics: 4 identical samples counted as 4 observations | `group_advantages` ran over all rows | advantages over **distinct** kernels; duplicates flagged, not exported, not used to update the bandit |
| `trace_reward 0.0` everywhere | `--trace` was off, the field is a constant 0 | summarizer reports it only for `--trace` rows |
| "static API finding: 0/74" | not a bug: the failures were wrong **sizes**, and every call had a legal signature. `lint_api` cannot see a bad literal. | `kernel_lint` asks where each size came from |

One thing the v5 text did right and v6 keeps: the verifier and the interpreter are more reliable than the
model's opinion of its own mistake.

## What changed in v6

| # | Change | Why |
|---|--------|-----|
| 1 | `kernel_lint.lint_hardcoded_dims` | Taint analysis over the kernel: parameters and anything derived from `.shape` are derived; names bound only to constants are constants; an `nl.ndarray` dimension or a slice bound that folds to a constant >= 2 is hard-coded. "Likely" when a checker input is smaller than the literal on that axis. Attached only to failures of category `tile_limits`, never to a kernel that passed. |
| 2 | `rewrite_tile_feedback` | `agent.enrich` answered a dma size mismatch with an example that allocates a **128x512** tile, the same mistake the trace repeats. v6 replaces it with: the tile must have the shape of the data; here are this checker's input shapes. |
| 3 | `--hints shape` (new default tier) | `none < api < shape < algo`. `shape` = `api` + one paragraph: the checker runs several shapes, some smaller than a tile; read sizes from the tensor. It documents the checker, not the algorithm. The `api` tier's cards are unchanged from v5, but the rest of the harness changed, so a v6 `api` run is not directly comparable to a v5 one. |
| 4 | Probe over **all** shapes for every failing sample | The verifier reports the first failing shape. v6 adds `PER-SHAPE RESULT: (32,12): RAISES ... \| (128,64): passes \| ...` and logs `probe_cases`. |
| 5 | Dense progress from the probe | A shape that passes counts 1, one that runs but is wrong counts half its correct-element fraction, one that raises counts 0. v5 gave every crash 0.30. |
| 6 | Real restarts | `stuck` now means two identical failures (was three) or an all-identical group. A restart is a fresh prompt (every mode, including plain) carrying the grounded analysis of the last failure. |
| 7 | Hint escalation | Each restart climbs one tier (`none -> api -> shape`). It stops before `algo`, which is close to the answer, unless `--escalate-algo`. Rows log `hints_effective`, and the summary warns when it differs from the arm's label. `--no-escalate` turns it off. |
| 8 | Budget | `--patience 3` (was 5) and `--max-restarts 2`. A stuck episode now costs about 3 rounds, not 7+. |
| 9 | Deduplicated learning signal | `dedup_advantages`; bandit updated once per distinct kernel; `--export-groups` and `--export-sft` skip duplicates. |
| 10 | Held-out shapes | `holdout.py`: extra shapes per level, graded with a different input seed, never shown in feedback. Shapes the shipped reference kernel fails are excluded from the verdict. `--holdout report` (default) logs it; `require` makes failing it a failure, with the message as feedback. |
| 11 | `--independent-episodes` | Resets policy, lessons and API facts before each episode. Without it, episode k inherits episodes 0..k-1 and per-episode solve rates are not independent samples. |
| 12 | `--no-analysis` | Skips lint, probe and analysis: the model sees only the verifier's text. The clean ablation of everything computed here. |
| 13 | Summarizer | pass@1 and pass@4 from round 0 with a bootstrap interval over episodes; episodes solved with a Wilson interval; failures counted over distinct kernels; arm labels and a confound report. |
| 14 | `run_matrix.py` | The experiment as a command (below). |

Unchanged: sanitizer, extractor, `ShrunkUCB`, lesson bank, exemplars, SFT export, per-request seed, variant
hints, duplicate re-ask, the level-2 flow check, `--mode bandit`.

## Run

From `projects/02-kernel-agent`:

```bash
python -m py_compile rl_trace_agent.py summarize_rl_trace.py run_matrix.py kernel_lint.py holdout.py
python test_rl_trace_agent.py            # 64 tests, offline
```

### 1. The comparison you meant to run (level 2)

```bash
export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1 KERNEL_AGENT_MODEL=Qwen/Qwen3-8B
python run_matrix.py --levels 2 --episodes 20 \
    --arms plain/none plain/api reflect/none reflect/api reflect/shape \
    -- --context 8192 --no-seed --same-temp
```

`plain/api` is `agent.py`'s prompt through this harness, so it ignores the hint tier by design; keep it only
as a second baseline, or drop it. Isolate one factor at a time with arms that differ in one flag:

```bash
--arms reflect/api reflect/api+no-analysis reflect/api+no-variants reflect/shape reflect/shape+no-escalate
```

### 2. Does it solve the other levels?

```bash
python run_matrix.py --levels 1 2 3 4 --episodes 6 --arms reflect/shape \
    -- --context 8192 --no-seed --same-temp
python summarize_rl_trace.py matrix_runs/*.jsonl --brief
```

Read per level: `solved` with its interval, and `pass@1 r0`. "Best reward" hides how unreliable a level is.
Levels 5 to 7 are graded on HBM traffic and 8 is attention; run them after 1 to 4 are solid
(`--levels 5 6 7 8`; `holdout.py` has shapes for 8 but not for 5 to 7).

### 3. Is a "verified" kernel actually right?

```bash
python holdout.py --level 2 --check my_kernel.py      # exits 1 if it fails a held-out shape
```

### Flags (new or changed)

| Flag | Default | Meaning |
|------|---------|---------|
| `--hints` | `shape` | `none`, `api`, `shape`, `algo` (see change 3). |
| `--patience` | 3 | Rounds without a better kernel before the episode ends (0 = off). |
| `--max-restarts` | 2 | Unchanged restarts before the episode ends (0 = off). |
| `--no-escalate` | off | A restart keeps its hint tier. |
| `--escalate-algo` | off | Let escalation reach the level-2 algorithm note. |
| `--no-plain-restart` | off | v5 behaviour: plain mode re-sends the repair prompt when stuck. |
| `--no-analysis` | off | No lint, probe or located analysis. |
| `--no-probe-all` | off | Probe only the best failing sample of a round. |
| `--holdout` | `report` | `off`, `report`, `require`. |
| `--independent-episodes` | off | Reset learned state per episode. `run_matrix.py` always sets it. |

Unchanged: `--mode`, `--samples`, `--episodes`, `--rounds`, `--trace`, `--seed-references`, `--state`,
`--export-sft`, `--export-groups`, `--no-variants`, `--no-seed`, `--same-temp`, `--no-flow`, `--exploration`,
`--verbose`, `--context`, `--max-tokens`.

Each JSONL row is `schema_version` 7. New fields: `hints_effective`, `restarts`, `escalation`, `duplicate`,
`shape_lint`, `probe_cases`, `holdout`, `config`. Older logs still load; arms are then labelled from
`mode/hints` only.

## Limitations

- **Not run against a model or the real simulator.** The 64 tests use a NumPy stand-in that enforces the
  signatures and the dma element-count assertion the real verifier printed in your trace. They check the
  harness. A mock-server smoke test of the real CLI path also passed. Whether v6 raises the level-2 solve
  rate on Qwen3-8B is exactly what step 1 above measures; I have no number for it.
- `kernel_lint` is a heuristic. It cannot see a size computed through a helper function or an attribute
  chain it does not model, and it reports a literal as "likely" only against the checker's input shapes.
  It is advisory, attached only to kernels that already failed, and the summary warns if a verified kernel
  ever carried a finding. Levels 3 and 4 legitimately allocate literal 128-row tiles; their findings, if
  any, will be `tile_limits` failures only.
- `holdout.py` shapes for levels 3, 4 and 8 are my choices and are untested against the shipped reference
  kernels on the real SDK. `check_holdout` excludes any shape the reference fails, but run
  `python holdout.py --level N --check reference_levelN.py` once per level and read the output before
  trusting a verdict.
- The per-shape table and the level-2 flow check reveal where outputs came from. That is directional
  feedback, in the spirit of the heat-rod finding that revealing the answer stops the model thinking.
  `--no-flow` removes the flow check; `--no-analysis` removes everything computed.
- Escalation and the `shape` default change what the model is told. Report the tier a result used
  (`hints_effective`), and do not compare a `shape` result to a v5 `api` result as if only the harness changed.
- Re-asking a duplicate costs another generation. When the server is effectively deterministic for a prompt
  the re-ask often returns the same code again; `--no-variants` turns the re-ask off.
- Still prompt-level learning: no model weights change. `nkibench.py` uses the simulator, so this says
  nothing about real Trainium latency.

## Next steps

1. Run section 1 above at 20 episodes per arm and read `pass@1 r0` and its interval before anything else.
2. Run section 2. If a level sits near 0%, read its `PER-SHAPE RESULT` lines before adding hints: the
   failure is usually one shape.
3. GRPO on `--export-groups` rows (`--samples >= 4`; duplicates are already removed). Needs the Neuron
   server stopped while training.
4. Rejection-sampling fine-tune on `--export-sft` rows once the solve rate is nonzero.
5. Flow checks for levels 1, 3 and 4 (trace each output back to its source for data movement; a per-tile
   error map for matmul). Only level 2 has one.