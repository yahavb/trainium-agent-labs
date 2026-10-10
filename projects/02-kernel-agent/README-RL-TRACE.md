# Verifier-grounded RL experiment (v5)

Additive experiment for `projects/02-kernel-agent`. It reuses `agent.py` (prompts and grader) and
`nkibench.py` (verifier) **unchanged**. All new logic is in `rl_trace_agent.py`.

## Files to replace

Replace the entire contents of these files, and add the fourth. Nothing else changes.
- `projects/02-kernel-agent/rl_trace_agent.py`
- `projects/02-kernel-agent/summarize_rl_trace.py`
- `projects/02-kernel-agent/README-RL-TRACE.md` (this file)
- `projects/02-kernel-agent/test_rl_trace_agent.py` (new; runs with no Neuron SDK and no model)

## What the v4.1 trace said

`rl_level2_v4.1.jsonl`, level 2, 3 episodes x 8 rounds x 4 samples. Episode 1 never got past
`ds() missing 1 required positional argument: 'size'`; episode 2 got past it only at round 4. Reading it:

1. **The four samples in a round were the same kernel.** `adv=+0.000` on every row of episode 1 (8 of 8 rounds
   were four identical kernels), and every round at t=0.3 and t=0.6 did the same. A group of identical samples has
   no baseline, so the advantage is zero by construction: the "RL" had nothing to learn from, and
   three of the four 70-second generations were wasted. (At t=0.9 diversity appeared, and so did the
   only progress in the run, 0.30 to 0.50.)
2. **Nothing told the model what `nl.ds` is.** `ds` appears in v4.1's API card and level-2 note
   ("`nl.ds(start, 1)` slices") and nowhere in `agent.py`'s own card. The model wrote `ds((src_idx, 1))`,
   `ds=nl.ds(...)`, `ds(size=...)`, `ds(nl.ds(...))`: at least six different wrong spellings of a two-argument
   call whose signature was never given. `agent.py`, with no such hint, solves this level 4 of 5
   (STATE.md). **Inferred, not measured**: the A/B in "Run" below measures it.
3. **The self-reflection step made it worse.** The 8B reviewer has the same blind spot as the writer.
   Its `CHANGE` was `ds(nl.ds((1, 1)))` three times, then `ds=nl.ds((1, 1, 1))`, which the next
   round faithfully implemented. README v4.1 predicted this ("a reviewer with the same blind spot
   reads the error and misses it again"); the loop still routed every repair through it.
4. **The errors were filed under `correctness`.** A `TypeError` on every shape is not a wrong answer.
   The bandit context was `failure=correctness`, so the policy could not tell "my API call is
   malformed" from "my index arithmetic is off".
5. **The reward had no gradient inside a failure bucket.** 0.30 (crash) scored RL 0.32 whatever the
   fix. The one round that mattered, a crash becoming a kernel that runs (0.50), moved RL from 0.32 to
   0.55 and was followed by 81.8% wrong elements with no hint about *which* ones.

## What changed in v5

The principle: **the verifier and the interpreter are more reliable than the model's opinion of its
own mistake, so everything that can be computed is computed, and the model is asked only when nothing
else is available.**

| # | Change | Why |
|---|--------|-----|
| 1 | `lint_api`: bind every `nki.*` call in the candidate against the real signature (`inspect`), and every attribute against the real module | Names the line, the call, the real signature and the fix, before any model sees the failure. Falls back to a two-entry table when the SDK is not importable. Advisory: a kernel that passes the simulator is never penalised for a finding. |
| 2 | `probe_kernel` + `located_analysis`: re-run the best failing sample once | For a raise: the failing **line**. For a wrong answer at level 2: `explain_permutation` traces every output value back to the input element it was copied from (inputs are random floats, so this is exact) and says "never moved", "dimensions the wrong way round", "never written", or gives the first wrong positions. "81.8% wrong" becomes a mapping. |
| 3 | Grounded analysis replaces model reflection when it exists | In `--mode reflect` the model is asked only if lint, location and flow are all silent, and its `CHANGE` is discarded if it breaks the API (`diagnosis_is_sane`) or repeats an earlier change. |
| 4 | `API_SIGS_CARD` (`--hints api`, default) | Signatures, not an algorithm: `nl.ds(start, size)`, slicing, `tensor_copy(dst=, src=)`. The v4.1 level-2 algorithm note is now `--hints algo`; `--hints none` keeps only the file rules. Ablate them. |
| 5 | Sample diversity: per-sample temperature spread, per-request `seed`, one-line variant hints (sample 0 keeps the canonical prompt), and a **re-ask for any duplicate** | Gives groups a baseline. `--no-variants` restores the v4.1 behaviour for comparison. |
| 6 | Grade cache keyed by code fingerprint | Identical code is simulated once. |
| 7 | Reward: `0.75*kernel + 0.10*progress + 0.05*lint_clean + 0.05*hygiene + 0.05*improved` | `progress` is the fraction of correct elements when the kernel runs (parsed from the verifier's own text), 0.3 if it runs for another reason, 0 if it raises. Verifier still dominates. |
| 8 | GRPO-style normalised advantage logged next to the raw one | `advantage_norm = (r - mean) / std`, 0 for a flat group. |
| 9 | New categories `api_signature`, `runtime_error` | Stops filing raises under `correctness`. |
| 10 | API facts in `rl_state.json` | Calls the checker rejected are stored with their real signature and shown in later prompts, across episodes. Deterministic, so no credit assignment is needed. Rejected lines are also listed as "do not write these again". |
| 11 | Stuck = identical failure 3 rounds running, not only identical code | v4.1's streak of 8 identical errors with changing code never counted as stuck. |
| 12 | `--patience 5` | Ends an episode after 5 rounds without a better kernel. v4.1 episode 1 spent 8 rounds, about 10 minutes, on one error. |
| 13 | `--mode plain` | `agent.py`'s exact prompts through this harness, same logs, for an honest A/B. |
| 14 | `--export-groups` | One row per sample group with normalised advantages, the input GRPO needs. |
| 15 | The located analysis runs every shape, not just the first failure | Reports which shapes pass and fail. A kernel that passes only the square shape (F1 == F2) is told that F1 and F2 are interchanged; an out-of-bound index on an axis of length F1 or F2 is explained as a wrong loop bound, and the verifier's generic "tile limits are a maximum" advice is dropped for it. Found in the first real v5 run: the model sat at 1 of 4 shapes for 2 rounds with the analysis saying only where it raised. |

Unchanged from v4.1: sanitizer, extractor, `ShrunkUCB`, lesson bank (now only for model-written
rules), exemplars, SFT export, episodes.

## Run

From `projects/02-kernel-agent`:

```bash
python -m py_compile rl_trace_agent.py summarize_rl_trace.py
python test_rl_trace_agent.py            # 26 tests, no SDK or model needed
```

The A/B that tells you whether any of this helps, same endpoint, same budget, 5 episodes each:

```bash
for arm in "plain --hints none" "reflect --hints none" "reflect --hints api" "bandit --hints api"; do
  rm -f rl_state.json
  python rl_trace_agent.py --mode $arm --level 2 --rounds 8 --samples 4 --episodes 5 \
    --context 8192 --seed-references --export-groups groups_level2.jsonl \
    --log "rl_level2_v5_${arm// /_}.jsonl"
done
python summarize_rl_trace.py rl_level2_v5_*.jsonl
```

Read the "Arm comparison" table last. `plain` is the number to beat (STATE.md: 4 of 5). `reflect/none`
vs `reflect/api` isolates the signature card; add `--hints algo` as a fifth arm only to see what the
algorithm note is worth, since it is close to giving the answer.

Useful while it runs: `Learning signal` in the summary should show far fewer rounds where every
sample was the same kernel than the 8 of 8 in v4.1 episode 1. If it does not, the endpoint is
ignoring temperature and seed, and the variant hints plus duplicate re-asks are doing all the work.

### Flags (new or changed)

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `reflect` | `reflect`: grounded analysis, model reflection as a fallback. `bandit`: UCB over strategies. `plain`: agent.py's prompts. |
| `--hints` | `api` | `none`, `api` (signatures), `algo` (level-2 algorithm note). |
| `--terse` | 0 | First-prompt length, as in agent.py. v4.1 silently used 1; 0 includes `agent.API_CARD` and its worked `copy_kernel` example. |
| `--no-variants` | off | Same prompt for every sample, no duplicate re-asks (the v4.1 behaviour). |
| `--no-seed` | off | Send no per-request `seed` (v5 sends one; some servers reject or crash on it). |
| `--same-temp` | off | One temperature for all samples in a round (v5 spreads them). |
| `--no-flow` | off | Level 2: do not tell the model where its output values came from. |
| `--patience` | 5 | Rounds without improvement before an episode ends; 0 disables. |
| `--export-groups PATH` | none | Group rows with normalised advantages. |

Unchanged: `--samples`, `--episodes`, `--rounds`, `--trace`, `--seed-references`, `--state`,
`--export-sft`, `--sft-min-reward`, `--exploration`, `--verbose`, `--context`, `--max-tokens`.

## How a round works

1. Choose the action (`reflect`: `direct` in round 0, then `reflect`; `bandit`: UCB; `plain`: none).
   A second `ShrunkUCB` picks the temperature arm, which is now the **centre** of a per-sample spread.
2. Build the guidance for the best failing code: the stored grounded analysis (lint + located raise or
   flow check). Only if there is none, one short model reflection, gated.
3. Send `--samples` prompts (sample 0 canonical, others with a variant line), re-ask any duplicate.
4. Extract, sanitize, grade with the unchanged `agent.grade` (cached by fingerprint), classify, lint.
5. Score, normalise advantages, update the policy and API facts, write the log and group rows.
6. Re-run the top failing sample once for its located analysis; repair from the best attempt next round.

Each JSONL row (`schema_version` 6) adds `progress`, `lint`, `lint_clean`, `advantage_norm`,
`group_distinct`, `resampled`, `grade_cached`, `diagnosis_source`, `analysis`, `hints`, `low_diversity`,
and the per-sample `temperature` (the arm is `temp_arm`).

## Limitations

- **Not run against a model or the real simulator.** The tests use a NumPy stand-in that enforces the
  signatures the v4.1 verifier printed (`ds(start, size)`, `tensor_copy(dst, src, engine, name)`).
  They check the harness. Whether the 8B model then solves level 2 is the first real result, and one
  run is an anecdote.
- Lint compares against whatever `inspect.signature` reports. If an NKI function is a generic wrapper
  (`*args, **kwargs`), nothing is flagged; that is a missed finding, not a false one. The summary warns
  if a verified kernel ever carried a finding.
- The flow check reveals where each output element came from. That is directional, in the spirit of
  the heat-rod project's finding that revealing the answer stops the model thinking, but it is level 2
  only and `--no-flow` turns it off.
- The `api` card gives the shape of `nl.ds` and slicing. That moves the prompt toward the answer;
  report which `--hints` tier a result used.
- Still prompt-level learning: no model weights change. `nkibench.py` uses the simulator, so this says
  nothing about real Trainium latency.
- The level-2 flow check is the only level-specific diagnoser. Levels 1, 3 and 4 get the lint and the
  located raise, not a flow check.

## Next steps

1. **One verified level-2 kernel in each arm** and a solve rate per arm. Nothing downstream works at 0.
2. **GRPO** on `--export-groups` rows, `--samples >= 4`. Needs the Neuron server stopped while training.
3. **Rejection-sampling fine-tune** on `--export-sft` rows once success is nonzero.
4. **Mutation search** from the best parsed kernel: N single-line edits, ranked by the verifier. The
   lint and flow check make good mutation targets.
5. **Flow checks for levels 1, 3 and 4**: the same "trace each output back to its source" idea works for
   any pure data-movement kernel; matmul needs a different diagnoser (per-tile error map).