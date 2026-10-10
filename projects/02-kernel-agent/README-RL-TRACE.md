# Verifier-grounded RL trace experiment (v4.1)

Additive experiment for `projects/02-kernel-agent`. It reuses `agent.py` (prompts and grader) and
`nkibench.py` (verifier) **unchanged**. All new logic is in `rl_trace_agent.py`.

## Files to replace

Replace the entire contents of these three files, nothing else:
- `projects/02-kernel-agent/rl_trace_agent.py`
- `projects/02-kernel-agent/summarize_rl_trace.py`
- `projects/02-kernel-agent/README-RL-TRACE.md` (this file)

## Change log

### v4.1 (current): the same LLM as its own optimizer
Driven by the v4 run on level 2: all 24 attempts parsed and passed the rules, then all 24 failed
identically with `NameError: name 'P' is not defined` at import. The sanitizer had stripped nothing,
so the undefined name sits in the model's own code evaluated at import time (a default argument,
decorator argument, annotation, or a kept top-level constant). Round after round the model returned
the same code, even at temperature 0.9 and across 4 parallel samples.

| # | Change | Why |
|---|--------|-----|
| 1 | `locate_load_error`: import the candidate the way the checker does and report the failing **line** | The checker said which name was undefined and never where. The fix in this repo has been a better error message every time. Applies to `load_error` and `no_backend`. |
| 2 | For `NameError` at import, the feedback says to define the name inside the function body | Names the change, not just the mistake. |
| 3 | `sanitize_module` also removes type annotations | Annotations are evaluated at import, so `x: nl.ndarray[P, F]` raises `NameError`. NKI does not need them. Reported in the log as a source change, not as a stripped statement. |
| 4 | New category `load_error` | Separates "file would not import" from the catch-all `other`. |
| 5 | `--mode reflect` (default): the model diagnoses its own failure | Each round after the first, a short separate call asks the same model for `CAUSE`, `CHANGE`, `RULE` about the best failing code. The `CHANGE` goes into the generation prompt. See "LLM as its own optimizer" below. |
| 6 | Lesson bank in `rl_state.json` | Each `RULE` is credited with the verifier gain of the round it was used in. Only rules with positive total gain are shown in later prompts (top 3). The verifier scores the lessons, not the model. |
| 7 | Stuck detection | If every sample repeats earlier code, or all samples are identical and failing, the next round drops the "keep everything identical" repair framing, tells the model it is repeating itself, and samples at 0.9. |
| 8 | Repeat penalty removed from the reward | In v4 it made round-0 arms look best only because round 0 cannot repeat (RL 0.37 vs 0.22 with the same kernel score). Repetition is now handled by behavior (item 7), not by reward. |
| 9 | `--mode bandit` keeps the v4 controller | For A/B comparison. |

### v4
Sanitizer for module-level calls; robust extractor; best-attempt repair anchor; `ShrunkUCB` bandit
with a temperature arm; trace off by default; `exemplar` strategy; persistent state; episodes;
SFT export; parallel samples; per-category feedback notes; rewritten summarizer.

### v3
Contextual UCB over four strategies, trace-coverage reward, NKI API card, level-2 transpose guidance.

## LLM as its own optimizer: what it can and cannot do

The idea: use the same model to improve itself, like frying the fish in its own oil. There are three
different things that phrase can mean, and they are not equally useful here.

1. **Model as critic of its own failures (implemented, `--mode reflect`).** Known as Reflexion or
   self-refine. The model reads the failing code and the checker report, writes a diagnosis and a
   general rule. Works when the model can *see* the problem once it is pointed out. It cannot add
   knowledge it lacks: a reviewer with the same blind spot reads `NameError: P` and misses it
   again, which is why item 1 above (the failing line) matters more than the reflection itself.
2. **Model as judge/reward model (not implemented, on purpose).** The verifier is ground truth. A
   same-model judge has correlated errors and can be talked into rewarding wrong kernels. Use the
   model only where the verifier is silent, never instead of it.
3. **Model trained on its own verified outputs (STaR / rejection sampling; `--export-sft`).** The
   strongest form, and the one that changes weights. It needs a nonzero success rate to start. This
   run has had zero verified kernels so far, so there is nothing to bootstrap from yet.

Self-improvement amplifies what the model can already do sometimes. If the 8B model never produces
a correct level-2 kernel, reflection will not conjure one; it pays off when success is 5 to 30
percent and the loop needs to find it faster. Measure with `--mode bandit` vs `--mode reflect`.

## Run

From `projects/02-kernel-agent`:

```bash
python -m py_compile rl_trace_agent.py summarize_rl_trace.py
rm -f rl_state.json
python rl_trace_agent.py --level 2 --rounds 8 --samples 4 --episodes 3 \
  --context 8192 --seed-references --log rl_level2_v41.jsonl
python summarize_rl_trace.py rl_level2_v41.jsonl
```

Baseline, same settings:

```bash
rm -f rl_state.json
python rl_trace_agent.py --mode bandit --level 2 --rounds 8 --samples 4 --episodes 3 \
  --context 8192 --seed-references --log rl_level2_v41_bandit.jsonl
```

To see what the model is actually producing in a failing round, the feedback printed in the console
now includes `LOCATION: ... line N: ...`. For the code itself:

```bash
python - <<'EOF'
import json
rows = [json.loads(l) for l in open("rl_level2_v41.jsonl")]
r = rows[0]
print(r["failure_category"], "\n", r["feedback"], "\n----\n", r["code"])
EOF
```

Lessons the model has written, with how much each helped:

```bash
python -c "import json; [print(c['uses'], round(c['gain'],2), r) for r,c in json.load(open('rl_state.json'))['lessons'].items()]"
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `reflect` | `reflect`: model diagnosis and lesson bank. `bandit`: UCB over fixed strategies. |
| `--samples` | 1 | Replies per round, in parallel. Use 4 or more for a group baseline. |
| `--episodes` | 1 | Independent attempts per level; state carries over between them. |
| `--trace` | off | Ask for the structured TRACE note and include it in the reward. |
| `--seed-references` | off | Show a verified or reference kernel from a *different* level as an API example. |
| `--state PATH` | `rl_state.json` | Policy counts, verified kernels, lessons. `''` disables. Delete for a from-scratch run. |
| `--export-sft PATH` | none | Write `(prompt, reply)` rows with kernel reward >= `--sft-min-reward` (0.999). |
| `--exploration` | 0.5 | UCB bonus weight (temperature arm, and strategy arm in bandit mode). |
| `--verbose` | off | Print the head of replies that produced no code. |

## How a round works

1. Choose the action: `bandit` mode picks a strategy by `ShrunkUCB`; `reflect` mode uses `direct`
   in round 0 and `reflect` afterwards. A second `ShrunkUCB` picks the temperature (0.9 when stuck).
2. In `reflect` mode: one short call asks the model for CAUSE / CHANGE / RULE about the best failing
   code (or the latest code, when stuck). Unusable output falls back to the plain repair prompt.
3. Generate `--samples` replies in parallel. Extract the kernel, sanitize it, grade it with the
   unchanged `agent.grade`, classify and enrich the feedback (with the failing line when it can).
4. Score, update the policy, credit the round's RULE with `best kernel after - best before`.
5. Repair next round from the best attempt so far, never from a regression. Stop on a verified kernel.

Reward per sample, in [0, 1]:

```
R = 0.90*kernel + 0.05*hygiene + 0.05*improved                          (default)
R = 0.80*kernel + 0.10*trace + 0.05*hygiene + 0.05*improved            (--trace)
```
`hygiene` is 1 if the file parsed, nothing was stripped, and the reply was not cut off. `improved` is
1 if the sample beat the episode's best kernel reward.

Each JSONL row (`schema_version` 5) stores run, episode, mode, round, sample, action, temperature,
context, stuck flag, the diagnosis and lesson, all reward parts, advantage, failure category,
feedback, sanitized code, stripped statements, finish reason, the full prompt and the reply.

## Baseline comparison

Same model, endpoint, level, context, generation budget. Independent trials (`--episodes`, and
repeat the command after `rm -f rl_state.json`). Report: first-attempt reward and verified rate;
rounds to first verified kernel; mean verifier reward under a fixed call budget (note that
`reflect` spends one extra short call per round); failure-category rates; repeat, cut-off,
stripped and restart rates; wall-clock time. Do not compare a warm-state run against a baseline
without saying so, and keep `--seed-references` identical across arms.

## Next steps

1. **One verified level-2 kernel.** Nothing downstream works at 0 percent.
2. **Rejection-sampling fine-tune** on `--export-sft` rows, then measure first-attempt success.
3. **GRPO** using the logged group advantages (`--samples >= 4`). A Neuron device cannot be shared by
   two processes, so the server must be stopped while training and re-served after.
4. **Mutation search**: sample N small edits of the best parsed kernel and let the verifier rank them.
5. **More located errors.** Anything repeating in `other` or `correctness` deserves its own
   "where and what to change" rule, in `rl_trace_agent.py` first and in `agent.py` if it generalizes.

## Limitations

- Prompt-level learning only. It does not update model weights.
- `nkibench.py` uses the NKI simulator. It does not establish real Trainium latency.
- One run is an anecdote.
- Lessons are model-written text. They are filtered by measured gain, but a rule can coincide with
  a gain it did not cause; treat the bank as hypotheses, and read it.
- `sanitize_module` removes top-level statements and annotations. A kernel that genuinely needs
  either would be broken by it (none in this repo's levels).
- The cause of the v4 `NameError` is inferred from the error and the empty stripped list. The
  first v4.1 log's `code` and `feedback` fields will show it directly.

## Testing

Run without Neuron hardware, using a stub `nki` package and a scripted mock model server:
- annotations that reference undefined names are fixed by the sanitizer alone;
- `P=P` in a signature reproduces the v4 `NameError`; the feedback now carries the failing line and
  the fix, and a scripted reflection plus diagnosis turns the next round into a verified kernel;
- unusable reflection output falls back, and identical outputs trigger the stuck restart;
- the lesson bank credits the rule with the measured gain and persists it;
- `--mode bandit` still runs end to end, including truncation, stripping and best-attempt anchoring.

The stub checks the harness, not NKI semantics or your model. The first real run is the real test.