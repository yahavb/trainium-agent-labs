# Verifier-grounded RL trace experiment (v4)

Additive experiment for `projects/02-kernel-agent`. It reuses `agent.py` (prompts and grader) and
`nkibench.py` (verifier) **unchanged**. All new logic is in `rl_trace_agent.py`.

## Files to replace

Replace the entire contents of these three files, nothing else:
- `projects/02-kernel-agent/rl_trace_agent.py`
- `projects/02-kernel-agent/summarize_rl_trace.py`
- `projects/02-kernel-agent/README-RL-TRACE.md` (this file)

## Change log

### v4 (current)
Driven by the v3 run on level 2 (`rl_level2_v3.jsonl`): 8 rounds, best kernel reward 0.30, no kernel
ever reached the simulator.

| # | Change | Why |
|---|--------|-----|
| 1 | `sanitize_module`: strip module-level statements before grading | 4 of 8 v3 attempts failed with `No backend set`. That error means something ran at import time (a call to the kernel, `nl.*` or `nisa.*` at top level), before the simulator exists. A correct kernel with one stray call scored 0.30; with the call stripped it scores 1.00 (reproduced against a stub, see Testing). |
| 2 | `extract_kernel_code`: cut `<TRACE>` out first, ignore json/shell blocks, rank blocks by whether they contain the entry name and `@nki.jit`, accept an unclosed fence | The v3 extractor took the longest fenced block. The other 4 of 8 failed with `invalid syntax on line 1`, which is what a JSON trace or prose block produces when it is picked as "the code". |
| 3 | Prompt now states the file layout rule (only imports and the function, no top-level calls) | The model needs to be told; the sanitizer only rescues it. |
| 4 | Repair from the **best** attempt so far, not the latest | v3 repaired from whatever came last, including unparseable garbage, so one bad round poisoned the next. Now a regression is discarded and only mentioned as "do not do this". |
| 5 | Enriched feedback per failure category (`no_backend`, `parse`, `truncated`) plus a note when statements were stripped | The repo's recurring lesson: the fix is a better error message. `No backend set` alone does not say what to change. |
| 6 | New categories: `no_backend`, `truncated`, `api_name`, `tile_limits` | Contexts and feedback are keyed on these. Token-limit cut-offs are no longer mistaken for model failures. |
| 7 | `ShrunkUCB` replaces `UCBPolicy` | The v3 bandit tried every (context, action) pair once before using any value. With 8 rounds and a context that changes almost every round, that is round-robin, not learning. v4 shrinks each estimate toward its level and global averages, so a new context starts from what is already known. |
| 8 | Sampling temperature is a second learned arm (0.3 / 0.6 / 0.9) | Low for repair, higher when stuck. `agent.ask` hard-codes 0.6, so `rl_trace_agent.py` has its own `ask_model`. |
| 9 | Every sample updates the policy; `advantage = reward - group mean` is logged | With `--samples > 1` this is a GRPO-style group baseline. v3 only credited the best sample. |
| 10 | Repeat detection | Resubmitting identical code (by AST fingerprint) costs 0.10 reward and adds "change the approach" to the next prompt. |
| 11 | Trace is **off by default** (`--trace` turns it on) | All 8 v3 attempts scored `trace=1.00`, so it was a constant 0.10 of reward with no signal, it spent output tokens on a model with a small budget, and it is a plausible cause of the extraction failures. Keep it as an A/B arm. |
| 12 | `evidence` reward term removed | It paid 0.05 for failing in familiar ways. |
| 13 | `exemplar` strategy | Shows one verified kernel from a *different* level as an API-usage example. Never the current level's answer. Pool = kernels this agent verified earlier (persisted) plus, with `--seed-references`, the shipped `reference_level{n}.py` of other levels. |
| 14 | Persistent state (`--state rl_state.json`), `--episodes`, `--export-sft` | Learning accumulates across runs; repeated episodes give a learning curve; verified runs can feed a later fine-tune. |
| 15 | Parallel samples (`--samples N`) | One server round-trip's wall-clock instead of N. Grading stays serial because `agent.grade` writes one temp file. |
| 16 | `summarize_rl_trace.py` rewritten | Fixes a literal `\n` printed in headings; adds episodes, first-verified round, failure categories, temperature table, repeat, cut-off and stripped rates. |

### v3
Contextual UCB over four strategies, trace-coverage reward, NKI API card, level-2 transpose guidance.

## Run

From `projects/02-kernel-agent`:

```bash
python nkibench.py --selftest
python -m py_compile agent.py nkibench.py rl_trace_agent.py summarize_rl_trace.py
echo "$KERNEL_AGENT_BASE_URL"; echo "$KERNEL_AGENT_MODEL"
```

First run, a fresh policy and 4 parallel samples per round (this is the setting that gives a group
baseline):

```bash
rm -f rl_state.json
python rl_trace_agent.py --level 2 --rounds 8 --samples 4 --episodes 3 \
  --context 8192 --seed-references --log rl_level2_v4.jsonl
python summarize_rl_trace.py rl_level2_v4.jsonl
```

All levels, carrying what is learned from one level into the next:

```bash
python rl_trace_agent.py --all --rounds 8 --samples 4 --episodes 2 --seed-references \
  --log rl_all_v4.jsonl --export-sft rl_sft_v4.jsonl
```

If `parse` still dominates, look at the raw replies (they are in the log now):

```bash
python rl_trace_agent.py --level 2 --rounds 4 --verbose --log dbg.jsonl
python - <<'EOF'
import json
for l in open("dbg.jsonl"):
    r = json.loads(l)
    if r["failure_category"] in ("parse", "truncated"):
        print(r["round"], r["finish_reason"], repr(r["reply"][:300]))
EOF
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--samples` | 1 | Replies per round, in parallel. Use 4 or more for a group baseline. |
| `--episodes` | 1 | Independent attempts per level; policy carries over between them. |
| `--trace` | off | Ask for the structured TRACE note and include it in the reward. |
| `--seed-references` | off | Allow the `exemplar` strategy to show other levels' reference kernels. |
| `--state PATH` | `rl_state.json` | Policy counts and verified kernels. `''` disables. Delete for a from-scratch run. |
| `--export-sft PATH` | none | Write `(prompt, reply)` rows with kernel reward >= `--sft-min-reward` (default 0.999). |
| `--exploration` | 0.5 | UCB bonus weight. Rewards are in [0, 1]. |
| `--verbose` | off | Print the head of replies that produced no code. |

## How the controller works

Two bandits choose before each round, both `ShrunkUCB`:
1. **Strategy**: `direct`, `contract_trace`, `counterexample`, `repair_diagnosis` (only once there is
   a failure to repair), `exemplar` (only when an example from another level exists).
2. **Temperature**: 0.3, 0.6, 0.9.

Context is `level=N|failure=<category from the last round>`. An estimate for a context is shrunk
toward its level and then toward the global average (`prior_strength=2`), so sparse contexts borrow
statistics instead of starting blind. Optimistic prior 0.5, so untried arms still get tried.

Reward per sample, in [0, 1]:

```
R = 0.90*kernel + 0.05*hygiene + 0.05*improved - 0.10*repeated      (default)
R = 0.80*kernel + 0.10*trace   + 0.05*hygiene + 0.05*improved - 0.10*repeated      (--trace)
```
- `kernel`: the unchanged verifier reward from `agent.grade`.
- `hygiene`: 1 if the file parsed, nothing had to be stripped, and the reply was not cut off.
- `improved`: 1 if this beat the episode's best kernel reward so far.
- `repeated`: identical code (AST fingerprint) already submitted this episode.

Code that is graded is the **sanitized** code, and the sanitized code is what the next repair builds
on. The raw reply is kept in the log, and `sanitizer_dropped` lists what was removed.

Each JSONL row (`schema_version` 4) stores run/episode/round/sample, action, temperature, context,
all reward parts, advantage, failure category, verifier feedback, sanitized code, stripped
statements, finish reason, the full prompt and the reply.

## Baseline comparison

Same model, endpoint, level, context, and generation budget. Independent trials (`--episodes`, and
repeat the whole command with `rm -f rl_state.json`). Report:
1. first-attempt reward and fully verified rate;
2. attempts to first verified kernel (`summarize_rl_trace.py` prints it per episode);
3. mean verifier reward under a fixed call budget (rounds x samples);
4. rates of `bad_import`, `api_name`, `parse`, `truncated`, `no_backend`, `correctness`;
5. repeat, cut-off and stripped rates;
6. wall-clock time and call count.

Ablations worth running, one flag at a time: `--trace`; `--seed-references`; a fresh policy vs a
warm `rl_state.json`; `--samples 1` vs `4`. **Do not compare a warm-state run against a baseline
without saying so**, and keep `exemplar` out of the baseline comparison.

## Next steps, in order of expected payoff

1. **Get one verified level-2 kernel** with v4. Everything downstream needs a nonzero success rate.
2. **Rejection-sampling fine-tune (RFT).** `--export-sft` collects verified `(prompt, reply)`
   pairs. Train on those and measure first-attempt success. This is the first step that changes
   weights, and it needs only supervised training, not an RL framework.
3. **GRPO on the same data.** The log already has group-relative `advantage` for `--samples >= 4`.
   Caveat: a Neuron device cannot be shared by two processes, so the model server must be stopped
   while training runs, and the trained model re-served afterward.
4. **Mutation search before RL.** Take the best parsed kernel and sample N small edits to it; the
   verifier ranks them. Often the cheapest route from 0.5 to 1.0 on a stubborn level.
5. **Better verifier messages.** Anything that lands in `other` or `correctness` and repeats is a
   candidate for a new `enrich` rule in `agent.py`. That change has paid off repeatedly in this repo.

## Limitations

- Online policy learning over prompt strategies. It does not update model weights.
- `nkibench.py` uses the NKI simulator for functional checks. It does not establish real Trainium
  latency or downstream model speedups.
- One run is an anecdote. The bandit needs several episodes to show anything.
- `sanitize_module` removes top-level statements. A kernel that genuinely needs a top-level call
  (none in this repo's levels) would be broken by it.
- The v4 diagnosis of the v3 parse errors is inferred from the extractor's behavior and the error
  text, not from the raw replies, which v3 did not log. Check the first v4 run's `reply` field.

## Testing

Run here without Neuron hardware, using a stub `nki` package and a scripted mock model server:
- the shipped `reference_level2.py` passes `nkibench.py --check` against the stub;
- the reference kernel plus one stray top-level call scores 0.30 raw and 1.00 sanitized;
- extraction handled: JSON fence longer than the code, bare fence, unfenced TRACE, unclosed fence,
  a stray `python` line, and no fence;
- a scripted 3-round episode (wrong mapping with junk, truncated reply, correct kernel) ran end to
  end: stripping, truncation category, best-attempt anchoring with the regression note, and
  verification all behaved as described.

The stub checks the harness, not NKI semantics or your model. Treat the first real-hardware run as
the actual test.