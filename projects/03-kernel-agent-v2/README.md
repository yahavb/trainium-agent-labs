# Project 3 — the kernel-agent harness, v2 + dashboard

A rebuilt harness for the kernel-writing agent: same challenge (`../02-kernel-agent/CHALLENGE-kernel-agent.md`),
new machinery. Every change here exists because a specific measured wall in project 02 or in the
challenge document said so — nothing is a style preference.

> **Status: works end to end.** Verifier self-test 33/33 on the laptop and in a seat pod; the loop
> solved level 1 on round 0 of its first live run (Qwen3-8B, reward 1.0, all 40 hostile cases).
> The 10-level baseline campaign is running; the dashboard shows it live.

---

## What v2 changes, and the measurement that caused it

| wall (measured) | v2 mechanism |
|---|---|
| a rules list in the prompt makes the model audit itself in its hidden channel and return nothing (02, twice) | **the prompt carries no rules**. The verifier owns the constraints; the model generates freely and gets one named change back per failure |
| a verdict reproduced verbatim reproduces the violation ("line 16: calls banned max" → same violation) | **every failure returns an INSTRUCTION** — imperative, one named change (`errors.py`, mechanical table) |
| the model guesses instead of computing (01: guessed coefficient patterns; 02: invented a fake API name every round) | **SCRATCH** — a persistent numpy REPL the model aims itself (`SCRATCH: <expr>` lines in its reply) |
| it invents APIs it cannot check | **DOCS** — retrieval over distilled local cards (tiling patterns, numeric gotchas, an NKI API card extracted from the installed SDK) |
| long prompts cause reasoning spirals on BOTH workshop endpoints | **context.py** — every prompt part is token-accounted and auto-compacted under a budget; `--context auto` asks the server its `max_model_len`. The 8192 wall is one config point, not the design |
| same prompt + greedy decoding = same answer forever | **the prompt must change**: ledger of failed attempts, injected tool results, taxonomy-targeted experiment nudges, cycle detector (counts repeated failure *signatures*, incl. alternation) |
| confidently wrong is the worst behaviour on silently-wrong kernels | every attempt ends with `CONFIDENCE: high\|medium\|low`; the trace records claim vs verified result; **confidently-wrong is its own taxonomy label** and the dashboard counts it |
| one run is not a result | `--repeat N` built in; the dashboard leads with solve RATES |
| nothing could see a run while it happened | **dashboard/** — results at the top, tokens-by-component per attempt, taxonomy distribution, calibration, full traces; polls every 3 s while a run is live |

### Where the REPL lives, and how Qwen knows about it

The REPL is a **harness-side process** (`tools.py`, one `Scratch` per level): a persistent Python
namespace with numpy preloaded, no import/open/eval, output capped, 5 s timeout. Qwen cannot call
tools on this endpoint (`tools=` is a no-op there), so the protocol is **taught in the prompt**: the
header explains that lines like `SCRATCH: <python>` / `DOCS: <topic>` will be executed and their
results returned. The agent parses those lines, runs them, and appends the results to the next
prompt — the same measured mechanism as project 1's `COMPUTE:` calculator. Tool exchanges don't
burn an attempt. State persists across SCRATCH lines within a level, so the model can build an
experiment up piece by piece.

## Run it

In a seat pod (model server already at `localhost:8000`, env already set):

```bash
cd /workspace/projects/03-kernel-agent-v2
python3 selftest.py                                  # 33 checks, ~1 min -- run before trusting anything
python3 agent.py --level 1                           # one level, quick look
nohup python3 agent.py --all --rounds 6 --samples 2 > run.log 2>&1 < /dev/null &
tail -f run.log                                      # reconnect-safe: tail the log, not the process
python3 agent.py --all --holdout                     # include self-holdout levels 11-13
python3 agent.py --offline --level 2                 # no endpoint; exercises the loop only
```

From the laptop (RBAC allows exec but not port-forward, so the harness runs in the pod and the
dashboard reads synced traces):

```bash
projects/03-kernel-agent-v2/sync.sh up     # push the project to $SEAT (default seat-217)
projects/03-kernel-agent-v2/sync.sh run --all --rounds 6 --samples 2   # start it there, nohup'd
projects/03-kernel-agent-v2/sync.sh watch  # pull runs/ every 60 s
python3 dashboard/server.py                # -> http://localhost:8765
```

## The verifier (read this before trusting a score)

`python3 selftest.py` plants one of each bug and asserts the catch AND that the instruction names a
change: ragged-edge, whole-input `np.sum`, boolean-mask indexing, whole-array arithmetic, loopless
kernel, overflowing softmax, input mutation, one-pass variance cancellation, lost cumsum carry,
wrong signature, nondeterminism. The tolerance per level is **stated with its justification**
(`ladder.py`, `tol_why`) — e.g. level 8 rejects `E[x^2] - E[x]^2` at 1e-3 of output RMS because the
battery's "high mean" regime (mean 1e4, variance ~1) turns the one-pass formula into noise while a
two-pass or float64 kernel sits under 1e-5.

The static scan targets the WHOLE-INPUT cheat only: `np.sum(x, axis=1)` is banned, `np.sum(t,
axis=1)` on a tile slice is legal. Rejecting a correct kernel is worse than missing a cheat — the
measured rule from project 02.

## Files

```
ladder.py      10 graded levels (from 02's kernelbench.py, attributed) + 3 self-holdouts,
               per-level tolerance + justification, hostile value battery (5 regimes/shape)
verifier.py    parse -> static rules -> run -> numerics; every failure = {case, verdict,
               taxonomy, INSTRUCTION}; partial credit 0.1/0.2/0.2/0.5
errors.py      the failure taxonomy + verdict->instruction translation + exception enrichment
context.py     token budget manager with auto-compaction; compact events are traced
tools.py       SCRATCH (persistent REPL) + DOCS (BM25-lite over docs/) + reply parsing
memory.py      optimization memory distilled across runs (runs/memory.json, synced)
agent.py       the loop; JSONL traces with per-attempt tokens BY COMPONENT
docs/          rules card, tiling patterns, numeric gotchas, NKI API card (from the installed SDK)
dashboard/     stdlib server + vanilla-JS SPA (dark, validated palette)
selftest.py    the planted-bug proof (challenge: build this BEFORE the agent)
sync.sh        push/pull/watch over kubectl exec tarpipe (no port-forward in the RBAC)
```

## Scoring readiness (the honest map)

- **Ready:** verification harness with stated tolerances; hostile eval battery (prime dims, dim 1,
  ragged tiles, overflow, cancellation, ties); attempt log with token instrumentation; failure
  taxonomy; calibration tracking; repro note (this file).
- **The three judge-held-back levels** are unknown by design; the ladder's `level()` API plus our
  three self-holdouts (11 cumsum-carry, 12 variance, 13 argmax-ties — never tuned against) measure
  generalization.
- **Not built (out of scope here):** NKI reference kernels for 02's optimization levels 5–7;
  on-device latency layers. The NKI API card in `docs/` is extracted from the pod's installed SDK
  so a Stage-B port has real names to call.

## Measured so far

- Self-test: 33/33 on the laptop and in seat-217.
- Smoke (level 1, Qwen3-8B): **solved round 0**, reward 1.0, confidence high — correct on all 40
  hostile cases including the prime/ragged shapes.
- First baseline attempt died on a missing `import numpy as np` for three straight rounds — the
  harness's own fault: the NameError instruction didn't name the import. Fixed in `errors.py`; that
  fix is exactly the loop this project is about (read the failure, change the message, re-measure).
