# Project 2, the kernel agent: submission note (interim)

**Status: interim.** A full L1-L10 pass of the final agent (code version `c5e98d5a75`, 5 runs per
level, three pods) is running as this is written; its logs, the regenerated `FINAL.md` and the final
tables will be pushed to this branch when it finishes. Everything else below is complete.

Paths are relative to this directory. This is a snapshot of our team repository
(`github.com/LiYou22/kernel-agent`, branch `final`); its history, pull requests and review
discussion are there.

## Start here

| what | where |
|---|---|
| reproduction note (local, no model, ~2 min; and on a seat pod) | `README.md` |
| design and experiment log E1-E30, including the negative results | `notes/notes.tex`, `notes/paper.tex` |
| the agent | `kagent/agent.py` (loop), `kagent/translate.py` (verdict -> one instruction) |
| the verification harness, tolerance = a proven float32 error bound, and its reasoning | `kagent/harness.py`, `kagent/guard.py`, `kagent/static.py`; notes section "Tolerance" |
| eval set, 207 dev + 156 holdout cases incl. hostile values | `EVALSET.md` |
| failure taxonomy with counts | `TAXONOMY.md` |
| token instrumentation (real run) | `notes/fig-tokens-run4.png`, `scripts/plot_tokens.py` |
| one kernel per level that passes the organizers' `kernelbench.py` | `FINAL.md`, `kernels/final/` |
| demo script: a failure and its recovery, naive vs ours, replayable without a model | `demo/DEMO.md` |
| every run, as JSONL with code and feedback per attempt | `results/<pod>/` |
| Stage B: the upstream NKI agent, baseline and feedback sidecars (NKI L1 0/5 -> 5/5, NKI L4 0/5 -> 1/5) | `upstream-nki/` |

## What we did, in one paragraph

The rules live in the verifier, not in the prompt. The model writes freely; the harness finds what
is wrong (rule scan, runtime tile guard, numerics against a proven float32 bound, level-specific
diagnoses); the agent translates that verdict into **one** instruction. The ablation `naive` sends the
checker's report back verbatim and is otherwise identical. A dev pass is re-checked on holdout cases
the model never saw, and since v4 also by the organizers' own `kernelbench.py`; only then is a
kernel reported `verified`. What the live runs taught us, each with the kernel that taught it, is in
the notes: "change only that line" blocks fixes that need loops rebuilt (E17); a draft that breaks
five rules at once needs a rewrite in the op's legal loop shape, not five patches; a crash on a
banned line is the rule, not the crash; a repeated error that is shrinking is progress, not a loop;
and our own checker had a hole (a column slice longer than one tile), found by running our verified
kernels back through it and fixed before any result was reported.

## Results so far (5 runs per cell unless stated; `verified` = dev + holdout + organizers' checker where available)

| level | naive (v2) | ours v2 | ours v3 | ours v3-rewrite / final |
|---|---|---|---|---|
| L1 relu | 1/3 | 5/5 | | |
| L2 row sum | 0/3 | 4/5 | | |
| L3 row max | 1/3 | 5/5 | | |
| L4 rmsnorm | 0/2 | 1/5 | 1/3 | |
| L5 softmax | 0/5 | 0/5 | 0/3 | running |
| L6 transpose | 1/5 | 0/5 | 2/3 | running |
| L7 matmul | 0/5 (3 slow-but-correct) | 0/5 (1 slow-but-correct) | | running |
| L8 layernorm | 0/5 | 0/5 | 0/5 | **5/5** |
| L9 band attention | 0/5 | 0/5 | 0/5 | **4/5** |
| L10 conv1d | 4/5 | 5/5 | 4/5 | 4/5 |

Honest caveats: the server is near-deterministic at temperature 0.6, so each repeat uses a different
task wording and 5 runs carry less than 5 runs of information (E10); L6-L7 under v2 went the wrong way
for us (naive ahead); the skeleton rewrites cover the named ops L5-L9 only, so a held-back op falls
back to the generic rule fixes; and "dev-only" L7 kernels were correct but slow (E21).
