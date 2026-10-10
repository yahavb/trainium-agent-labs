# Team plan (deadline 2026-10-10 21:00 EDT)

Every seat has its own chip and model server, so the benchmark is split into four slices that run
in parallel, one per seat. Only seat-130 edits the agent/harness code.

| seat | role | main task | benchmark slice on this pod |
|---|---|---|---|
| **130** | core dev | agent / harness; L8–10 harness; fix holes found by the red team; merge | `ours` L1–4 |
| **132** | experiments & analysis | watch all four pods, collect results, tables and plots, confidence calibration | `naive` L1–4 |
| **133** | red team | kernels that fool the harness (false passes) and correct kernels it rejects (false fails) | `ours` L5–7 |
| **134** | write-up & demo | one-page reproduction note, demo storyline, Stage B (NKI) comparison, deliverables checklist | `naive` L5–7 |

`ours` = our agent (verdict translated into one instruction + ledger of failed attempts + reply
gate). `naive` = ablation (the checker's report sent back verbatim). Everything else is identical:
stopping rule, holdout check, scoring. Each repeat uses a different wording of the task (notes E10).

## Status and open items (updated 17:50 EDT)

**Final agent: `main`, version `f0c39f1e0f`** = v4 (organizers' signatures + their `kernelbench.py` as
the final gate) merged with v3-rewrite (PR #10). 74 tests. Freeze 18:00; submission = a PR to
`yahavb/trainium-agent-labs` from the fork `LiYou22/trainium-agent-labs`, folder
`projects/team-27-kernel-agent/`.

| result | version |
|---|---|
| `ours` 15/20 vs `naive` 4/20 on L1–4 | v2 `e22ccab872` |
| v3 3/9 vs v2 1/25 on L4–6 (pre-registered, adopted) | v3 `29fd8d90d5` |
| v4: L4 0/3, L7 0/3 | v4 `174153fd32` |
| L8 5/5, L9 4/5 under our checker (old L8 signature) | v3-rewrite |
| L8 1/3, passing the organizers' checker | final `f0c39f1e0f` |
| final kernel per level, both checkers | `FINAL.md` |

**Running (Erfu, 17:37)**: version `c5e98d5a75` on seats 134/245/246, L1–L10, 5 runs. Note: at 17:38
`scripts/sync.sh` from seat-130 overwrote `/workspace/stageA` on those pods with `main`; the running
processes are unaffected (all code is imported at start), but re-sync your own version before any
restart, and push `c5e98d5a75` to a branch so its results can be reproduced.

**Why v4** (notes E29): under the organizers' `kernelbench.py`, L4 failed 0/32 because their signature
is `kernel(x, eps=1e-6)` (ours had `g`), L8 likewise (`kernel(x, eps=1e-5)`), and L7/L10 kernels failed
on float32 products. v4 uses their signatures and adds their checker as the final gate (loaded from
`/workspace/projects/02-kernel-agent/kernelbench.py` on the pod; the log of a verified kernel says
`organizers' checker: n/n cases`). v3 was adopted under the pre-registered rule (3/9 vs v2's 1/25, E30).

```bash
git fetch origin && git switch v4 && git pull
export POD=seat-134            # or 245 / 246
scripts/sync.sh
kubectl exec $POD -c app -- bash -c 'cd /workspace/stageA && python3 -m pytest -q tests 2>&1 | tail -1'   # 65 passed
kubectl exec $POD -c app -- bash -c 'cd /workspace/stageA && python3 -c "from kagent import judge; print(judge.available())"'   # True
REPEAT=3 scripts/launch.sh ours 8 9 10
kubectl exec $POD -c app -- bash -c 'head -2 /workspace/stageA/runs/ours-L8910-$POD.log'   # must show v174153fd32
git switch main
```

When done: branch from `origin/main`, `scripts/fetch.sh`, commit `results/$POD/...`, open a PR.

**Deliverables** (from the challenge):

| # | deliverable | status | owner |
|---|---|---|---|
| 1 | the agent | done (`kagent/`) | 130 |
| 2 | verification harness + tolerance reasoning | done (proven float32 bounds; `kagent/harness.py`, notes) | 130 |
| 3 | eval set, incl. hostile values | done: `EVALSET.md` (207 dev + 156 holdout cases) | 130 |
| 4 | failure taxonomy with counts | done: `TAXONOMY.md` (re-run `scripts/taxonomy.py` as results arrive) | 130, 132 to review |
| 5 | token instrumentation | done: real-run plot `notes/fig-tokens-run4.png` (max 706 tokens / attempt) | 130 |
| 6 | one-page reproduction note | done: `README.md` (local commands verified) | 130, 134 to review |
| – | demo script (failure + recovery, naive vs ours, efficiency) | not started | 134, with 130 |

**Open work, in order**

- **A. Deliverables: done** (eval set, real token plot, taxonomy, reproduction note).
- **B. v3 agent: ready on branch `v3`** (code version `ea9befddac`, 61 tests pass; notes E23):
  structural directives ask for a loop rewrite (E17); one remedy per directive (E13); crash-specific
  directives (E21); 12 attempts; 120 s holdout timeout. **`main` stays on v2 (`e22ccab872`)** so the
  running v2 arms stay comparable. To run it on a free pod (planned: `ours` on L4–L6):
  `git switch v3 && scripts/sync.sh && scripts/launch.sh ours 4 5 6`, then `git switch main`. Its
  logs carry version `ea9befddac`, which is how the analysis tells it apart from v2.
- **C. Final analysis and demo (from ~16:00, when the v2 slices finish)**:
  `scripts/analyze.py ... --recheck` for the final tables and calibration (slow vs wrong);
  `scripts/opt_report.py` for the efficiency figure; demo script.

**Headline results so far** (details in `notes/notes.tex`, E18–E21):
- `ours` L1–4: 15 of 20 runs verified; L4 is the wall (1/5).
- `naive` run 1 is stuck at L1, where `ours` needs 1.6 attempts: the verbatim verdict makes the
  model repeat the violation (the challenge's prediction).
- Efficiency stage: 8 of 15 verified model kernels are per-element loops; optimiser v2 accepted
  5 of 11 rewrites, up to 79x fewer instructions, every kernel still verified (`notes/fig-opt2.png`).

## Checkpoints

| time (EDT) | checkpoint |
|---|---|
| 14:00 | all four pods running their slice |
| 16:00 | first L1–7 results table (132) |
| 16:30 | L8–10 harness done (130); every pod adds its L8–10 slice |
| 18:30 | all results aggregated (132) |
| 19:00 | code freeze |
| 20:00 | write-up draft (134) |
| 20:45 | final submission |

## Everyone, first: get your pod running (~10 min)

On your laptop:

```bash
git clone git@github.com:LiYou22/kernel-agent.git && cd kernel-agent
# paste the latest AWS credentials from the channel (macOS/Linux block), then:
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
export POD=seat-13X                      # your seat

# 1. is the model server up? expect 200
kubectl exec $POD -c app -- bash -c 'curl -s -m 5 localhost:8000/health -o /dev/null -w "%{http_code}\n"'
#    not 200: kubectl exec -it $POD -- bash, then cd /workspace && ./serve.sh, wait for READY (~4 min)

# 2. push the code and self-test (all should pass)
scripts/sync.sh
kubectl exec $POD -c app -- bash -c 'cd /workspace/stageA && python3 -m pytest -q tests 2>&1 | tail -1'

# 3. start your slice (background; survives a dropped connection)
scripts/launch.sh ours 1 2 3 4      # seat-130
scripts/launch.sh naive 1 2 3 4     # seat-132
scripts/launch.sh ours 5 6 7        # seat-133
scripts/launch.sh naive 5 6 7       # seat-134

# 4. watch progress
kubectl exec $POD -c app -- bash -c 'tail -n 20 /workspace/stageA/runs/*.log'

# 5. when done, fetch results and open a PR (results/ is tracked in git)
git switch -c results-$POD
scripts/fetch.sh && git add results/$POD && git commit -m "results: $POD"
git push -u origin results-$POD && gh pr create --fill
```

Notes:
- **`main` is protected: every change goes through a pull request** (branch, push, `gh pr create`).
  Direct pushes and force pushes to `main` are rejected. Keep PRs small: one result set or one
  red-team kernel per PR, so they merge without conflicts.
- **Never send `seed` in a request**: the Neuron build of vLLM crashes (notes E5).
- Do not run two benchmarks on one pod at once; the model server serves at most 4 requests
  concurrently and they slow each other down.
- When credentials expire (`ExpiredToken`), paste the new block; processes in the pod keep running.

## Task details

### seat-130: core dev
- L8 layernorm, L9 windowed attention, L10 1-D convolution: reference, error bound, hostile
  cases, hand-written and deliberately broken kernels, translator coverage.
- Fix holes reported by 133 (`kernels/redteam/`); add a test for each fix.
- Code freeze at 19:00.

### seat-132: experiments & analysis
- Aggregate: `uv run python scripts/analyze.py ours=results/seat-130/<file>.jsonl naive=results/seat-132/<file>.jsonl --tex notes/generated/results-L1234.tex`
  (same for L5–7 with the files from 133 and 134).
- Token plots: `uv run python scripts/plot_tokens.py results/seat-130/<file>.jsonl`.
- Calibration: share of `verified` vs `dev-only`; any `verified` kernel that is actually wrong
  (re-check with 133's red-team cases).
- Deliverables: results tables, plots, failure-class counts (first version 16:00, final 18:30),
  in `notes/generated/`.

### seat-133: red team
From the challenge: "if the harness can't catch your own deliberate bug, it will not catch the model's."
- **False passes**: write wrong kernels that pass (wrong only on particular shapes, rule
  violations the checks miss, errors that hide inside the bound, ...).
- **False fails**: write correct kernels that get rejected or flagged (unusual but legal code).
- For each finding: `kernels/redteam/l<N>_<name>.py` with a one-line comment at the top saying
  what it shows; open a PR and tell 130.
- Check locally: `uv run python -m kagent.harness <N> kernels/redteam/<file>.py -v --fixes`.

### seat-134: write-up & demo
- One-page reproduction note (environment, commands, expected output).
- Demo storyline: one failure and its recovery. Ready-made examples: L2 indexes the partial tile
  out of bounds, then recovers; L3 / E9 (the translator could not name the cause, then a new
  diagnostic could).
- Stage B comparison from `upstream-nki/` (upstream NKI agent, 5 runs: L2 5/5, L1/L3 always
  0.30 = every shape crashed, L4 always 0.62; feedback sidecar v2 solved NKI L4, see
  `upstream-nki/FEEDBACK-EXPERIMENTS.md`): a section on "on NKI the failures are API misuse, not
  wrong numerics", and what fixed L4.
- Check every deliverable: agent, harness (with tolerance rationale), eval set, failure taxonomy,
  token instrumentation, reproduction note.
- Draft 20:00, final 20:45.

## References
- Design and experiment log: `notes/notes.tex` (English). Plain-language Chinese and Korean
  versions live outside the repo, maintained by 130.
- Upstream NKI baseline and feedback sidecars: `upstream-nki/` (maintained by its author; the
  sidecars run on a seat pod against the upstream agent, not locally).
