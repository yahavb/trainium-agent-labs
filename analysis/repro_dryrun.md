# Reproduction dry run: following SUBMISSION.md §1 and SETUP_PYTHON.md from a fresh clone

Done 14:27-14:30 on a Mac (Apple silicon, Docker, Homebrew Python 3.14), in a new directory, from a fresh
`git clone https://github.com/liuyq123/trainium-agent-labs.git` of team/master (9ea5407, then a0af434 after
the fix in row 0). Only the documents' own steps were used; where they did not say enough, that is recorded
as a finding, not filled in. The pod steps could not be run from here and were checked as text only.

**Result: 4 steps fail when followed literally, 3 more need a guess. The most severe: on a seat, `/workspace`
is the organizers' repository (`k8s/workshop-seats.yaml` clones yahavb/trainium-agent-labs there), and
§1 never says to get this repository onto the pod, so `--eval` and `feedback_v7.py` do not exist there.**
One more, found and already fixed: most attempt logs were not in the repository at all (row 0).

| # | step (as written) | ran it? | result | time | finding |
|---|---|---|---|---|---|
| 0 | `analysis/logs/baseline/attempts.jsonl` (the input §1's laptop commands need) | yes | **missing from a fresh clone**: `.gitignore`'s `attempts.jsonl` and `*.log` rules dropped 9 of 12 log files from 9ea5407 | - | **fixed in a0af434** (files added, `!analysis/logs/**` in `.gitignore`); a fresh clone now has all 13 files |
| 1 | SETUP_PYTHON §3: `docker run ... python:3.12-slim`, then §1's `python3.12 -m venv ~/venvs/nki` and `pip install ... nki==0.6.0 numpy==2.5.3 httpx==0.28.1` | yes | installed nki 0.6.0 | 8 s | ok (the `-it` shell was replaced by `docker exec` with the same commands) |
| 2 | SETUP_PYTHON §3: `NEURON_PLATFORM_TARGET_OVERRIDE=trn2 ~/venvs/nki/bin/python nkibench.py --selftest` | yes | `SELFTEST PASSED` | <1 s | ok |
| 3 | §1: `python nkibench.py --selftest` and the `--eval` loop, as written | yes, in the same container | **fails**: `ModuleNotFoundError: No module named 'numpy'` (`python` there is the system Python, not the venv) | - | §1 gives only the on-seat form. With `~/venvs/nki/bin/python` and the trn2 override: **20/20, 16/16, 4/4, 16/16**, exactly as §1 says (3 s) |
| 4 | V7.md "Run it" exports + `python3 feedback_v7.py --offline --all --rounds 8 --samples 4 --context 8192 --repeat 5 --log ... --verdicts ...` (venv activated) | yes | exit 0; 160 attempts, 20 agent verdicts, 20 v7 verdicts; every level 5/5 VERIFIED; Brier 0.123. `NKI_VERDICT_COMPILE=1` without a compiler degrades to "not compiled on this machine" | 16 s | ok. Offline writes no `usage_v7.jsonl` (no model calls); worth one sentence so nobody looks for it |
| 5 | §1 laptop: `python scripts/summarize.py ...` (and taxonomy, token_budget) | yes | **fails**: `python: command not found` (macOS has `python3` only) | - | with `python3`: summarize and taxonomy work with the standard library alone (baseline gives L1 0/5, L2 3/5, L3 0/5, L4 0/5) |
| 6 | §1 laptop: `python3 scripts/token_budget.py <a log with token data>` | yes, on step 4's attempts | **fails**: `ModuleNotFoundError: No module named 'matplotlib'` | - | after `python3 -m venv .venv && .venv/bin/pip install matplotlib` (4 s): 40 rounds -> .png + .csv. On the baseline it exits 1 by design ("no attempts with token accounting"): the baseline predates token logging |
| 7 | §1 laptop: the `[[TBD]]` file lists | yes, with the baseline | needs a guess today | - | `scripts/report.py` finds and classifies the files itself; one command replaces the three |
| 8 | §1 pod: `./serve.sh` | text only | on a seat the pod sets `MAX_MODEL_LEN=8192` (`k8s/workshop-seats.yaml`); **off a seat `serve.sh` defaults to 4096** while the agent budgets for 8192 | - | needs a guess off a seat; write it explicitly |
| 9 | §1 pod: `./serve.sh` then the next commands | text only | `serve.sh` keeps the shell (the server runs in the foreground), so the next lines need a second `kubectl exec` shell; and a run of `--repeat 5` takes far longer than a connection lasts | - | needs a guess: say "second shell" and `nohup` |
| 10 | §1 pod: `cd /workspace/projects/02-kernel-agent` then `--eval` / `feedback_v7.py` | text only | **fails on a fresh seat**: `/workspace` holds the organizers' clone (`git clone --depth 1 https://github.com/yahavb/trainium-agent-labs.git /workspace`), which has neither `--eval` nor `feedback_v7.py`, nor our checker | - | **most severe**: add a step that brings this repository onto the pod |

## Suggested replacement for §1 "Reproduce" (exact text)

````markdown
**Reproduce.** Everything except the model runs on a laptop; the agent itself runs on a seat pod.

On a seat pod (`kubectl exec -it seat-<N> -- bash`; wait for the `root@seat-<N>:/workspace#` prompt). `/workspace`
is the organizers' repository; put ours next to it:

```bash
git config --global --add safe.directory '*'
git clone https://github.com/liuyq123/trainium-agent-labs.git /workspace/team
cd /workspace && MAX_MODEL_LEN=8192 ./serve.sh      # the model server: about 4 minutes, keeps this shell
```

In a second shell on the same pod:

```bash
cd /workspace/team/projects/02-kernel-agent
python nkibench.py --selftest                                                        # SELFTEST PASSED
for l in 1 2 3 4; do python nkibench.py --level $l --eval reference_level$l.py | head -1; done   # 20/20 16/16 4/4 16/16
# the exports in V7.md "Run it", with per-level file names, then for each level N:
export NKI_VERDICTS=nki_verdicts_LN.jsonl USAGE_LOG=usage_LN.jsonl
nohup python3 feedback_v7.py --level N --rounds 8 --samples 4 --context 8192 --repeat 5 \
    --log attempts_LN.jsonl --verdicts verdicts_LN.jsonl > run_LN.log 2>&1 < /dev/null &
```

Without a seat, the checker and the agent's loop still run (no model: `--offline` replays the reference
kernels). Set up NKI 0.6.0 per [SETUP_PYTHON.md](SETUP_PYTHON.md) (on a Mac, its Docker step), then, with
`~/venvs/nki/bin` on your PATH and `export NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, the same `--selftest` and
`--eval` lines, and V7.md's exports with `python3 feedback_v7.py --offline --all`. Offline runs make no model
calls, so they write no `USAGE_LOG`.

The tables, from the logs (on a laptop; `python3` with matplotlib for the token chart):

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python scripts/report.py analysis/final analysis/logs/final      # checks, summary, taxonomy, token chart
.venv/bin/python scripts/report.py /tmp/baseline analysis/logs/baseline    # the baseline: L1 0/5, L2 3/5, L3 0/5, L4 0/5
```
````

Not verified here, because they need a pod: the `git clone` into `/workspace/team`, `serve.sh`, and the live run.
The `safe.directory '*'` line is what NOTES.md §3 prescribes for any git command in a pod, widened to cover
the new clone.
