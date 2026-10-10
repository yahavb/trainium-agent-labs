# Figures

Made with `scripts/figures.py` from the attempts logs (every number drawn is in `figures.json`, with the files
used and the runs left out). **Draft from the logs pulled by 17:50**: the v8.2 level-3 and level-4 bars hold only
the runs pulled so far (1 and 2; the full v8.2 count is 5/5 each, see SUBMISSION.md). They will be redrawn from
`analysis/logs/final/` when the final logs are in.

- `solved_by_version.png`: the share of runs that reached 1.0 on each level, for the baseline (seat-116 +
  replica, 10 runs per level), v7 (round 2), v8.2 and the final version (v8.2 on levels 1, 3, 4; v8.3 on level 2),
  labelled solved/runs. Levels 1 and 4 went from 0/10 to every run solved; level 2 from 5/10 to 8/10.
- `attempts_to_solve.png`: for every solved run of the final version, the round in which it first scored 1.0
  (round 0 is the first prompt; 4 samples per round), with the median per level. Most solves need a repair round
  or two: the checker's feedback, not the first prompt, does the work.
- `failure_modes_by_version.png`: the 8 most common failure modes (named as in `scripts/taxonomy.py`) as a share
  of each version's failed attempts, with counts. The baseline's API and tiling errors (invented names,
  partition over 128, reshape, 1-D tiles) nearly disappear in v8.2; what remains is mostly index arithmetic
  (copy-size mismatch).

A run counts once it has finished: a run that is the last in its log, not solved and short of round 7 may
still have been running when the log was pulled, so it is left out (listed in `figures.json`). The baseline's
`--all` runs stop early on repeated failures, so all of its runs count (`--complete baseline`).

Reproduce (paths as in this draft):

```bash
R=../trainium-agent-labs/runs
python scripts/figures.py analysis/figures \
  "baseline=analysis/logs/baseline/attempts.jsonl,analysis/logs/replica_seat119/attempts.jsonl" \
  "v7=$R/seat-119/Ev7_L1/Ev7_L1.jsonl,$R/seat-116/Ev7_L2/Ev7_L2.jsonl,$R/seat-117/Ev7_L3/Ev7_L3.jsonl,$R/seat-118/Ev7_L4/Ev7_L4.jsonl" \
  "v8.2=$R/seat-*/v82/v82_L*.jsonl,$R/seat-118/v82_L2/v82*_L2_s118.jsonl,$R/seat-*/v82x_partial/v82x_L*.jsonl,$R/seat-*/stopped_1713/v82x_L*.jsonl" \
  "final (v8.2 + v8.3 L2)=$R/seat-*/v82/v82_L[134]*.jsonl,$R/seat-*/v82x_partial/v82x_L1*.jsonl,$R/seat-*/stopped_1713/v82x_L1*.jsonl,$R/seat-*/v83/v83_L2*.jsonl,$R/seat-119/stopped_1713/v83_L2*.jsonl" \
  --complete baseline --note "..."
```
