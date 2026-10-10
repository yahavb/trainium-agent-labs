# Figures

Made with `scripts/figures.py` from the attempts logs (every number drawn is in `figures.json`, with the files
used and the runs left out). Drawn from the final logs (`analysis/logs/final`, pushed 18:27); the final version is v8.2's runs on levels 1, 3 and 4 and v8.3's + v8.5's on level 2. Unfinished runs are left out (see `figures.json` and `analysis/final/README.md`).

- `solved_by_version.png`: the share of runs that reached 1.0 on each level, for the baseline (seat-116 +
  replica, 10 runs per level), v7 (round 2), v8.2 and the final version (v8.2 on levels 1, 3, 4; v8.3 on level 2),
  labelled solved/runs. Levels 1, 3 and 4 went from 0/10 to every run solved (7/7, 5/5, 5/5); level 2 from 5/10 to 13/17.
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

Reproduce: see the `scripts/figures.py` command in `analysis/final/README.md`.
