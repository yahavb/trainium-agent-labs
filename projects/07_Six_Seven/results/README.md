# Results — what is here, and what is counted

| file | what it is |
|---|---|
| `table.md`, `graph.png` | the result: solve rate and mean best score per level × feedback A/B/C (from `veriloop/plot.py`) |
| `TAXONOMY.md` | every failed attempt from counted runs, grouped into failure modes, with examples (from `veriloop/taxonomy.py`) |
| `calibration.txt`, `calibration.csv` | the model's own confidence vs the checker's verdict, 80 designs (from `veriloop/calibrate.py`) |
| `<date>_s<seat>_summary.csv` | one row per run: level, feedback level, solved, rounds used, best score, tokens, seconds, max rounds |
| `<date>_s<seat>_attempts.jsonl` | every attempt of every run: the Verilog, its score, the feedback sent, tokens, time |
| `<date>_s<seat>-r12_*` | the same, for the 12-round level-4 runs (reported as their own setting) |

**Counted:** only the runs in the `*_summary.csv` files at this level of the folder, and only attempts that
belong to those runs. Offline test runs are never counted.

## `excluded/` — kept for honesty, not counted

| folder | why it is excluded |
|---|---|
| `excluded/pilot-seat7/` | the first level-4 runs, made with the original loop. The model sent back the identical design for 6 rounds; that finding led to the repeat note and per-attempt prompts. Different loop → not comparable. |
| `excluded/before-session-tag/` | level-4 attempts made before the session tag. Runs on different seats turned out to be byte-identical copies, so they were not independent samples. Level 4 was restarted on every seat with the fix. |

Every number in `SUBMISSION.md` comes from the counted files; the scripts above regenerate it.
