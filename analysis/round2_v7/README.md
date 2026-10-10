# Round 2: feedback_v7 on levels 3 and 4 (draft numbers for SUBMISSION §5-§7)

Made with `scripts/report.py` from the logs executor 1 pulled at 14:35 (seat-117, level 3) and 14:46
(seat-118, level 4). Each level: `feedback_v7.py --level N --rounds 8 --samples 4 --context 8192 --repeat 5`
with V7.md's configuration (trn2 set). All four kinds of file present for both levels, 5 runs each, every
attempt matched to the server's token counts (checks.md). These are round-2 numbers. The final run replaces them.

| file | what |
|---|---|
| `summary.md` / `.csv` | per level: solved, scores, attempts to first 1.0, tokens per run, both verdicts, baseline column |
| `checks.md` | the files used and the completeness checks |
| `taxonomy.md` / `.csv` | failure modes of every failed attempt |
| `token_budget.png` / `.csv` | tokens per round by segment, server-counted (USAGE_LOG) |

## Results, and how much they are worth

| level | solved | attempts to first 1.0 | distinct trajectories | re-audit (fresh process, trn2) | held-out |
|---|---|---|---|---|---|
| 3 | **5/5** (baseline 0/5) | 1, 1, 1, 1, 2 (all round 0) | 3 distinct round-0 code sets, 4 distinct whole runs | 2 distinct solving kernels, both PASS | 5/5 VERIFIED |
| 4 | **5/5** (baseline 0/5) | 9 in every run (round 2) | **1**: the five runs are identical, code for code | 1 solving kernel, PASS | 5/5 VERIFIED |

"Distinct trajectories" counts runs whose code differs. Level 4's five runs are identical round by round:
the same prompt lengths (1,894, 2,993, 4,103 characters, 20 requests each in USAGE_LOG), the same code, the
same feedback. Even with `SAMPLING=qwen` (temperature 0.7), they are one trajectory repeated, so level 4's 5/5 is
**evidence of one path, not five**. Level 3's 5/5 rests on four different runs.

## The three sentences

1. **Where the tokens go.** A first-round prompt averages 1,148 tokens: 77% the API card and worked example,
   14% instructions, 10% the NumPy reference. A repair prompt averages 744: 47% the checker's feedback, 44% the
   previous kernel, 9% instructions. Counts are the server's own (USAGE_LOG; 80 of 80 attempts matched; levels
   3 and 4; 40 first-round attempts, 40 repair attempts, all of the repairs on level 4).
2. **Which verdict is better calibrated.** Scored against our held-out set, v7's verdict did better on level 3
   (Brier 0.067 against our 0.137) and the two tied on level 4 (0.014 against 0.010). All ten kernels passed
   held-out, and neither verdict was confident and wrong (0 of 10 each). The gap on level 3 is our
   heuristic's own doing: it docks every level-3 kernel for having been tested on a single shape (x0.7), so it
   said 0.63 for all five although all five were right.
3. **What still fails under v7, against the baseline.** Level 3's walls in the baseline (reshape 48, copy size 26,
   1-D tile 22, out of bounds 16, over 112 failed attempts) are gone; its 7 remaining failures are all a tile
   in the wrong memory (`wrong_buffer`), and every run solved in round 0. Level 4 still meets the baseline's
   wall first: partition over 128, 20 attempts in round 0 (baseline: 55, never passed). Then a broadcast shape
   mismatch, 20 in round 1 (baseline: 9). Now it gets through both and solves in round 2 every time.

## Reproduce

```bash
python scripts/report.py analysis/round2_v7 runs/seat-117/Ev7_L3 runs/seat-118/Ev7_L4 \
    --baseline runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python scripts/reaudit.py runs/seat-117/Ev7_L3/Ev7_L3.jsonl runs/seat-118/Ev7_L4/Ev7_L4.jsonl
```

The logs themselves are in `runs/` (pulled from the seats). They will be copied into `analysis/logs/` with the final run.
