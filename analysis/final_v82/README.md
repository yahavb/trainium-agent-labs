# v8.2 (96a9fc9): results by level

**Cut-off: logs pulled by 17:06** (`runs/seat-116/v82`, `117/v82`, `118/v82`, `119/v82`, `118/v82_L2`, and the finished level-2 run in `117/v82x_partial`; `v82x_*` are extra runs of the same 96a9fc9). v8.2 is `feedback_v8.py` at
96a9fc9 with V7.md's v7 settings plus `SKELETON=0 L1FIX=1 TRUNCFIX=1 MIXSAMP=1 L2HINT=1 MIX=1` (every run log's
header says so). Only finished runs are counted. Left out: `v82_L1c_s117` and `v82_L4c_s119`, still running at
the cut-off, the extra level-1 runs `v82x_L1_s117` and `v82x_L1_s119` (still running at 16:57), and
`seat-117/v82_L1_partial`, an early copy of files already in `seat-117/v82`. 116's `final_all` had not been pulled.

Made with `scripts/report.py` (checks, summary, taxonomy, token chart; every attempt matched to the server's
token counts) and `scripts/compare.py` (against v7 round 2 and the baseline + replica). All 11 distinct
solving kernels re-audited in a fresh process with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`: 11 PASS.

| level | solved | scores | first 1.0 (round, from 0) | distinct trajectories | distinct solving kernels | held-out (ours / v7's verdict) | minutes per run | tokens per run, prompt + answer (server counts) | cut off | full trn2 build + birsim |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **4/4** | 1 1 1 1 | 2, 2, 4, 0 | 4 | 4 | VERIFIED 4 / VERIFIED 4 | 8.8, 10.1, 8.7, 1.4 | 13.6k+10.9k, 13.7k+11.0k, 21.9k+9.7k, 4.9k+1.8k | 7 attempts, about 19 min in total | **5 of 5 L1 kernels match** (the 4 here + 1b0b793f), shapes (4,8,8)/2 and (8,12,12)/3, worst error 1.9e-07 of RMS; `runs/seat-117/l1_compile/compile.txt` |
| 2 | **5/9** | .5 1 .5 1 .5 1 1 .5 1 | -, 4, -, 3, -, 2, 1, -, 1 | 9 | 4 | VERIFIED 5, NOT SOLVED 4 / VERIFIED 5, FAILED 4 | 8.7, 4.7, 10.1, 4.2, 7.9, 2.8, 1.7, 7.9, 1.9 | 8.1k-31.5k prompt, 2.2k-11.9k answer per run | 1 | pending (executor 1) |
| 3 | **1/1** | 1 | 0 | 1 | 2 (two samples of round 0) | VERIFIED 1 / VERIFIED 1 | 1.0 | 4.6k+1.3k | 0 | - |
| 4 | **2/2** | 1 1 | 2, 2 | **1** | 1 (`8339bbd6f2`, the same kernel v7 found) | VERIFIED 2 / VERIFIED 2 | 9.5, 11.7 | 13.1k+4.0k, 13.1k+4.0k | 0 | - |

Baseline (seat-116 + replica, 10 runs each): L1 0/10, L2 5/10, L3 0/10, L4 0/10. v7 round 2: L1 0/2, L2 0/4,
L3 5/5, L4 5/5 (`compare.md`).

Calibration: every verdict pairs with the held-out result. Brier, ours / v7's: L1 0.010 / 0.010, L2 0.006 /
0.005, L3 0.137 / 0.067, L4 0.010 / 0.014. Nobody was confident and wrong.

## Three sentences for SUBMISSION §1

v8.2 solved level 1 in 4 of 4 runs, level 2 in 5 of 9, level 3 in 1 of 1 and level 4 in 2 of 2 (the baseline:
0, 5, 0 and 0 of 10), and all 11 distinct solving kernels pass our held-out set and a fresh-process re-audit
under trn2. Level 1 rests on four different trajectories and four different kernels, but level 4's two runs are
one trajectory with one kernel: one piece of evidence, not two. Three of the four level-2 failures sat at 0.50 on a
recognisable wrong transform (mostly a dropped loop variable, once a partial write; analysis/l2_deep.md). The
fourth had no kernel that ran until its last round, which returned x unchanged. Both verdicts called all four
unsolved, so the agent claimed no success it did not have.

## Notes on the check table

`checks.md` lists "console starts 6" for level 1 and "3" for level 4 against 4 and 2 runs: the console logs of
the left-out runs still running sit in the same folders. Nothing else is flagged.
