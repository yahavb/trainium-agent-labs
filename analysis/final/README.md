# Final results (logs in analysis/logs/final, pushed 18:27 in 67bf16b)

Made with:

```bash
python scripts/final_auto.py analysis/logs/final analysis/final --exclude warm_L7_s117.jsonl v83_L2d_s119.jsonl \
    v82x_L1_s117.jsonl v82x_L1_s119.jsonl v85_L2d_s117.jsonl v85_L2d_s118.jsonl
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python scripts/traffic_check.py <warm_*, v84w_*, v83_L5-7 attempts> -o analysis/final/traffic_l57
```

Left out: runs that each seat's `v82_queue.log` shows started but never finished (no `done` line): warm_L7_s117,
v83_L2d_s119, v82x_L1_s117, v82x_L1_s119, v85_L2d_s117, v85_L2d_s118. warm_L7_s118 is kept: it finished (early
stop after 4 rounds, 0.75).

Groups: `final/` = v8.2 (96a9fc9) on L1, L3, L4; v8.3 (5c3aba2) on L2; WARM (15fb0d5) on L5-L7. `v8.2_L2/`,
`v8.5_L2/` (0eb2695), `v83/` (L5-7 without WARM), `v84w/`, `final_all/` (one `--all` run) and `L9-L14/` are
separate. `compare.md` puts them side by side with the baseline.

| level | final version | first 1.0 (round) | held-out |
|---|---|---|---|
| 1 | 7/7 | 2, 4, 2, 2, 0, 2, 7 | VERIFIED 7 |
| 2 | 9/11 (v8.3; v8.5 alone: 4/6, v8.2: 5/9) | 2, 1, 3, 4, 1, 2, 0, 1, 1 | VERIFIED 9, NOT SOLVED 2 |
| 3 | 5/5 (and 0/1 in the `--all` run) | 0, 0, 2, 0, 0 | VERIFIED 4, passes the loop's shape only 1 |
| 4 | 5/5 | 2, 2, 2, 2, 2 | VERIFIED 5 |
| 5 | 0/7, best 0.88 every run | - | NOT SOLVED 6 (one run has no verdict) |
| 6 | 0/7, best 0.75 every run | - | NOT SOLVED 6 (one run has no verdict) |
| 7 | 0/4, best 0.75 every run | - | NOT SOLVED 4 |
| 9 | 2/2 | 2, 2 | UNVERIFIED (no held-out set for 9-14) |
| 10 | 1/1 | 2 | UNVERIFIED |
| 11 | 0/2 (0.67, 0.30) | - | - |
| 12 | 0/2 (0.50, 0.30) | - | - |
| 13 | 0/1 (0.50) | - | - |
| 14 | 0/1 (0.50) | - | - |

No level 5-7 run scored 1.0, so `traffic_l57` has no kernels to check.
