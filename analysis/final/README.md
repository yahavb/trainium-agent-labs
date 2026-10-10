# Final results (logs in analysis/logs/final, pushed 18:27 in 67bf16b)

Made with:

```bash
python scripts/final_auto.py analysis/logs/final analysis/final --l2 v83,v85 --l57 warm,v84w \
    --exclude warm_L7_s117.jsonl warm_L5c_s116.jsonl warm_L6_s118.jsonl final_L11b_s117.jsonl v83_L2d_s119.jsonl \
    v82x_L1_s117.jsonl v82x_L1_s119.jsonl v85_L2d_s117.jsonl v85_L2d_s118.jsonl
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python scripts/traffic_check.py <warm_*, v84w_*, v83_L5-7 attempts> -o analysis/final/traffic_l57
```

Left out: runs that each seat's `v82_queue.log` shows started but never finished (no `done` line): warm_L7_s117,
v83_L2d_s119, v82x_L1_s117, v82x_L1_s119, v85_L2d_s117, v85_L2d_s118. warm_L7_s118 is kept: it finished (early
stop after 4 rounds, 0.75). Also left out on the same evidence: warm_L5c_s116, warm_L6_s118, final_L11b_s117.

Groups: `final/` = v8.2 (96a9fc9) on L1, L3, L4; v8.3 (5c3aba2) + v8.5 (0eb2695) on L2; WARM (15fb0d5:
warm_*, v84w_*) on L5-L7. `v8.2_L2/`, `v8.5_L2/`, `v83/` (L5-7 without WARM), `final_all/` (one `--all` run)
and `L9-L14/` are separate. `compare.md` puts them side by side with the baseline. Token charts:
`analysis/final/<group>/token_budget.png`.

| level | final version | first 1.0 (round, from 0) | distinct trajectories | held-out (ours) / the agent's own verdict | re-audit |
|---|---|---|---|---|---|
| 1 | 7/7 | 2, 4, 2, 2, 0, 2, 7 | 7 | VERIFIED 7 / VERIFIED 7 | PASS (executor 1) |
| 2 | 13/17 (v8.3 9/11, v8.5 4/6; v8.2 alone 5/9) | 2, 1, 3, 4, 1, 2, 0, 1, 1 (v8.3); 4, 2, 2, 1 (v8.5) | 17 | VERIFIED 13, NOT SOLVED 4 / VERIFIED 13, FAILED 4 | v8.3 PASS (executor 1); v8.5: 5 distinct kernels PASS (`reaudit_v85_L2.txt`) |
| 3 | 5/5 (and 0/1 in the `--all` run) | 0, 0, 2, 0, 0 | 5 | VERIFIED 4 + passes the loop's shape only 1 / VERIFIED 5 | PASS (executor 1) |
| 4 | 5/5 | 2, 2, 2, 2, 2 | 4 | VERIFIED 5 / VERIFIED 5 | PASS (executor 1) |
| 5 | 0/6, best 0.88 in every run | - | 6 | NOT SOLVED 6 / FAILED 6 | - |
| 6 | 0/6, best 0.75 in every run | - | 6 | NOT SOLVED 6 / FAILED 6 | - |
| 7 | 0/6, best 0.75 in every run | - | 6 | NOT SOLVED 6 / FAILED 6 | - |
| 9 | 2/2 | 2, 2 | 2 | no held-out set for 9-14 / VERIFIED 2 | 2 PASS with feedback_v8's grade, trn2 (`regrade_L9_L10.txt`) |
| 10 | 1/1 | 2 | 1 | - / VERIFIED 1 | PASS (`regrade_L9_L10.txt`) |
| 11 | 0/1 (0.67) | - | 1 | - / FAILED | - |
| 12 | 0/2 (0.50, 0.30) | - | 2 | - / FAILED | - |
| 13 | 0/1 (0.50) | - | 1 | - / FAILED | - |
| 14 | 0/1 (0.50) | - | 1 | - / FAILED | - |

Where levels 5-7 stop: L5's best kernels pass 3 of 4 shapes and are correct on the fourth (K=256 M=512 N=1024)
but move too many HBM bytes for its bar; L6's pass 2 of 4, again correct but over the traffic bar on K=256 M=256
N=1024; L7's pass 2 of 4, and their best attempts mostly fail on numbers (98 of 102: NUMERICAL MISMATCH, 7.71 of
RMS on K=256 M=256 N=1024), the rest on traffic.

No level 5-7 run scored 1.0, so `traffic_l57` has no kernels to check.

Figures (analysis/figures) from the same logs:

```bash
F=analysis/logs/final; R=../trainium-agent-labs/runs   # R: the v7 round-2 logs pulled from the seats
python scripts/figures.py analysis/figures \
  "baseline=analysis/logs/baseline/attempts.jsonl,analysis/logs/replica_seat119/attempts.jsonl" \
  "v7=$R/seat-119/Ev7_L1/Ev7_L1.jsonl,$R/seat-116/Ev7_L2/Ev7_L2.jsonl,$R/seat-117/Ev7_L3/Ev7_L3.jsonl,$R/seat-118/Ev7_L4/Ev7_L4.jsonl" \
  "v8.2=$F/seat-*/v82_L*.jsonl,$F/seat-*/v82x_L*.jsonl" \
  "final version=$F/seat-*/v82_L[134]*.jsonl,$F/seat-*/v82x_L[134]*.jsonl,$F/seat-*/v83_L2*.jsonl" \
  --complete baseline --note "..."
```
