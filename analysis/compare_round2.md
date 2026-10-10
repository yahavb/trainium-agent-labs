# Configurations compared

Divergence is measured from **v7**. Cell: solved/runs · round of the first 1.0 per run (- = never) · minutes per run (USAGE_LOG; n/a without it) · attempts cut off by max_tokens · distinct trajectories among the runs · held-out claims (V verified, LOOP-ONLY passes the loop's shapes only, NS not solved, U unverified) · first round whose code differs from the reference's runs.

| config | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| v7 | **0/2** · first 1.0 r-,- · min 28.9 2.0 · trunc 12 · traj 2 · NS1 · div - | - | **5/5** · first 1.0 r0,0,0,0,0 · min 0.9 1.0 0.9 1.0 0.9 · trunc 0 · traj 3 · V5 · div - | **5/5** · first 1.0 r2,2,2,2,2 · min 3.4 3.3 3.3 3.3 3.3 · trunc 0 · traj 1 · V5 · div - |
| baseline | **0/5** · first 1.0 r-,-,-,-,- · min n/a n/a n/a n/a n/a · trunc 0 · traj 1 · - · div r0 | **3/5** · first 1.0 r0,-,0,-,0 · min n/a n/a n/a n/a n/a · trunc 0 · traj 5 · - · div no ref runs | **0/5** · first 1.0 r-,-,-,-,- · min n/a n/a n/a n/a n/a · trunc 0 · traj 5 · - · div r0 | **0/5** · first 1.0 r-,-,-,-,- · min n/a n/a n/a n/a n/a · trunc 0 · traj 5 · - · div r0 |
| E-v3 | - | - | - | **1/4** · first 1.0 r-,5,-,- · min n/a n/a n/a n/a · trunc 0 · traj 4 · - · div r0 |
| E-A | **0/5** · first 1.0 r-,-,-,-,- · min n/a n/a n/a n/a n/a · trunc 0 · traj 1 · - · div r0 | - | - | - |

## Checks (from report.py; missing verdicts or usage logs are not listed)

- **v7**: level 1: 1 verdicts for 2 runs; 1 v7 verdicts for 2 runs
- **E-v3**: level 4: attempts hold 4 runs but the console logs started level 4 3x: probably an earlier invocation appended; no verdicts; no v7 verdicts; no usage log

## Inputs

- v7 attempts: `../trainium-agent-labs/runs/seat-119/Ev7_L1/Ev7_L1.jsonl`
- v7 attempts: `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_L3.jsonl`
- v7 attempts: `../trainium-agent-labs/runs/seat-118/Ev7_L4/Ev7_L4.jsonl`
- v7 verdicts: `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_verdicts_L3.jsonl`
- v7 verdicts: `../trainium-agent-labs/runs/seat-119/Ev7_L1/verdicts_v7_L1.jsonl`
- v7 verdicts: `../trainium-agent-labs/runs/seat-118/Ev7_L4/verdicts_v7_L4.jsonl`
- v7 usage: `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_usage_L3.jsonl`
- v7 usage: `../trainium-agent-labs/runs/seat-119/Ev7_L1/usage_v7_L1.jsonl`
- v7 usage: `../trainium-agent-labs/runs/seat-118/Ev7_L4/usage_v7_L4.jsonl`
- baseline attempts: `analysis/logs/baseline/attempts.jsonl`
- E-v3 attempts: `analysis/logs/ev3_L4/attempts_v3_L4.jsonl`
- E-A attempts: `analysis/logs/expA_L1/attempts_expA_L1.jsonl`
