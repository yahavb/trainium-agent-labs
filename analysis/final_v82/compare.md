# Configurations compared

Divergence is measured from **v8.2**. Cell: solved/runs · round of the first 1.0 per run (- = never) · minutes per run (USAGE_LOG; n/a without it) · attempts cut off by max_tokens · distinct trajectories among the runs · held-out claims (V verified, LOOP-ONLY passes the loop's shapes only, NS not solved, U unverified) · first round whose code differs from the reference's runs.

| config | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| v8.2 | **4/4** · first 1.0 r2,2,4,0 · min 8.8 10.1 8.7 1.4 · trunc 7 · traj 4 · V4 · div - | **5/9** · first 1.0 r-,4,-,3,-,2,1,-,1 · min 8.7 4.7 10.1 4.2 7.9 2.8 1.7 7.9 1.9 · trunc 1 · traj 9 · V5 NS4 · div - | **1/1** · first 1.0 r0 · min 1.0 · trunc 0 · traj 1 · V1 · div - | **2/2** · first 1.0 r2,2 · min 9.5 11.7 · trunc 0 · traj 1 · V2 · div - |
| v7 | **0/2** · first 1.0 r-,- · min 28.9 2.0 · trunc 12 · traj 2 · NS1 · div r0 | **0/4** · first 1.0 r-,-,-,- · min 10.1 8.4 6.7 5.5 · trunc 0 · traj 4 · NS3 · div r0 | **5/5** · first 1.0 r0,0,0,0,0 · min 0.9 1.0 0.9 1.0 0.9 · trunc 0 · traj 3 · V5 · div r0 | **5/5** · first 1.0 r2,2,2,2,2 · min 3.4 3.3 3.3 3.3 3.3 · trunc 0 · traj 1 · V5 · div r0 |
| baseline | **0/10** · first 1.0 r-,-,-,-,-,-,-,-,-,- · min n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a · trunc 0 · traj 1 · - · div r0 | **5/10** · first 1.0 r0,-,0,-,0,0,-,-,0,- · min n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a · trunc 0 · traj 9 · - · div r0 | **0/10** · first 1.0 r-,-,-,-,-,-,-,-,-,- · min n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a · trunc 0 · traj 10 · - · div r0 | **0/10** · first 1.0 r-,-,-,-,-,-,-,-,-,- · min n/a n/a n/a n/a n/a n/a n/a n/a n/a n/a · trunc 0 · traj 10 · - · div r0 |

## Checks (from report.py; missing verdicts or usage logs are not listed)

- **v7**: level 1: 1 verdicts for 2 runs; 1 v7 verdicts for 2 runs
- **v7**: level 2: 3 verdicts for 4 runs; 3 v7 verdicts for 4 runs

## Inputs

- v8.2 attempts: `../trainium-agent-labs/runs/seat-117/v82/v82_L1_s117.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-118/v82/v82_L1_s118.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-117/v82/v82_L1b_s117.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-118/v82/v82_L1b_s118.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-116/v82/v82_L2_s116.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-118/v82_L2/v82_L2_s118.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-119/v82/v82_L2_s119.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-116/v82/v82_L3_s116.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-119/v82/v82_L4_s119.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-119/v82/v82_L4b_s119.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-117/v82x_partial/v82x_L2_s117.jsonl`
- v8.2 attempts: `../trainium-agent-labs/runs/seat-118/v82_L2/v82x_L2_s118.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-117/v82/v82_L1_s117_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-118/v82/v82_L1_s118_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-117/v82/v82_L1b_s117_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-118/v82/v82_L1b_s118_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-116/v82/v82_L2_s116_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-118/v82_L2/v82_L2_s118_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-119/v82/v82_L2_s119_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-116/v82/v82_L3_s116_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-119/v82/v82_L4_s119_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-119/v82/v82_L4b_s119_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-117/v82x_partial/v82x_L2_s117_verdicts.jsonl`
- v8.2 verdicts: `../trainium-agent-labs/runs/seat-118/v82_L2/v82x_L2_s118_verdicts.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-117/v82/v82_L1_s117_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-118/v82/v82_L1_s118_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-117/v82/v82_L1b_s117_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-118/v82/v82_L1b_s118_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-116/v82/v82_L2_s116_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-118/v82_L2/v82_L2_s118_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-119/v82/v82_L2_s119_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-116/v82/v82_L3_s116_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-119/v82/v82_L4_s119_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-119/v82/v82_L4b_s119_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-117/v82x_partial/v82x_L2_s117_usage.jsonl`
- v8.2 usage: `../trainium-agent-labs/runs/seat-118/v82_L2/v82x_L2_s118_usage.jsonl`
- v7 attempts: `../trainium-agent-labs/runs/seat-119/Ev7_L1/Ev7_L1.jsonl`
- v7 attempts: `../trainium-agent-labs/runs/seat-116/Ev7_L2/Ev7_L2.jsonl`
- v7 attempts: `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_L3.jsonl`
- v7 attempts: `../trainium-agent-labs/runs/seat-118/Ev7_L4/Ev7_L4.jsonl`
- v7 verdicts: `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_verdicts_L3.jsonl`
- v7 verdicts: `../trainium-agent-labs/runs/seat-119/Ev7_L1/verdicts_v7_L1.jsonl`
- v7 verdicts: `../trainium-agent-labs/runs/seat-116/Ev7_L2/verdicts_v7_L2.jsonl`
- v7 verdicts: `../trainium-agent-labs/runs/seat-118/Ev7_L4/verdicts_v7_L4.jsonl`
- v7 usage: `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_usage_L3.jsonl`
- v7 usage: `../trainium-agent-labs/runs/seat-119/Ev7_L1/usage_v7_L1.jsonl`
- v7 usage: `../trainium-agent-labs/runs/seat-116/Ev7_L2/usage_v7_L2.jsonl`
- v7 usage: `../trainium-agent-labs/runs/seat-118/Ev7_L4/usage_v7_L4.jsonl`
- baseline attempts: `analysis/logs/baseline/attempts.jsonl`
- baseline attempts: `analysis/logs/replica_seat119/attempts.jsonl`
