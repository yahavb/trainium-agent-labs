# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 5/5 | 1.00 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 1, round 0; run 2: attempt 1, round 0; run 3: attempt 1, round 0; run 4: attempt 1, round 0; run 5: attempt 2, round 0 | 4,588+1,209; 4,588+1,322; 4,588+1,209; 4,588+1,264; 4,588+1,209 | 5 | VERIFIED 5 | 0.63 | 0.137 | 0 | VERIFIED 5 | 0.74 | 0.067 | 0 | 0/5, mean 0.30 |
| 4 | 5 | 5/5 | 1.00 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2; run 2: attempt 9, round 2; run 3: attempt 9, round 2; run 4: attempt 9, round 2; run 5: attempt 9, round 2 | 10,548+4,292; 10,548+4,292; 10,548+4,292; 10,548+4,292; 10,548+4,292 | 5 | VERIFIED 5 | 0.90 | 0.010 | 0 | VERIFIED 5 | 0.88 | 0.014 | 0 | 0/5, mean 0.60 |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.63 | 0.137 | 0 | 0.74 | 0.067 | 0 |
| 4 | 5 | 0.90 | 0.010 | 0 | 0.88 | 0.014 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 80 of 80 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_L3.jsonl` | `499182950b1e5011cf72f44911f30504` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-118/Ev7_L4/Ev7_L4.jsonl` | `aba7763b56afe7cd7f36087d76f98d45` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_verdicts_L3.jsonl` | `b44b501e3ace46aef781816861f322a1` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-118/Ev7_L4/verdicts_v7_L4.jsonl` | `2fb540a44bd88135955bf30b5c80e393` | untracked |
| baseline | `../trainium-agent-labs/runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | untracked |
| usage | `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_usage_L3.jsonl` | `247bbf75fd990c9be6e9c90354f8219b` | untracked |
| usage | `../trainium-agent-labs/runs/seat-118/Ev7_L4/usage_v7_L4.jsonl` | `99df02b3d075dea754906a3b729ad78e` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_nki_verdicts_L3.jsonl` | `5c32425946a4b5028ac21f0b8e9e0240` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-118/Ev7_L4/nki_verdicts_L4.jsonl` | `d1f91fb93f8fcae668282e325ef1401a` | untracked |
