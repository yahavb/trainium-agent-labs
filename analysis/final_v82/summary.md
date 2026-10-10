# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 4/4 | 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2; run 2: attempt 9, round 2; run 3: attempt 17, round 4; run 4: attempt 3, round 0 | 13,595+10,935; 13,702+11,023; 21,915+9,679; 4,856+1,793 | 3 in 2 round(s), 7.5 min; 3 in 1 round(s), 7.4 min; 1 in 1 round(s), 3.7 min; 0 | 8.8 10.1 8.7 1.4 | 4 | VERIFIED 4 | 0.90 | 0.010 | 0 | VERIFIED 4 | 0.90 | 0.010 | 0 | 0/5, mean 0.30 |
| 2 | 5 | 3/5 | 0.50 1.00 0.50 1.00 1.00 | 0.80 | 0.50 | 1.00 | run 2: attempt 19, round 4; run 4: attempt 11, round 2; run 5: attempt 5, round 1 | 31,498+10,720; 19,381+5,844; 30,640+11,932; 11,939+3,471; 8,074+2,186 | 0; 0; 1 in 1 round(s), 3.5 min; 0; 0 | 8.7 4.7 10.1 2.8 1.7 | 5 | VERIFIED 3, NOT SOLVED 2 | 0.54 | 0.006 | 0 | VERIFIED 3, FAILED 2 | 0.54 | 0.006 | 0 | 3/5, mean 0.72 |
| 3 | 1 | 1/1 | 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 1, round 0 | 4,624+1,280 | 0 | 1.0 | 1 | VERIFIED 1 | 0.63 | 0.137 | 0 | VERIFIED 1 | 0.74 | 0.067 | 0 | 0/5, mean 0.30 |
| 4 | 2 | 2/2 | 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2; run 2: attempt 9, round 2 | 13,077+3,967; 13,101+3,967 | 0; 0 | 9.5 11.7 | 2 | VERIFIED 2 | 0.90 | 0.010 | 0 | VERIFIED 2 | 0.88 | 0.014 | 0 | 0/5, mean 0.60 |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.90 | 0.010 | 0 | 0.90 | 0.010 | 0 |
| 2 | 5 | 0.54 | 0.006 | 0 | 0.54 | 0.006 | 0 |
| 3 | 1 | 0.63 | 0.137 | 0 | 0.74 | 0.067 | 0 |
| 4 | 2 | 0.90 | 0.010 | 0 | 0.88 | 0.014 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 180 of 180 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `../trainium-agent-labs/runs/seat-117/v82/v82_L1_s117.jsonl` | `1690af776147b8860029ac59162b7aca` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-118/v82/v82_L1_s118.jsonl` | `bb221a965bb664d94dc4653002fba90f` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-117/v82/v82_L1b_s117.jsonl` | `12a7c993ff1f2834bddcf8cf77fbbae5` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-118/v82/v82_L1b_s118.jsonl` | `10ddf1b9e2be6cac478096977b1fbd0b` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-116/v82/v82_L2_s116.jsonl` | `46c5bf5a7132ce5766799b7d1379298f` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-119/v82/v82_L2_s119.jsonl` | `0cf0419ba67318dc5f4f4fbff6626f8a` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-116/v82/v82_L3_s116.jsonl` | `1093d4a6eaba2e69c9daa9f6c940f984` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-119/v82/v82_L4_s119.jsonl` | `d73b32266a111bd66fb7b5eacb52f9a0` | untracked |
| attempts | `../trainium-agent-labs/runs/seat-119/v82/v82_L4b_s119.jsonl` | `d73b32266a111bd66fb7b5eacb52f9a0` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-117/v82/v82_L1_s117_verdicts.jsonl` | `7b14820a72293a9906b57f4d62f25a71` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-118/v82/v82_L1_s118_verdicts.jsonl` | `ac45d3a0770a5c7e6503d5e5ffe5a651` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-117/v82/v82_L1b_s117_verdicts.jsonl` | `8ec2cd920be77e920117f026e8ab7625` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-118/v82/v82_L1b_s118_verdicts.jsonl` | `0fa6ed9b5b13aa24b9685d69e30d4825` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-116/v82/v82_L2_s116_verdicts.jsonl` | `ae1b61d6ba0c9ed67d63b357fefb3c71` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-119/v82/v82_L2_s119_verdicts.jsonl` | `5ace6ec4711154d43fbfeb6578e54b99` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-116/v82/v82_L3_s116_verdicts.jsonl` | `4f8b053b8db984ba276fb76899f23165` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-119/v82/v82_L4_s119_verdicts.jsonl` | `ea0cfb846d758399239fd9e86e2959a0` | untracked |
| verdicts | `../trainium-agent-labs/runs/seat-119/v82/v82_L4b_s119_verdicts.jsonl` | `1b0b2ea32f8fd25645e395548fee18cc` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `../trainium-agent-labs/runs/seat-117/v82/v82_L1_s117_usage.jsonl` | `b3ef2dce0bb6cac56db4f315209fd497` | untracked |
| usage | `../trainium-agent-labs/runs/seat-118/v82/v82_L1_s118_usage.jsonl` | `24f117bf63e405e61b523c58ec21fff3` | untracked |
| usage | `../trainium-agent-labs/runs/seat-117/v82/v82_L1b_s117_usage.jsonl` | `fd6b8f980ed7ccf24acf8055b1a08c9e` | untracked |
| usage | `../trainium-agent-labs/runs/seat-118/v82/v82_L1b_s118_usage.jsonl` | `4d8ee51c469c1823a704d6b282cd752a` | untracked |
| usage | `../trainium-agent-labs/runs/seat-116/v82/v82_L2_s116_usage.jsonl` | `892551065e9c5d741e8f0d3c1de98e2b` | untracked |
| usage | `../trainium-agent-labs/runs/seat-119/v82/v82_L2_s119_usage.jsonl` | `c534947b7a26d7fd8ff52d890835a51f` | untracked |
| usage | `../trainium-agent-labs/runs/seat-116/v82/v82_L3_s116_usage.jsonl` | `b3282d4db6f8049d0a088663254a1415` | untracked |
| usage | `../trainium-agent-labs/runs/seat-119/v82/v82_L4_s119_usage.jsonl` | `7fffd2cebe19cad448f0faf876c8bc7d` | untracked |
| usage | `../trainium-agent-labs/runs/seat-119/v82/v82_L4b_s119_usage.jsonl` | `738f9cbb88d8642a4adaf23eccd85968` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-117/v82/v82_L1_s117_nki_verdicts.jsonl` | `6925daaab82a7949e763b1002fff324d` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-118/v82/v82_L1_s118_nki_verdicts.jsonl` | `36ce6bcfd8f67f1617b8c4d6d3abaf99` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-117/v82/v82_L1b_s117_nki_verdicts.jsonl` | `99131aa8e1141368b38717b2a80be98b` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-118/v82/v82_L1b_s118_nki_verdicts.jsonl` | `d9db158d3b4dd2af9ca8b188c9360509` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-116/v82/v82_L2_s116_nki_verdicts.jsonl` | `ca22daa9bf4a3bd973fe271a3d7c704a` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-119/v82/v82_L2_s119_nki_verdicts.jsonl` | `8d3ccff3e6d12a27812ff9a61a89ccbc` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-116/v82/v82_L3_s116_nki_verdicts.jsonl` | `6c34567c2fca41f52a2607e8389482b8` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-119/v82/v82_L4_s119_nki_verdicts.jsonl` | `176ece2076ec88809f42ab9fde1cd456` | untracked |
| v7 verdicts | `../trainium-agent-labs/runs/seat-119/v82/v82_L4b_s119_nki_verdicts.jsonl` | `ea8a7bf570f8cdb0c636344873eead74` | untracked |
