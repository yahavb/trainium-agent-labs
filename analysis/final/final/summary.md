# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 7/7 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2; run 2: attempt 17, round 4; run 3: attempt 9, round 2; run 4: attempt 9, round 2; run 5: attempt 3, round 0; run 6: attempt 9, round 2; run 7: attempt 29, round 7 | 13,595+10,935; 21,915+9,679; 13,518+6,589; 13,702+11,023; 4,856+1,793; 13,692+6,854; 35,152+17,485 | 3 in 2 round(s), 7.5 min; 1 in 1 round(s), 3.7 min; 1 in 1 round(s), 3.6 min; 3 in 1 round(s), 7.4 min; 0; 1 in 1 round(s), 3.7 min; 2 in 2 round(s), 7.3 min | 8.8 8.7 6.1 10.1 1.4 6.3 15.3 | 7 | VERIFIED 7 | 0.90 | 0.010 | 0 | VERIFIED 7 | 0.90 | 0.010 | 0 | 0/5, mean 0.30 |
| 2 | 11 | 9/11 | 1.00 1.00 1.00 1.00 0.50 1.00 1.00 1.00 0.50 1.00 1.00 | 0.91 | 0.50 | 1.00 | run 1: attempt 9, round 2; run 2: attempt 5, round 1; run 3: attempt 15, round 3; run 4: attempt 19, round 4; run 6: attempt 5, round 1; run 7: attempt 11, round 2; run 8: attempt 3, round 0; run 10: attempt 5, round 1; run 11: attempt 5, round 1 | 12,072+3,786; 8,133+2,298; 16,139+7,147; 19,717+6,353; 31,505+10,383; 8,129+2,274; 11,886+3,733; 4,348+1,193; 32,075+10,395; 8,127+2,264; 8,127+2,563 | 0; 0; 1 in 1 round(s), 3.6 min; 0; 0; 0; 0; 0; 0; 0; 0 | 3.1 1.8 6.5 5.2 8.2 1.9 3.0 0.9 8.4 1.8 2.1 | 11 | VERIFIED 9, NOT SOLVED 2 | 0.74 | 0.008 | 0 | VERIFIED 9, FAILED 2 | 0.74 | 0.008 | 0 | 3/5, mean 0.72 |
| 3 | 5 | 5/5 | 1.00 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 1, round 0; run 2: attempt 1, round 0; run 3: attempt 9, round 2; run 4: attempt 1, round 0; run 5: attempt 1, round 0 | 4,624+1,280; 4,624+1,238; 12,631+3,704; 4,624+1,255; 4,624+1,230 | 0; 0; 0; 0; 0 | 1.0 1.0 2.9 1.0 1.0 | 5 | VERIFIED 4, PASSES THE LOOP'S SHAPES ONLY 1 | 0.63 | 0.189 | 1 | VERIFIED 5 | 0.74 | 0.163 | 1 | 0/5, mean 0.30 |
| 4 | 5 | 5/5 | 1.00 1.00 1.00 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2; run 2: attempt 9, round 2; run 3: attempt 9, round 2; run 4: attempt 9, round 2; run 5: attempt 9, round 2 | 13,077+3,967; 13,077+3,967; 13,089+4,046; 13,101+4,028; 13,101+3,961 | 0; 0; 0; 0; 0 | 9.5 7.7 19.5 6.5 6.3 | 5 | VERIFIED 5 | 0.90 | 0.010 | 0 | VERIFIED 5 | 0.88 | 0.014 | 0 | 0/5, mean 0.60 |
| 5 | 7 | 0/7 | 0.88 0.88 0.88 0.88 0.88 0.88 0.88 | 0.88 | 0.88 | 0.88 | - | 10,172+7,399; 18,134+13,970; 12,848+9,610; 10,200+7,798; 10,172+7,443; 12,843+9,404; 10,172+7,458 | 0; 0; 0; 0; 0; 0; 0 | 5.8 11.2 7.6 6.3 5.9 7.5 5.9 | 6 | NOT SOLVED 6 | 0.00 | 0.000 | 0 | FAILED 6 | 0.00 | 0.000 | 0 | - |
| 6 | 7 | 0/7 | 0.75 0.75 0.75 0.75 0.75 0.75 0.75 | 0.75 | 0.75 | 0.75 | - | 15,565+12,548; 12,727+10,164; 12,858+10,762; 12,785+10,732; 2,490+2,025; 15,887+12,623; 10,294+8,435 | 0; 0; 0; 0; 0; 0; 0 | 26.4 21.3 8.5 8.3 1.6 9.9 6.6 | 6 | NOT SOLVED 6 | 0.00 | 0.000 | 0 | FAILED 6 | 0.00 | 0.000 | 0 | - |
| 7 | 4 | 0/4 | 0.75 0.75 0.75 0.75 | 0.75 | 0.75 | 0.75 | - | 10,320+7,417; 10,320+7,514; 10,325+7,535; 10,320+7,556 | 0; 0; 0; 0 | 5.8 5.8 5.9 5.9 | 4 | NOT SOLVED 4 | 0.00 | 0.000 | 0 | FAILED 4 | 0.00 | 0.000 | 0 | - |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 1 | 7 | 0.90 | 0.010 | 0 | 0.90 | 0.010 | 0 |
| 2 | 11 | 0.74 | 0.008 | 0 | 0.74 | 0.008 | 0 |
| 3 | 5 | 0.63 | 0.189 | 1 | 0.74 | 0.163 | 1 |
| 4 | 5 | 0.90 | 0.010 | 0 | 0.88 | 0.014 | 0 |
| 5 | 6 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| 6 | 6 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| 7 | 4 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 676 of 676 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L5_s115.jsonl` | `0d3437cac4f6c66ce3624b6bc1a46afa` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L6_s115.jsonl` | `aba7df5028b1cf1410ee5bf271d0b21b` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7_s115.jsonl` | `ff5270cea6c33a91b49740d0fffe4293` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7b_s115.jsonl` | `fbb9a55d43706078021d235324f179bd` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__v82_L3_s116.jsonl` | `1093d4a6eaba2e69c9daa9f6c940f984` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5_s116.jsonl` | `5b4a02367932a1317947322e0aefcadb` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5b_s116.jsonl` | `c78e326a378685714c25079b12dd15ef` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5c_s116.jsonl` | `24995586f9044f5ca2cbbc7b896d4238` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6_s116.jsonl` | `869bc05f7e0cb65ff3c4c4a3a526d47e` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6b_s116.jsonl` | `7a4eae3854838fb6901a3333d504ab59` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1_s117.jsonl` | `1690af776147b8860029ac59162b7aca` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1b_s117.jsonl` | `12a7c993ff1f2834bddcf8cf77fbbae5` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1c_s117.jsonl` | `bc5508f9cb15304b10257244e65ba913` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2_s117.jsonl` | `29cb242cb2c5c063c6283d7d97950b23` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2b_s117.jsonl` | `9e4d908a4f722084c3617820f042c60b` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2c_s117.jsonl` | `7eb42ae3e688ef75d385dc666fabba8e` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2d_s117.jsonl` | `383871059f5c439ecfa543e33c615bb2` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L5_s117.jsonl` | `e77be8cade8120998468085389945e0f` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L6_s117.jsonl` | `a6f9a33b325479f31f74e5512f5b4275` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1_s118.jsonl` | `bb221a965bb664d94dc4653002fba90f` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1b_s118.jsonl` | `10ddf1b9e2be6cac478096977b1fbd0b` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L3_s118.jsonl` | `bd5f79730a83556673b232f2100994da` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1_s118.jsonl` | `407860896f39ec99073f52b437962bcf` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1b_s118.jsonl` | `6918253173c6087a960446399e28ab4f` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L5_s118.jsonl` | `d7e248a4e2def3a729db6c38df80ff74` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L6_s118.jsonl` | `953ed0f462702f636f5aa9cec3ec4925` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L7_s118.jsonl` | `b1732622f63a24bea45479de1a8551be` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L3d_s119.jsonl` | `22df0cdf3f4f532480cdc4065a0db56c` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4_s119.jsonl` | `d73b32266a111bd66fb7b5eacb52f9a0` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4b_s119.jsonl` | `d73b32266a111bd66fb7b5eacb52f9a0` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4c_s119.jsonl` | `5f7c89e125d9c1ea9af601fcb6ad75d4` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2_s119.jsonl` | `b5b1a72526731ae3f9b4edbbf3f71451` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2b_s119.jsonl` | `1067e99c133a7deacb7a67e527e0ed36` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2c_s119.jsonl` | `c311cd0f8803b2094f5bd21a1b3fe820` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L5_s119.jsonl` | `5fdb964caf44a75a04912aa81a174ea4` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6_s119.jsonl` | `fc77d0adc2be8f9d40c4f5a18ba84f3e` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6b_s119.jsonl` | `7a4214817220cb41949604add5d1eead` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L7_s119.jsonl` | `1531d866194770f16031d6eac08ff304` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L5_s115_verdicts.jsonl` | `dab10358f365cecc30ed89423510df46` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L6_s115_verdicts.jsonl` | `c4e372797e1a261d0032f43dd29e113e` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7_s115_verdicts.jsonl` | `a78fb90e09a41c84755519e1f130b31b` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7b_s115_verdicts.jsonl` | `6f846cbaf828fa0869a801eed8270d1f` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__v82_L3_s116_verdicts.jsonl` | `4f8b053b8db984ba276fb76899f23165` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5_s116_verdicts.jsonl` | `98d6c5508f83cd0ad10cc389a4e989fb` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5b_s116_verdicts.jsonl` | `1952c6c072edc2613aeb61449a489b5b` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6_s116_verdicts.jsonl` | `904aff117e75a4d5ab05654932ca4a99` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6b_s116_verdicts.jsonl` | `1b4b4771a737219380a18cd64c08d99a` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1_s117_verdicts.jsonl` | `7b14820a72293a9906b57f4d62f25a71` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1b_s117_verdicts.jsonl` | `8ec2cd920be77e920117f026e8ab7625` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1c_s117_verdicts.jsonl` | `3083c2df21609abbfbdf7a34f278f8ca` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2_s117_verdicts.jsonl` | `8fa6afc9c200c3f1ff3e72cef50154a2` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2b_s117_verdicts.jsonl` | `aec046732cece5d255b1b4b6f704abf6` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2c_s117_verdicts.jsonl` | `d5371628f6ee3b9fa6b75bc08f7d5d09` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2d_s117_verdicts.jsonl` | `62d0ba99e0d39c2f7d7877819880b8d1` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L5_s117_verdicts.jsonl` | `0cf6f6ac1a5ef37021dae0e769033680` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L6_s117_verdicts.jsonl` | `1585d3e6ed6f4af809e72a451780cc29` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1_s118_verdicts.jsonl` | `ac45d3a0770a5c7e6503d5e5ffe5a651` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1b_s118_verdicts.jsonl` | `0fa6ed9b5b13aa24b9685d69e30d4825` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L3_s118_verdicts.jsonl` | `6dd4e240c6d672195e34d57f43170081` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1_s118_verdicts.jsonl` | `b633a7109e749e79d76e16e0871782e4` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1b_s118_verdicts.jsonl` | `fee1464db71d925c1e6bbe1e55a07bbb` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L5_s118_verdicts.jsonl` | `6accb5d0227615744607461ff2a3341e` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L7_s118_verdicts.jsonl` | `18d6107365056def67e9040e6a0918f7` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L3d_s119_verdicts.jsonl` | `4c2591513b4b15a632ff653d6cd06e8e` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4_s119_verdicts.jsonl` | `ea0cfb846d758399239fd9e86e2959a0` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4b_s119_verdicts.jsonl` | `1b0b2ea32f8fd25645e395548fee18cc` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4c_s119_verdicts.jsonl` | `6199da2184a514c8f076e04f041d9fc6` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2_s119_verdicts.jsonl` | `fb93006a43e7ceca95ce1511873753f6` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2b_s119_verdicts.jsonl` | `2412d4a181d9303ded53ccc3cb6c7c2a` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2c_s119_verdicts.jsonl` | `b5ca2bde45808602b92dc5a86bf04e94` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L5_s119_verdicts.jsonl` | `6c7ca424bb0edf71ac8e7afb2161cbf8` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6_s119_verdicts.jsonl` | `9f9d5bca49eb03443ecd07819ea12d47` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6b_s119_verdicts.jsonl` | `a98d7dcb5aa8b2c54be33ab8911274ab` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L7_s119_verdicts.jsonl` | `dd1314a126897866551606fdc2467493` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L5_s115_usage.jsonl` | `37ec002f7dbce6f3883f1d22a847057d` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L6_s115_usage.jsonl` | `4bae0042099ffa8c3b6b764e5b3a1040` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7_s115_usage.jsonl` | `744825b3ab4dd9c17ebac44d9d48ccc8` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7b_s115_usage.jsonl` | `7a627b0195a6a43abfd63f29b0853a50` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__v82_L3_s116_usage.jsonl` | `b3282d4db6f8049d0a088663254a1415` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5_s116_usage.jsonl` | `0b824decb85b9ed7326d1db025200b94` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5b_s116_usage.jsonl` | `19f1d9ff9de7913e75adbe2a8ff65211` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5c_s116_usage.jsonl` | `a0615331324de63bbcdd2c2085ea3188` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6_s116_usage.jsonl` | `1d2bd33c2341d3bdca9fa34822468442` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6b_s116_usage.jsonl` | `f36e87bfee9ca143ac4effdbd3ee2f97` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1_s117_usage.jsonl` | `b3ef2dce0bb6cac56db4f315209fd497` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1b_s117_usage.jsonl` | `fd6b8f980ed7ccf24acf8055b1a08c9e` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1c_s117_usage.jsonl` | `61007d957f3dcfcf97acad2e919f6c7c` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2_s117_usage.jsonl` | `2cb257b20471b90032b3ef762ecd7fbc` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2b_s117_usage.jsonl` | `8a61f8eb588533bcb947e58a911fa8e0` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2c_s117_usage.jsonl` | `e5e299d3545a0c276c65fac36632bfa8` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2d_s117_usage.jsonl` | `a823e7c3e3a53fd76c265c627e18d8dc` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L5_s117_usage.jsonl` | `fa28634c2c47420496025b2b79f00f7e` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L6_s117_usage.jsonl` | `6c0d464229fa3fa73a4081770a9a7f81` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1_s118_usage.jsonl` | `24f117bf63e405e61b523c58ec21fff3` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1b_s118_usage.jsonl` | `4d8ee51c469c1823a704d6b282cd752a` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L3_s118_usage.jsonl` | `aad5d1a7e5e9e74f8aa2a764f1a37e87` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1_s118_usage.jsonl` | `1ffa285e226b7b071ddfca754c0e2ba8` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1b_s118_usage.jsonl` | `7d50a04b22970d0a20ed4670792b334f` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L5_s118_usage.jsonl` | `2eae75c477cd5043255dd865a35b6fdc` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L6_s118_usage.jsonl` | `aba52abe77f5b54a313263ec411ae9aa` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L7_s118_usage.jsonl` | `c4139f154bef9e5fd9406af835b2aa89` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L3d_s119_usage.jsonl` | `ed28ec6da41b3dcc7cc50c18c4d032d4` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4_s119_usage.jsonl` | `7fffd2cebe19cad448f0faf876c8bc7d` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4b_s119_usage.jsonl` | `738f9cbb88d8642a4adaf23eccd85968` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4c_s119_usage.jsonl` | `6013a1ca318203fdd36b45a161f1cc72` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2_s119_usage.jsonl` | `f2f371143149961421d42d9f124b5e47` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2b_s119_usage.jsonl` | `65c08a24d9c7b773b808cff539c469d1` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2c_s119_usage.jsonl` | `14d2620e5a60d9913ad2fb0ea3ad7ad3` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L5_s119_usage.jsonl` | `0d4fd57fc18a35f9bb4c0b4b0372282a` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6_s119_usage.jsonl` | `0bd7aba7e93aadb0aa6003f0b189aaea` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6b_s119_usage.jsonl` | `4f51ec67cbec33f2e58d47180a67f1c0` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L7_s119_usage.jsonl` | `08468d7905850f8c8035c53c07f1d211` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L5_s115_nki_verdicts.jsonl` | `122f3ee413aea001f4b35d6c25e7841b` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L6_s115_nki_verdicts.jsonl` | `bf7111370e784ea2d35d2120f31dab7b` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7_s115_nki_verdicts.jsonl` | `58fe132ebfed765899a19c841cd6fbc2` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-115__warm_L7b_s115_nki_verdicts.jsonl` | `b88f37e965ad32848dfe16853517c318` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__v82_L3_s116_nki_verdicts.jsonl` | `6c34567c2fca41f52a2607e8389482b8` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5_s116_nki_verdicts.jsonl` | `069b72c526a697ea48c37bc5bf2311ea` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L5b_s116_nki_verdicts.jsonl` | `0714db4c0bdfa5923e1d53cd8bc6af8d` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6_s116_nki_verdicts.jsonl` | `2af3175f92bd6125fb6ae501d6dd9f3a` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-116__warm_L6b_s116_nki_verdicts.jsonl` | `18e13cc86adc97c8c1363a2a2401140c` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1_s117_nki_verdicts.jsonl` | `6925daaab82a7949e763b1002fff324d` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1b_s117_nki_verdicts.jsonl` | `99131aa8e1141368b38717b2a80be98b` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v82_L1c_s117_nki_verdicts.jsonl` | `2d37b7b00d07ae50cd27eaeaf99e7cb4` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2_s117_nki_verdicts.jsonl` | `898e6aaf0430934399ac2b4f77acc0cf` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2b_s117_nki_verdicts.jsonl` | `cc9b7846885dd2d70b1423b011060444` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2c_s117_nki_verdicts.jsonl` | `fd761ff35b7e782fb657e5f1eef95f83` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__v83_L2d_s117_nki_verdicts.jsonl` | `4a5499222bf73af95cf03993d69219b6` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L5_s117_nki_verdicts.jsonl` | `bc6b24fff1137c3f2a1f48cfaa795bd9` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-117__warm_L6_s117_nki_verdicts.jsonl` | `0e275ecc5263fbf9c790826b30ce5cc4` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1_s118_nki_verdicts.jsonl` | `36ce6bcfd8f67f1617b8c4d6d3abaf99` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L1b_s118_nki_verdicts.jsonl` | `d9db158d3b4dd2af9ca8b188c9360509` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82_L3_s118_nki_verdicts.jsonl` | `7bcb54e7ca344fa1e49c6d90d70e7177` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1_s118_nki_verdicts.jsonl` | `46fe6d65e4fdc9b0b3ff58e3e6362799` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__v82x_L1b_s118_nki_verdicts.jsonl` | `821c283423911b16cb385f1150df0340` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L5_s118_nki_verdicts.jsonl` | `a53e710d95628af14a9e9abfaf1f383d` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-118__warm_L7_s118_nki_verdicts.jsonl` | `5c71d109bc18779610de34e3246224d2` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L3d_s119_nki_verdicts.jsonl` | `c5926d8207e93fb7ded35275305b11d3` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4_s119_nki_verdicts.jsonl` | `176ece2076ec88809f42ab9fde1cd456` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4b_s119_nki_verdicts.jsonl` | `ea8a7bf570f8cdb0c636344873eead74` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v82_L4c_s119_nki_verdicts.jsonl` | `b731a7b03778ef0ed662cc8ca77e4a39` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2_s119_nki_verdicts.jsonl` | `7e04845dd8bc268b0b5fcc00cd6d6b05` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2b_s119_nki_verdicts.jsonl` | `d32c73d3dbf540683cb9e344f046b318` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__v83_L2c_s119_nki_verdicts.jsonl` | `1335c0f7a89a6b81873ec31c69174c93` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L5_s119_nki_verdicts.jsonl` | `f57d79910c4c0eb422f5b8180dd14ae4` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6_s119_nki_verdicts.jsonl` | `a786eb08792e8bab2bc2e09944e466e3` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L6b_s119_nki_verdicts.jsonl` | `c36facac11517e08b6ddf7b9c8fe817f` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/final/final_seat-119__warm_L7_s119_nki_verdicts.jsonl` | `41ca8dd029a209c8c3f8ad69dc03eecd` | untracked |
