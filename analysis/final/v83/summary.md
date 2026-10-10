# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 0/1 | 0.88 | 0.88 | 0.88 | 0.88 | - | 26,072+8,050 | 0 | 6.5 | 1 | NOT SOLVED 1 | 0.00 | 0.000 | 0 | FAILED 1 | 0.00 | 0.000 | 0 | - |
| 6 | 1 | 0/1 | 0.75 | 0.75 | 0.75 | 0.75 | - | 34,442+11,348 | 0 | 9.2 | 1 | NOT SOLVED 1 | 0.00 | 0.000 | 0 | FAILED 1 | 0.00 | 0.000 | 0 | - |
| 7 | 1 | 0/1 | 0.75 | 0.75 | 0.75 | 0.75 | - | 30,178+9,951 | 0 | 7.9 | 1 | NOT SOLVED 1 | 0.00 | 0.000 | 0 | FAILED 1 | 0.00 | 0.000 | 0 | - |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| 6 | 1 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| 7 | 1 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 84 of 84 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L5_s118.jsonl` | `0b3cd75b7e50ea837a93fde13f03bac5` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L7_s118.jsonl` | `a7f40bf0f7a7f4c750455be034e74149` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-119__v83_L6_s119.jsonl` | `67710dd67433de6b776a2a006a91874a` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L5_s118_verdicts.jsonl` | `800615300a32e711de13fdcaff60556d` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L7_s118_verdicts.jsonl` | `bacad893db4514db0b4b9ebccd25358b` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-119__v83_L6_s119_verdicts.jsonl` | `96906f9b5e30e037ae2356506d1fb898` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L5_s118_usage.jsonl` | `23350d4e712e5b76cd8f6ffd43d9881c` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L7_s118_usage.jsonl` | `d91ec4b836e674a22b4d784ca129efa7` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-119__v83_L6_s119_usage.jsonl` | `e014a6b34a142124e17f5608140703b5` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L5_s118_nki_verdicts.jsonl` | `013f6e2b3ee798ad2b389d14c260a8a6` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-118__v83_L7_s118_nki_verdicts.jsonl` | `2d0a48ce95c2dca540fea4948d487e9e` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/v83/final_seat-119__v83_L6_s119_nki_verdicts.jsonl` | `e082b29b2dd653abede8dce7802a2c2b` | untracked |
