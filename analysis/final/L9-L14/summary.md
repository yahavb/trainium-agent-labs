# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 2 | 2/2 | 1.00 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2; run 2: attempt 9, round 2 | 13,395+5,614; 13,459+5,847 | 0; 0 | 4.4 4.6 | 2 | UNVERIFIED 2 | 0.90 | - | 0 | VERIFIED 2 | 0.80 | - | - | - |
| 10 | 1 | 1/1 | 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2 | 14,168+6,973 | 0 | 5.5 | 1 | UNVERIFIED 1 | 0.90 | - | 0 | VERIFIED 1 | 0.80 | - | - | - |
| 11 | 2 | 0/2 | 0.67 0.30 | 0.48 | 0.30 | 0.67 | - | 29,031+10,249; 18,300+6,433 | 0; 0 | 10.9 6.7 | 1 | UNVERIFIED 1 | 0.00 | - | 0 | FAILED 1 | 0.00 | - | - | - |
| 12 | 2 | 0/2 | 0.50 0.30 | 0.40 | 0.30 | 0.50 | - | 34,159+11,750; 33,257+11,363 | 0; 0 | 9.4 9.5 | 2 | UNVERIFIED 2 | 0.00 | - | 0 | FAILED 2 | 0.00 | - | - | - |
| 13 | 1 | 0/1 | 0.50 | 0.50 | 0.50 | 0.50 | - | 21,354+5,690 | 0 | 4.5 | 1 | UNVERIFIED 1 | 0.00 | - | 0 | FAILED 1 | 0.00 | - | - | - |
| 14 | 1 | 0/1 | 0.50 | 0.50 | 0.50 | 0.50 | - | 28,707+15,601 | 2 in 2 round(s), 7.2 min | 14.1 | 1 | UNVERIFIED 1 | 0.00 | - | 0 | FAILED 1 | 0.00 | - | - | - |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| - | 0 | | | | | | |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 204 of 204 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L10_s115.jsonl` | `1deb0b3d956e6b33a9a462250ffc7c42` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L14_s115.jsonl` | `0ae9972a399ea3da39b31925fede8ce2` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-117__final_L11_s117.jsonl` | `0fc10c9f5e22309d1bb396d31c046112` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-117__final_L11b_s117.jsonl` | `809b4dc242e153e8e8b14d826b518ff7` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12_s118.jsonl` | `9db5d2581cc06b1a3a9429c4572a3663` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12b_s118.jsonl` | `094248183d6a482d21ea1f835fddb1fe` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L13_s119.jsonl` | `7f17e38982cbc1863b9f9c4ab3457f2b` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9_s119.jsonl` | `626c69ef83ce521a457c13f530161df2` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9b_s119.jsonl` | `912c4f96fb3ed762efb53704de2a276f` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L10_s115_verdicts.jsonl` | `1eb0aa23419399d33fd5b12bfc38c02f` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L14_s115_verdicts.jsonl` | `d8c8de40562bf2946ade1fda32036c87` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-117__final_L11_s117_verdicts.jsonl` | `50a23feea1b61a26df98395be8d69fa6` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12_s118_verdicts.jsonl` | `72ebac1a88c9750bec27a610300e3cf2` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12b_s118_verdicts.jsonl` | `a41347422b42223ae0c93500fed26b45` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L13_s119_verdicts.jsonl` | `2502653857b96da6344513214aab05db` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9_s119_verdicts.jsonl` | `17cff310b9373349c225d7bd4487f1ba` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9b_s119_verdicts.jsonl` | `08391c0e29b7def3ba632a6e489f4153` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L10_s115_usage.jsonl` | `3edd93d60121103e79c550d43da4ae60` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L14_s115_usage.jsonl` | `06472f8e17241cc54e2d3dcdc9179181` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-117__final_L11_s117_usage.jsonl` | `19b62352ec4f8de9740fc89e4b8f6a82` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-117__final_L11b_s117_usage.jsonl` | `cdcc9f1909217362936a0f58ef494fb2` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12_s118_usage.jsonl` | `5305735c11d091df06f6dd3e60b35cdb` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12b_s118_usage.jsonl` | `4d5041e9fc406fbed9400c70c4b86e3c` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L13_s119_usage.jsonl` | `bfda286b4805a1f32c54747a953a33d3` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9_s119_usage.jsonl` | `4b18d29df9bc67d35ba25cd74566e229` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9b_s119_usage.jsonl` | `36cb016b30dbf5cbf44daa31e0162946` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L10_s115_nki_verdicts.jsonl` | `c3446e10ea5df11d78ab64ee2ae0a3ab` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-115__final_L14_s115_nki_verdicts.jsonl` | `3f4bc16e23b163006b0ff83eba358be7` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-117__final_L11_s117_nki_verdicts.jsonl` | `6ff2eebca7d9a5f2c42e43d460ca80cd` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12_s118_nki_verdicts.jsonl` | `86d5726f0e1122f7c284ab070c70d91a` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-118__final_L12b_s118_nki_verdicts.jsonl` | `9492876beef4706ca78f3e4222c0c5c6` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L13_s119_nki_verdicts.jsonl` | `0a68d9c1c087d074ce6ac95b14b7a54b` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9_s119_nki_verdicts.jsonl` | `ff0080f966ac1e7ad1c1962a208cf769` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/L9-L14/final_seat-119__final_L9b_s119_nki_verdicts.jsonl` | `703b3c3721a48bb388a42c66e30afda3` | untracked |
