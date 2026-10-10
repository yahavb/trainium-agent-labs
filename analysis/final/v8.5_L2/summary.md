# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 4/6 | 1.00 1.00 1.00 0.30 0.50 1.00 | 0.80 | 0.30 | 1.00 | run 1: attempt 17, round 4; run 2: attempt 9, round 2; run 3: attempt 11, round 2; run 6: attempt 5, round 1 | 19,617+8,060; 12,070+3,358; 12,103+5,729; 32,374+11,861; 19,772+6,065; 8,127+2,326 | 1 in 1 round(s), 3.5 min; 0; 1 in 1 round(s), 3.5 min; 1 in 1 round(s), 3.5 min; 0; 0 | 7.4 2.7 5.4 10.2 4.8 1.9 | 6 | VERIFIED 4, NOT SOLVED 2 | 0.60 | 0.007 | 0 | VERIFIED 4, FAILED 2 | 0.60 | 0.007 | 0 | 3/5, mean 0.72 |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 2 | 6 | 0.60 | 0.007 | 0 | 0.60 | 0.007 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 104 of 104 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2_s117.jsonl` | `c1e2d08069430df89066c69470666e96` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2b_s117.jsonl` | `e7b8f0f4caa543a08012a9fc7ea21b82` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2c_s117.jsonl` | `131108a68c6b1e1c7ec9fbbce13e7adb` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2_s118.jsonl` | `6869be96eeb4cca71e3a99ad0f083917` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2b_s118.jsonl` | `ba135d33e38fc8962f0866373f1d4082` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2c_s118.jsonl` | `1df44354357d0a1430c781f31aac85c1` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2_s117_verdicts.jsonl` | `2acfe42e629bb030342251678e045e0a` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2b_s117_verdicts.jsonl` | `9a620de4645d14015dc6c1c0ee04ad75` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2c_s117_verdicts.jsonl` | `04efe546d52b9ed24864392c4ebd835d` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2_s118_verdicts.jsonl` | `4b28e8443822374b3147b4e7a3b4c17e` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2b_s118_verdicts.jsonl` | `a673384a16e3dfadf728b5dce1224b1b` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2c_s118_verdicts.jsonl` | `c9dc197821a5edf8b5ac6ad23059f8b8` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2_s117_usage.jsonl` | `5a2894c9d9e0728cdf76170a6b8ab6a0` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2b_s117_usage.jsonl` | `2ee87619b4a5753851d616c490bef88d` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2c_s117_usage.jsonl` | `242493e45dfa0fe6728277ebefdb5c6d` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2_s118_usage.jsonl` | `65a1e73f06a8c233e3f36cf38ef21dcc` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2b_s118_usage.jsonl` | `7ad07bdb29b77864b0bbe8e7c4b147c0` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2c_s118_usage.jsonl` | `16c13571fb3700cc8da1b27b34bb4baa` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2_s117_nki_verdicts.jsonl` | `f0650e58c1e17b69f71dd08426d507d4` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2b_s117_nki_verdicts.jsonl` | `7d6c167922685445611542bac86899a0` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-117__v85_L2c_s117_nki_verdicts.jsonl` | `1b8a405ffc036a2c42382f0d37f80573` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2_s118_nki_verdicts.jsonl` | `03f8a7ce1e26ce917d14d71bd16217f7` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2b_s118_nki_verdicts.jsonl` | `0d7509b98cb74b9b4810299572ea18f4` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.5_L2/final_seat-118__v85_L2c_s118_nki_verdicts.jsonl` | `2b394e44d970f56219342b0e276db09c` | untracked |
