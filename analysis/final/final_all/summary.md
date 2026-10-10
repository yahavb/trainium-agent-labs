# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1/1 | 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 25, round 6 | 30,704+17,375 | 3 in 2 round(s), 7.8 min | 13.9 | 1 | VERIFIED 1 | 0.90 | 0.010 | 0 | VERIFIED 1 | 0.90 | 0.010 | 0 | 0/5, mean 0.30 |
| 2 | 1 | 1/1 | 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 29, round 7 | 31,631+11,567 | 1 in 1 round(s), 3.5 min | 10.0 | 1 | VERIFIED 1 | 0.90 | 0.010 | 0 | VERIFIED 1 | 0.90 | 0.010 | 0 | 3/5, mean 0.72 |
| 3 | 1 | 0/1 | 0.30 | 0.30 | 0.30 | 0.30 | - | 16,994+4,906 | 0 | 3.7 | 1 | NOT SOLVED 1 | 0.00 | 0.000 | 0 | FAILED 1 | 0.00 | 0.000 | 0 | 0/5, mean 0.30 |
| 4 | 1 | 1/1 | 1.00 | 1.00 | 1.00 | 1.00 | run 1: attempt 9, round 2 | 13,110+3,986 | 0 | 3.1 | 1 | VERIFIED 1 | 0.90 | 0.010 | 0 | VERIFIED 1 | 0.88 | 0.014 | 0 | 0/5, mean 0.60 |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.90 | 0.010 | 0 | 0.90 | 0.010 | 0 |
| 2 | 1 | 0.90 | 0.010 | 0 | 0.90 | 0.010 | 0 |
| 3 | 1 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |
| 4 | 1 | 0.90 | 0.010 | 0 | 0.88 | 0.014 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 88 of 88 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/final_all/final_seat-116__final_all.jsonl` | `f617a2711fa212e62a118cc2b56bd6a0` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/final_all/final_seat-116__final_all_verdicts.jsonl` | `350f68daeae626994cbb872b275c84c4` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/final_all/final_seat-116__final_all_usage.jsonl` | `792e0b1acb8cc12adfa80d1d26f63598` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_xujj35q9/final_all/final_seat-116__final_all_nki_verdicts.jsonl` | `acec6668f44e44b177556bc9c0d430d9` | untracked |
