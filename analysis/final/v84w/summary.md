# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 2 | 0/2 | 0.75 0.75 | 0.75 | 0.75 | 0.75 | - | 10,326+7,466; 15,736+11,511 | 0; 0 | 5.8 9.0 | 2 | NOT SOLVED 2 | 0.00 | 0.000 | 0 | FAILED 2 | 0.00 | 0.000 | 0 | - |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 7 | 2 | 0.00 | 0.000 | 0 | 0.00 | 0.000 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 40 of 40 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7_s116.jsonl` | `d250fc2b74e7bde740f833100279c925` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7b_s116.jsonl` | `889a9110373ff7bd9871479879b483f7` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7_s116_verdicts.jsonl` | `ea72c69ea83d2f8ac5ac8a06fbcb84c2` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7b_s116_verdicts.jsonl` | `5f5c41e6410153e8ebc19a9fc5f73f16` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7_s116_usage.jsonl` | `7807c63655859420178c83ab86baadce` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7b_s116_usage.jsonl` | `cf6800ae158d6c6f5dc392c16fc16780` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7_s116_nki_verdicts.jsonl` | `ed293dd5a8ed4afb2f6ae25409cf4b9e` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v84w/final_seat-116__v84w_L7b_s116_nki_verdicts.jsonl` | `f846d751cf4dfc15b0c784fe27d95635` | untracked |
