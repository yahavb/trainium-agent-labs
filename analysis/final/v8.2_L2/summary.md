# Results by level

| level | runs | solved | best score per run | mean | min | max | solve: attempts / round | tokens per run (prompt+answer) | cut off by max_tokens, per run (attempts, rounds, those rounds' wall time) | minutes per run (first request to last answer) | verdicts | held-out claims | mean confidence | Brier | confident (>=0.5) but wrong | v7 verdicts | v7 mean confidence | v7 Brier (vs our held-out) | v7 confident but failed our held-out | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 9 | 5/9 | 0.50 1.00 0.50 0.50 1.00 0.50 1.00 1.00 1.00 | 0.78 | 0.50 | 1.00 | run 2: attempt 19, round 4; run 5: attempt 13, round 3; run 7: attempt 7, round 1; run 8: attempt 11, round 2; run 9: attempt 5, round 1 | 31,498+10,720; 19,381+5,844; 30,640+11,932; 30,545+9,759; 16,111+5,134; 31,278+9,904; 8,324+2,390; 11,939+3,471; 8,074+2,186 | 0; 0; 1 in 1 round(s), 3.5 min; 0; 0; 0; 0; 0; 0 | 8.7 4.7 10.1 7.9 4.2 7.9 1.9 2.8 1.7 | 9 | VERIFIED 5, NOT SOLVED 4 | 0.50 | 0.006 | 0 | VERIFIED 5, FAILED 4 | 0.50 | 0.005 | 0 | 3/5, mean 0.72 |

*solve: attempts* counts every attempt in that run up to and including the first 1.0 (all samples of every earlier round). Scores are the best loop reward per run. Held-out claims, confidence and Brier are counted per level over every verdict in the --verdicts files, so pass the verdict files that belong to these runs.

## Two verdicts, one yardstick

Both verdicts after every level, scored against the same outcome: did the kernel pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs with a held-out result count. Ours: confidence from `agent.confidence()`. v7: `verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).

| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |
|---|---|---|---|---|---|---|---|
| 2 | 9 | 0.50 | 0.006 | 0 | 0.50 | 0.005 | 0 |

Unpaired verdicts: none.

Tokens: server counts from the usage log for 192 of 192 attempts; 0 unmatched kept chars/4 estimates (marked "est." per run).

## Inputs

| role | file | md5 | last commit |
|---|---|---|---|
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-116__v82_L2_s116.jsonl` | `46c5bf5a7132ce5766799b7d1379298f` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-117__v82x_L2_s117.jsonl` | `848f50ab3a926dc32adb9cbc8a042e89` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82_L2_s118.jsonl` | `78adfdffbdfb8d82e2981f09878761ee` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82x_L2_s118.jsonl` | `4793ca4698decf6c1703da3a9e389969` | untracked |
| attempts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-119__v82_L2_s119.jsonl` | `0cf0419ba67318dc5f4f4fbff6626f8a` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-116__v82_L2_s116_verdicts.jsonl` | `ae1b61d6ba0c9ed67d63b357fefb3c71` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-117__v82x_L2_s117_verdicts.jsonl` | `083126798345967eda8537edc1c46751` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82_L2_s118_verdicts.jsonl` | `71d40d1a997fa666a3ef968b5bc02ad8` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82x_L2_s118_verdicts.jsonl` | `8742eca7ec1dfc0d3c52200bc4d77657` | untracked |
| verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-119__v82_L2_s119_verdicts.jsonl` | `5ace6ec4711154d43fbfeb6578e54b99` | untracked |
| baseline | `analysis/logs/baseline/attempts.jsonl` | `9275898538ed1a69c28a8a5fa52a3788` | a0af434 |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-116__v82_L2_s116_usage.jsonl` | `892551065e9c5d741e8f0d3c1de98e2b` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-117__v82x_L2_s117_usage.jsonl` | `f73402c946654fb1d25456ff509c51d5` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82_L2_s118_usage.jsonl` | `cde1feed5fbff207ecdf5af2564e026e` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82x_L2_s118_usage.jsonl` | `435b6e1e35f3eb44c08331d7991882f8` | untracked |
| usage | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-119__v82_L2_s119_usage.jsonl` | `c534947b7a26d7fd8ff52d890835a51f` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-116__v82_L2_s116_nki_verdicts.jsonl` | `ca22daa9bf4a3bd973fe271a3d7c704a` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-117__v82x_L2_s117_nki_verdicts.jsonl` | `bfd7428125729d179671ba31c5c1513b` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82_L2_s118_nki_verdicts.jsonl` | `f085cbbd519021b43d6b8b2201155338` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-118__v82x_L2_s118_nki_verdicts.jsonl` | `1764d55cdf4265bdee62cb5b94a0fccc` | untracked |
| v7 verdicts | `/var/folders/gf/gz7tm2sj57s7m9lll10z8yb40000gn/T/final_tables_muekf0h7/v8.2_L2/final_seat-119__v82_L2_s119_nki_verdicts.jsonl` | `8d3ccff3e6d12a27812ff9a61a89ccbc` | untracked |
