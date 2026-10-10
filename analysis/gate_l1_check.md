# The level-1 gate rule (89417dd): does it hit kernels we already solved?

**Conclusion: no false hits.** On all 11 distinct full-score kernels in our logs (levels 2, 3, 4), the new rule
fires 0 times, and the older gate rules fire 0 times too. On v7's level-1 log it fires on 13 of 40 attempts
(6 distinct kernels, all per-channel `t[c, ...]` reduces), but every one of those scored 0.30: they already
fail in the simulator. The gate only acts on a full score (`gated()` returns early below 1.0), so in this log
it changes no score and no message. Its effect will only show on level-1 kernels that reach 1.0, and none of
our logs has one.

Checked on this branch at 89417dd + 90b8c0a + 9a19bb0, NKI 0.6.0 in Docker, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`.
"New rule" = `gate_nki.partition_fixes` + `gate_nki.dst_shape_fixes` (the level-1 fix); "older rules" = the
rest of `gate_nki.fixes()`. `gated()` holds a full score at 0.95 when `fixes()` returns anything.

## 1. Her tests

```text
check/test_gate_l1.py   flagged & fails 9, flagged & builds 0, not flagged & builds 17, not flagged & fails 2 (known misses: 2)
                        0 failed
GATE= test_v7.py        0 failed
test_requests.py        0 failed
```

(env: `PYTHONPATH=.:check MESSAGES=v5 CARD=category PROMPT1=v2 SAMPLING=qwen REPAIR_PROMPT=restructure
GRADE_TIMEOUT=120 NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, as V7.md says)

## 2. Every full-score kernel we have

Distinct kernels (by sha1 of the code) that scored 1.0, per source. v7's level 2 is not in yet (not pulled);
the baseline's and the replica's level-2 solves stand in for level 2.

| source | level | kernel | first seen | new rule | older rules |
|---|---|---|---|---|---|
| v7 round 2 | 3 | `a680640e78` | `trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_L3.jsonl`:1 | 0 | 0 |
| v7 round 2 | 3 | `ad559224ac` | `trainium-agent-labs/runs/seat-117/Ev7_L3/Ev7_L3.jsonl`:2 | 0 | 0 |
| v7 round 2 | 4 | `8339bbd6f2` | `trainium-agent-labs/runs/seat-118/Ev7_L4/Ev7_L4.jsonl`:9 | 0 | 0 |
| E-div | 3 | `ad559224ac` | `trainium-agent-labs/runs/seat-119/Ediv_L3/Ediv_L3.jsonl`:1 | 0 | 0 |
| E-div | 4 | `8339bbd6f2` | `trainium-agent-labs/runs/seat-118/Ediv_L4/Ediv_L4.jsonl`:9 | 0 | 0 |
| baseline | 2 | `97daf585c6` | `tal-deliv/analysis/logs/baseline/attempts.jsonl`:33 | 0 | 0 |
| baseline | 2 | `d421937e99` | `tal-deliv/analysis/logs/baseline/attempts.jsonl`:212 | 0 | 0 |
| replica | 2 | `97daf585c6` | `tal-deliv/analysis/logs/replica_seat119/attempts.jsonl`:33 | 0 | 0 |
| replica | 2 | `d12ae37f0e` | `tal-deliv/analysis/logs/replica_seat119/attempts.jsonl`:280 | 0 | 0 |
| E-v3 | 4 | `c733c15749` | `tal-deliv/analysis/logs/ev3_L4/attempts_v3_L4.jsonl`:49 | 0 | 0 |
| teammate 117 | 2 | `4b6da1edcd` | `tal-deliv/analysis/logs/teammate_all_seat117/attempts.jsonl`:37 | 0 | 0 |

## 3. v7's level-1 log (`runs/seat-119/Ev7_L1/Ev7_L1.jsonl`, the 14:46 snapshot: run 1 complete, run 2 to round 1)

The new rule fires on these attempts. All scored 0.30 (they fail in the simulator), so the gate never sees them:

| line | run | round | reward | kernel | what the rule says |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 0.30 | `7fa6c4f94b` | `sum_tile[c, ...]` gives an instruction one partition starting at part... |
| 2 | 1 | 0 | 0.30 | `7fa6c4f94b` | `sum_tile[c, ...]` gives an instruction one partition starting at part... |
| 13 | 1 | 3 | 0.30 | `1af84998eb` | `s[i, ...]` gives an instruction one partition starting at partition `... |
| 14 | 1 | 3 | 0.30 | `1af84998eb` | `s[i, ...]` gives an instruction one partition starting at partition `... |
| 15 | 1 | 3 | 0.30 | `1af84998eb` | `s[i, ...]` gives an instruction one partition starting at partition `... |
| 16 | 1 | 3 | 0.30 | `1af84998eb` | `s[i, ...]` gives an instruction one partition starting at partition `... |
| 17 | 1 | 4 | 0.30 | `7a9205b80b` | `psum[i, ...]` gives an instruction one partition starting at partitio... |
| 18 | 1 | 4 | 0.30 | `7a9205b80b` | `psum[i, ...]` gives an instruction one partition starting at partitio... |
| 19 | 1 | 4 | 0.30 | `7a9205b80b` | `psum[i, ...]` gives an instruction one partition starting at partitio... |
| 20 | 1 | 4 | 0.30 | `7a9205b80b` | `psum[i, ...]` gives an instruction one partition starting at partitio... |
| 33 | 2 | 0 | 0.30 | `da12ca2e90` | `t[c, ...]` gives an instruction one partition starting at partition `... |
| 34 | 2 | 0 | 0.30 | `7fe8db273c` | `t[c, ...]` gives an instruction one partition starting at partition `... |
| 36 | 2 | 0 | 0.30 | `ea507b9686` | `t[c, ...]` gives an instruction one partition starting at partition `... |

The other 27 attempts do not trigger it (none scored 1.0 either; best 0.50).

## Reproduce

`analysis/gate_l1_check.py` (copy of the script used; paths are the Docker container's: repository at
`/h/tal-deliv`, pulled logs at `/h/trainium-agent-labs/runs`).
