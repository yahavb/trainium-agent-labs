# Attempt logs

Every attempt and its score, copied byte for byte from what was pulled off the seats (`runs/`, which git
ignores). Each `attempts*.jsonl` line is one attempt: level, round, reward, the parts of the reward, the code,
and the checker's feedback. Each `*.log` is the run's console output. None of these runs wrote a
`verdicts.jsonl` (all predate e7663a3). All of them ran Qwen3-8B on the seat's own vLLM server, scored in the
NKI 0.6.0 CPU simulator. On a seat the unset target resolves to trn2 (analysis/sim_target_check.md).

| dir | what | code | seat | runs | attempts md5 | compare? |
|---|---|---|---|---|---|---|
| `baseline/` | **The baseline.** `agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5`, 10:50-12:43. L1 0/5, L2 3/5 [1.00 0.30 1.00 0.30 1.00], L3 0/5, L4 0/5 [0.62 0.62 0.50 0.62 0.62] | the seat's code at 10:50, before any team change (5ed7ec2 and later); exact commit not recorded | 116 | 5 x L1-L4 (424 attempts) | `9275898538ed1a69c28a8a5fa52a3788` | **yes**: the reference. All 424 re-grade as logged (b39989f) |
| `replica_seat119/` | A teammate's re-run of the baseline. L1 0/5, L2 2/5 [1.00 0.30 0.30 1.00 0.30], L3 0/5, L4 0/5 [0.62 x4, 0.50] | upstream 8f1ca41 | 119 | 5 x L1-L4 (420) | `a62ffe8843b59ab5b9125ecf77848b2d` | **yes**. All 420 re-grade as logged (889bea9); its `--context` is not recorded |
| `expA_L1/` | Experiment A: `enrich()` known fixes for invented names (5ed7ec2), level 1 x5. 0/5, all 0.30 | 26c43ed (has the reused-path bug; re-graded) | 116 | 5 x L1 (160) | `e883ef6fb3bf27f2b6260cdeb0093540` | **yes**, with the baseline's L1. All 160 re-grade as logged (fc4dfbe) |
| `ev3_L4/` | Feedback v3 (`MESSAGES=v3 REPAIR_PROMPT=restructure`), level 4. **Lines 1-28 are an earlier invocation that is not in `run_v3_L4.log`**; lines 29-52 are run 1 (solved in round 5, the recovery in analysis/recovery_v3_L4_run1.md), 53-84 run 2 (best 0.75), 85-100 run 3, stopped after 4 rounds. **Count it as 1/2** (runs 1-2, lines 29-84) | feedback_v3 (fc137e7); its nkibench has the allocation audit (the log contains ILLEGAL ON HARDWARE), so 26c43ed or later; exact commit not recorded | 117 | 4 segments, 2 complete runs (100) | `37836741ad1814efb52219fa63a729ec` | **yes, lines 29-84 only**. All 100 re-grade as logged (this commit; run 1 also in 9e5a7ef) |
| `teammate_all_seat117/` | A teammate's `agent.py --all`, stopped at 14:00-14:05 during run 1 (levels 1-3 started). L1 0.30, L2 1.00, L3 0.30 | **unknown** | 117 | 1 partial run | `a30b0f6c0a36bd0b8ec7d369f0b16b92` | **no**: reference only, code version unknown |
| `teammate_all_seat118/` | The same, on 118 (levels 1-2 started). L1 0.30, L2 0.30 | **unknown** | 118 | 1 partial run | `21e07d51140cee394061057c0b885834` | **no**: reference only, code version unknown |

Console logs: `baseline/run.log` `8c370e736c34d9f0b3995800d964beee`, `replica_seat119/run.log`
`c9e72a525d45c21f2939d5e233e48d9b`, `expA_L1/run_expA_L1.log` `84c0db8f318cdfa8c263b413e3358ff6`,
`ev3_L4/run_v3_L4.log` `9bcd4442d6ce24e4dfcea59aee990e7e`, `teammate_all_seat117/run.log`
`26a40363ac9cd5c3fd8ded7445967d7f`, `teammate_all_seat118/run.log` `1862ac6ded3978bbbb0d857a4687b60c`.
12 files, 2.0 MB.

"Re-grade as logged": every attempt scored again with the current checker, one fresh file path each,
allocation audit on, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, and compared with the logged reward
(`scripts/calibrate.py`). This matters because runs before c39c0ce could hold false scores (ded1ef2).

## Reproduce the tables

```bash
python scripts/report.py /tmp/out analysis/logs/baseline --baseline ""     # L1 0/5, L2 3/5, L3 0/5, L4 0/5
python scripts/report.py /tmp/out analysis/logs/ev3_L4                     # flags the extra invocation
```

## Checked for secrets before committing

Run on the 12 files before they were added (`find -print0 | xargs -0`, so it works the same in bash and
zsh: an unquoted `$F` file list silently scans nothing in zsh, which we hit once and re-ran):

```bash
S() { find analysis/logs -type f ! -name README.md -print0 | xargs -0 grep "$@"; }
S -E -i -c 'AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}|aws_secret|secret_access|SESSION_TOKEN|AWS_[A-Z_]+=|amazonaws|eks\.|hack-hyd|ap-south-2|kubeconfig|Bearer |ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|xox[bp]-|password|BEGIN [A-Z ]*PRIVATE KEY'
S -E -o -h 'https?://[^ "'"'"')]+' | sort | uniq -c
S -E -o -h '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq -c
S -E -o -h '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}|/Users/[^ "/]+|/home/[^ "/]+' | sort | uniq -c
```

Result: the first command prints a count of **0 for each of the 12 files** (a planted `AKIA...` test string is
found, so the pattern works); the only URL is `http://localhost:8000/v1` (the
seat's own model server, 6 times, once per console log); no IP addresses; the only e-mail-shaped match is
`n@nki.jit` (an escaped newline before the decorator), no home paths.
