# Status board

**Last updated:** 2026-10-10 17:50 EDT · **Design version:** [`DESIGN.md`](DESIGN.md) 1.5.0

This is the single source of truth for who is doing what. Update it in its own small commit whenever a task changes state, so parallel work never collides.

**Statuses**

| Status | Meaning |
|---|---|
| `TODO` | Not started |
| `DOING @who HH:MM` | Claimed, with the start time |
| `DONE <sha>` | Finished; the commit proves it |
| `BLOCKED: <reason>` | Can't proceed |
| `CUT` | Dropped on purpose, with the reason in Notes |

**Rules**

1. Claim a task before starting it: set Owner and `DOING`, then commit and push. If the row is already `DOING`, talk to its owner first.
2. Mark `DONE` only when the task's acceptance check from `DESIGN.md` passes. Include the commit sha.
3. Never delete a row. Mark it `CUT` and say why.
4. Results go in the Results table only from committed run files.

## Team

| Role | Person | GitHub | Seat |
|---|---|---|---|
| P1: checker and translator | | | |
| P2: agent and workers | | | |
| P3: evaluation, measurement and docs | | | |
| (seat owner) | | bhaveshgupta01 | seat-93 |
| (team seat, available for runs) | | | seat-92 |

## Seat usage

All held-out runs finished by about 17:50 on seats 90 to 93, at code `0aaab14`. Seat 94 ran a separate side project. Nothing is running for this project now.

## Organiser questions (ORG)

| ID | Question | Status | Answer |
|---|---|---|---|
| ORG-1 | Approve the Verilog agent as our own project? | DONE | Approved (team confirmation, 17:30) |
| ORG-2 | For own projects, will you supply held-back cases, or is our fixed held-out split acceptable? | TODO | |
| ORG-3 | May we call teammates' seat endpoints (`seat-N.seat:8000`) to pool our chips? | TODO | |
| ORG-4 | OK to `apt-get install` iverilog and yosys, and to fetch VerilogEval via `setup.sh`? | TODO | |
| ORG-5 | gpt-oss-20b URL for run E; is our network allowlisted? | TODO | |
| ORG-6 | PR deadline, and the judging format (length, slides or live terminal)? | TODO | |
| ORG-7 | Folder name: numbered (`03-`) or a team name? | DONE (team decision) | `projects/19-verified-rtl-agent/`, 19 = team number (D-011) |
| ORG-8 | Does the PR only need to be opened, or must it be mergeable? | TODO | |
| ORG-9 | Is it fine that Claude or Kiro helped write the harness, if the in-loop model is on Trainium? | TODO | |
| ORG-10 | Where will fresh credentials be posted after 20:00? | TODO | |
| ORG-R1 | Tell them: identical prompts in parallel return identical answers on the seats | DONE | Reported in README, Observations |
| ORG-R2 | Tell them: a request with `seed` crashes the engine (`EngineDeadError`) | DONE | Reported in README, Observations |
| ORG-R3 | Tell them: today's kernel-agent baseline solved 0 of 3 runs on every level | DONE | Reported in README, Observations |

## Environment (ENV)

| ID | Task | Owner | Status | Acceptance | Notes |
|---|---|---|---|---|---|
| ENV-1 | Seat 93: model serving | bhaveshgupta01 | DONE (verified 2026-10-10) | `curl localhost:8000/health` returns ok | Restarted once after the `seed` crash |
| ENV-2 | Seat 93: iverilog 12.0 and yosys 0.33 | bhaveshgupta01 | DONE (verified 2026-10-10) | `iverilog -V`, `yosys -V` | Installed with apt; lost if the pod is replaced |
| ENV-3 | Seat 93: VerilogEval at `/root/verilog-eval` @ `c498220` | bhaveshgupta01 | DONE (verified 2026-10-10) | 156 `*_prompt.txt` files | Outside `/workspace`, so `setup.sh` must recreate it |
| ENV-4 | Cross-seat DNS and HTTP | bhaveshgupta01 | DONE (self only) | `seat-93.seat:8000/v1/models` returns 200 | Not yet tested seat to seat |
| ENV-5 | Seat B: `serve.sh`, tools, VerilogEval | | TODO | ENV-1 to ENV-3 checks pass on seat B | |
| ENV-6 | Seat C: `serve.sh`, tools, VerilogEval | | TODO | Same checks on seat C | |
| ENV-7 | Seat-to-seat call between teammates | | TODO | Seat A reaches seat B and seat C on `/v1/models` | Needs ORG-3 |
| ENV-8 | `setup.sh` written and tested on a second pod | parvapatel | DONE (verified on a fresh seat-92 pod 13:19) | Fresh pod goes to a passing `checker.py --selftest` | All four self-tests pass on seat-92. The false "not 12.x" warning (pipefail on `iverilog -V`) is fixed |
| ENV-9 | Seat 92: `serve.sh`, tools, VerilogEval, team repo at `1d47f65` | bhaveshgupta01 | DOING @bhaveshgupta01 13:19 | Same checks as ENV-1 to ENV-3 | Tools and self-tests done; model compiling |

## Checker (CHK), owner P1

| ID | Task | Owner | Status | Acceptance |
|---|---|---|---|---|
| CHK-1 | L0: code extraction and port check | parvapatel | DONE 4e51913 | DESIGN 6.2, L0 row |
| CHK-2 | L1: iverilog compile | parvapatel | DONE 4e51913 (verified on seat-93 13:07) | Bug 3 caught with `is not a valid l-value` |
| CHK-3 | L2: simulate; parse `Mismatches` and `Hint` lines | parvapatel | DONE 4e51913 (verified on seat-93 13:07) | Bugs 4 and 5 caught |
| CHK-4 | VCD reader: input values at the first mismatch | parvapatel | DONE 4e51913 | Inputs reported at `first_mismatch.time_ps` |
| CHK-5 | L3: yosys counterexample (comb) | parvapatel | DONE 4e51913 (verified on seat-93 13:07) | Bug 2 gives `c=1` (pipeline verified by hand on seat 93) |
| CHK-6 | `CheckResult` and scoring | parvapatel | DONE 4e51913 (verified on seat-93 13:07) | Reference scores 1.0 |
| CHK-7 | `--selftest` with 6 planted cases | parvapatel | DONE 4e51913 (verified on seat-93 13:07) | All 6 pass in a seat pod |

## Translator (TRN), owner P1

| ID | Task | Owner | Status | Acceptance |
|---|---|---|---|---|
| TRN-1 | `raw` mode | parvapatel | DONE 4e51913 | Returns `result.raw`, at most 1,500 characters |
| TRN-2 | Compile rules R1 to R4 | parvapatel | DONE 4e51913 | Each matches its observed error |
| TRN-3 | Mismatch messages (cycle, inputs, counterexample) | parvapatel | DONE 4e51913 | Matches DESIGN 6.3 patterns |
| TRN-4 | Rules from dev failures, with `observed_in` | parvapatel | DONE 47a5277 (on top of D-013/D-014): R1 for internal signals, R5 part-select, R6 scalar bit-select, unmatched errors quote the line; each cites a dev problem. DESIGN 1.4.0 | Every rule cites a dev problem |
| TRN-5 | Leak check: no `_ref.sv` lines in any message | parvapatel | DONE 4e51913 | Unit check passes |

## Agent (AGT), owner P2

| ID | Task | Owner | Status | Acceptance |
|---|---|---|---|---|
| AGT-1 | `workers.py` with guards | parvapatel | DONE 3e89968 (verified on seat-93 13:07) | A body containing `seed` raises; `finish_reason` logged; `SeatDown` on HTTP 500 |
| AGT-2 | `prompts.py`: S1, S2, S3, S4 | parvapatel | DONE 3e89968 | Text matches DESIGN 6.5 |
| AGT-3 | Runs A, B and C (sequential depth) | parvapatel | DOING @parvapatel 12:55 (3e89968); dev run C rep 1 running on seat-93 since 13:04 (`runs/dev/C-1.jsonl`) | Run C on dev completes end to end |
| AGT-4 | Run D (3 seats, breadth) | parvapatel | DOING @parvapatel 12:55 (3e89968) | Needs ENV-7 |
| AGT-5 | Run E (gpt-oss-20b) | parvapatel | DOING @parvapatel 12:55 (3e89968) | Needs ORG-5 |
| AGT-6 | JSONL logger | parvapatel | DONE 3e89968 | Every field in DESIGN 7.1 present |
| AGT-7 | Stop rules and per-problem claim | parvapatel | DONE 3e89968 | `stop_reason` and `claim` on each problem's last attempt |
| AGT-8 | **No-op repairs**, fixed by D-013 (S3 wording) and D-014 (copy guard, S5). Dev, same 20 problems: C-1 original 3 passes, 0 rescued, 71% copies, mean best 0.52 → C-2 (`runs/dev/C-2.jsonl`, d6cc379) 5 passes, 2 rescued (`Prob001_zero`, `Prob109_fsm1`), 35% copies, mean best 0.69, cycling stops 16 → 3 | bhaveshgupta01 (for parvapatel) | DONE d6cc379 | Copies down and rescues up on dev |

## Evaluation and measurement (EXP), owner P3

| ID | Task | Owner | Status | Acceptance |
|---|---|---|---|---|
| EXP-1 | `problems.py` and split; dev includes the 12 probed problems | parvapatel | DONE e1e1a88 | Deterministic; no overlap |
| EXP-2 | Commit `eval/heldout.txt`, the fixed held-out list | parvapatel | DONE e1e1a88 | Committed; every held-out run uses this list |
| EXP-3 | Run A, reps 1 and 2 | bhaveshgupta01 | DONE: A-1, A-2, A-3 at `0aaab14` | `runs/A-1.jsonl`, `runs/A-2.jsonl` |
| EXP-4 | Runs B and C, reps 1 and 2 | bhaveshgupta01 | DONE: B-1..3 and C-1..3 at `0aaab14`; rep 3 on swapped seats | Four run files |
| EXP-5 | Run D, reps 1 and 2 | bhaveshgupta01 | DONE: D-1..3 at `0aaab14`, 3 strategies on one chip | Two run files |
| EXP-6 | Run E, reps 1 and 2 (optional) |  | CUT: no gpt-oss URL (ORG-5) | First to cut if late |
| EXP-7 | `audit.py` calibration | bhaveshgupta01 | DONE: every held-out run audited (`runs/*.audit.json`) | The categories sum to the comb passes |
| EXP-8 | `report.py`: summary and chart | bhaveshgupta01 | DONE: `runs/summary.md`, Q added to `report.py` | `runs/summary.md` and the chart |
| EXP-9 | `TAXONOMY.md` | bhaveshgupta01 | DONE: held-out section in `TAXONOMY.md`, `runs/taxonomy.md` | Every failed attempt classified, with counts |
| EXP-10 | Control Q: 6 fresh attempts rotating S1, S2, S4, no feedback (`controls/prompt_rotation.py`), reps 1-4 | bhaveshgupta01 | DONE: Q-1..4 at `0aaab14` | Q matches B and C problem by problem; beats R |

## Docs and PR (DOC), owner P3

| ID | Task | Owner | Status | Acceptance |
|---|---|---|---|---|
| DOC-1 | `DESIGN.md` 1.0.0, `STATUS.md`, `CHANGELOG.md` | bhaveshgupta01 | DONE 2e0b0c1 | Pushed to the team repo |
| DOC-2 | `README.md` with a real transcript at the top | parvapatel, bhaveshgupta01 | DONE: held-out transcript, results, observations | Transcript taken from a committed run |
| DOC-3 | `NOTE.md`, one page | parvapatel, bhaveshgupta01 | DONE: results against the analysis plan, disclosures | Runs, spread, simulator vs formal labelled |
| DOC-4 | Fork the organisers' repo and open the PR (this folder only) | bhaveshgupta01 | DOING: PR from the fork after the final runs | `git diff --stat` shows only this folder |

**Verification so far (12:55, laptop, no iverilog/yosys):** `checker.py --selftest --parsers-only`, `translator.py --selftest`, `workers.py --selftest` and `agent.py --selftest` all pass. Tasks still `DOING` above have code in the named commit and need their acceptance check in a seat pod: run `./setup.sh` (it ends with every self-test), then `python checker.py --refs eval/dev.txt`, then a dev run C.

## Kernel transfer (KT), owner bhaveshgupta01

DESIGN 6.9: the organisers' NKI kernel agent with one feedback switch. Seat 93, `--all --rounds 8 --samples 1 --context 8192 --repeat 3`.

| ID | Task | Owner | Status | Acceptance |
|---|---|---|---|---|
| KT-1 | `kernel-transfer/kernel_agent.py`: `--feedback` with `raw`, `located` or `docs` | bhaveshgupta01 | DONE 829b8e1 | `--kt-selftest` passes on seat-93 (13:07) |
| KT-2 | K-located: the 2026-10-10 baseline, same code and settings | bhaveshgupta01 | DONE | `runs/K-located*`: 0/3 solved on every level |
| KT-3 | K-raw, 3 reps | bhaveshgupta01 | DONE (this commit): 0/3 on every level | `runs/K-raw.log` per-level summary |
| KT-4 | K-docs, 3 reps | bhaveshgupta01 | DONE (this commit): 0/3 on every level | `runs/K-docs.log` per-level summary |
| KT-5 | Compare the three settings and add them to the README | bhaveshgupta01 | DONE (this commit): `kernel-transfer/README.md` | A table per level, with every rep shown |

## Results (fill only from committed run files)

| Run | Rep | Solved / 40 | Median attempts to pass | Tokens per solve | Seconds per solve | Truncations | Escapes | Run file |
|---|---|---|---|---|---|---|---|---|
| A | 1 | | | | | | | |
| A | 2 | | | | | | | |
| B | 1 | | | | | | | |
| B | 2 | | | | | | | |
| C | 1 | | | | | | | |
| C | 2 | | | | | | | |
| D | 1 | | | | | | | |
| D | 2 | | | | | | | |
| E | 1 | | | | | | | |
| E | 2 | | | | | | | |

## Exploratory measurements (context, not results)

These were measured on seat 93 before the split existed. They are not part of the evaluation, and all 12 problems below go in the dev set.

- **One-shot probe:** Qwen3-8B on 12 VerilogEval problems passed 2 of 12 (`Prob027_fadd`, `Prob035_count1to10`).
  - 6 compile errors: `Prob001`, `Prob054`, `Prob071`, `Prob109`, `Prob142`, and `Prob156`, which was cut off at the 1,500-token cap.
  - 4 wrong outputs: `Prob050`, `Prob021`, `Prob086`, `Prob128`.
  - Script: `/root/ve_probe.py` in seat 93.
- **Seat behaviour:**
  - The kernel agent's 4 parallel samples were identical in every round.
  - Rounds took about 50 s with 4 samples and about 20 s with 1.
  - `seed` crashed the engine.
- **Kernel-agent baseline**, 3 runs with `--samples 1`: 0 of 3 solved on every level. Level 1: 0.30; level 2: best 0.50; level 3: 0.30; level 4: best 0.62.
