# Changelog, decisions and git workflow

This file records every change to this project, in the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versions follow semantic versioning for the folder `projects/19-verified-rtl-agent/`. Task state lives in [`STATUS.md`](STATUS.md); the specification lives in [`DESIGN.md`](DESIGN.md).

## Git workflow

### Remotes

| Remote | Repository | Use |
|---|---|---|
| `origin` | `github.com/bhaveshgupta01/hack-the-chip` (private) | All team work. Push here |
| `upstream` | `github.com/yahavb/trainium-agent-labs` (organisers) | Fetch only. **Never push.** The push URL is disabled locally |

One-time setup on a new machine:

```bash
git clone https://github.com/bhaveshgupta01/hack-the-chip.git && cd hack-the-chip
git remote add upstream https://github.com/yahavb/trainium-agent-labs.git
git remote set-url --push upstream DISABLED_do_not_push_to_organisers
```

### Branches

| Branch | Purpose | Who pushes |
|---|---|---|
| `master` | Integration. Always passes `python checker.py --selftest` | Anyone, fast-forward or a merged PR only. Never force-push |
| `p1/<topic>` | Checker and translator work | P1 |
| `p2/<topic>` | Agent and workers work | P2 |
| `p3/<topic>` | Evaluation, runs and docs | P3 |

Workflow for every change:

```bash
git switch master && git pull --ff-only
git switch -c p1/vcd-reader                  # one branch per task id, e.g. CHK-4
# ...work only in files you own (DESIGN.md section 5)...
python checker.py --selftest
git diff --stat origin/master                # only projects/19-verified-rtl-agent/ may appear
grep -rnE 'ASIA[A-Z0-9]{12}|AWS_SECRET|AWS_SESSION_TOKEN' . && echo "STOP: credentials" || true
git commit -m "checker: read first-mismatch inputs from wave.vcd (CHK-4)"
git fetch origin && git rebase origin/master
git push -u origin p1/vcd-reader             # then merge into master (PR or fast-forward)
```

### Commit messages

The format is `<area>: <imperative summary> (<task id>)`. The areas are `checker`, `translator`, `workers`, `prompts`, `agent`, `eval`, `audit`, `report`, `runs`, `docs`, and `status`.

Examples:
- `status: claim CHK-4`
- `runs: add C-1 on heldout (EXP-4)`
- `docs: DESIGN 1.1.0, add S4 prompt (AGT-2)`

### What gets committed

- **Code, docs, `eval/*.txt` and `runs/*.jsonl` are committed.**
- **Run files are append-only.** A committed run file is never edited. A corrected result is a new run with a new rep number.
- **Never committed:** credentials, `.venv/`, model weights, the VerilogEval dataset (`setup.sh` fetches it), and `wave.vcd` or other temporary simulation output.
- **Big logs:** keep each run's `.log` under 5 MB, or commit only its `.jsonl`.

### Tags

Tag `master` when it produces a result worth keeping: `vra-0.2.0`, and so on. The final submitted state is tagged `vra-1.0.0`.

## Versioning rules

| Bump | When |
|---|---|
| MAJOR | The measured claim, the run definitions (A–E) or the held-out set changes. This invalidates earlier runs |
| MINOR | A new component, file, interface field, rule or metric |
| PATCH | Bug fixes and clarifications that don't change any interface or result |

A change to `DESIGN.md` also bumps the "Document version" line at its top.

## Decision log

Decisions are numbered and never edited. To change one, add a new decision that supersedes it.

| ID | Date | Decision | Rationale |
|---|---|---|---|
| D-001 | 2026-10-10 | The model in the loop is Qwen3-8B on our Trainium seats; run E alone uses gpt-oss-20b on the shared Trainium endpoint | The event's claim is about a small model on a chip you control |
| D-002 | 2026-10-10 | The checker is deterministic (iverilog, the testbench, yosys); no model grades attempts | An LLM grader has no ground truth |
| D-003 | 2026-10-10 | Benchmark: VerilogEval spec-to-RTL at `c498220`; 20 dev and 40 held-out problems; the 12 already-probed problems go in dev | Avoids tuning on held-out problems |
| D-004 | 2026-10-10 | Budget of 6 attempts per problem; runs A–E as defined in DESIGN 6.6; 2 reps each | Equal budgets make B vs C and C vs D comparable |
| D-005 | 2026-10-10 | Plain-Python orchestrator, no agent framework | Fewer moving parts; matches the event repo |
| D-006 | 2026-10-10 | Thinking off, no `seed`, at most 4 requests in flight per seat, different prompts for parallel attempts | Measured seat behaviour (DESIGN section 3, C5–C8) |
| D-007 | 2026-10-10 | The private team repo is the working repo; the organisers' repo is fetch-only until the final fork and PR | Keeps work private and prevents accidental pushes upstream |
| D-008 | 2026-10-10 | Add run R: S1 six times, no feedback, same budget | Separates "feedback helps" from "more attempts help"; a judge will ask |
| D-009 | 2026-10-10 | Raw feedback (run B) includes the yosys counterexample table when L3 finds one | B and C must see the same information, so B vs C measures translation only |
| D-010 | 2026-10-10 | Dev runs write to `runs/dev/`, held-out runs to `runs/` | A dev `C-1` can never be confused with, or overwrite, the held-out `C-1` |
| D-011 | 2026-10-10 | The folder is `projects/19-verified-rtl-agent/` | 19 is our team number; answers ORG-7 |
| D-012 | 2026-10-10 | Add the kernel-transfer track (DESIGN 6.9): the organisers' NKI agent with `--feedback raw / located / docs`, 3 reps each on levels 1–4 at the 2026-10-10 baseline settings; the baseline is K-located | Tests whether located feedback helps when the model lacks the language (NKI) as well as when it knows it (Verilog); costs one patched function per setting |
| D-013 | 2026-10-10 | S3 ends "Rewrite the complete module TopModule so that what the checker names is fixed. Your module must differ from the code above." instead of "Change exactly what the checker names and keep everything else identical." | Dev, all 20 problems, paired: identical copies 71% → 42%, score-raising repairs 6 → 11, passes 3 → 4 (`runs/dev/C-1.jsonl` vs `runs/dev/C-s3rewrite-full-1.jsonl`) |
| D-014 | 2026-10-10 | Copy guard: after a repair that returns the code unchanged, the next attempt is S5 (fresh from the spec, feedback as a hint); copies don't count towards the cycling stop | 42% of repairs were still copies under D-013, and each one burned budget and ended problems early on cycling |
| D-015 | 2026-10-10 | No time-based milestones and no code freezes. We improve the code whenever it helps. The one rule that stays: compare only runs made with the same `code_version`, which every attempt records | We are optimising the result, not a schedule. Comparability is what protects the numbers, and that needs a version, not a clock. Supersedes DESIGN 9 rule 6 and the STATUS milestones |
| D-016 | 2026-10-10 | Ledger: repair prompts (S3, S5) list up to 4 earlier distinct findings for the problem | Dev C-2/C-3: fixes did not stick, e.g. Prob050_kmap1 0.82 -> 0.93 -> 0.82. The event repo's own lesson: keep a ledger. Nothing new is revealed; B and C get the same mechanism |
| D-017 | 2026-10-10 | Add control Q, 6 fresh attempts rotating S1, S2, S4 with no feedback, as a wrapper (`controls/prompt_rotation.py`) so the code version stays `0aaab14` | R-1's retries returned identical code 63% of the time, so R alone can't separate the checker's information from any new prompt. Added after viewing rep 1; disclosed in NOTE |

## [Unreleased]

### Added
### Changed
### Fixed

## [0.8.0] - 2026-10-10

### Added
- Held-out runs at code `0aaab14`: A-1..3, R-1..3, Q-1..4, B-1..3, C-1..3, D-1..3 (`runs/`), each with a formal audit (`runs/*.audit.json`), plus `runs/summary.md` and `runs/taxonomy.md`.
- `controls/prompt_rotation.py`: control run Q (D-017). `report.py` reports Q and its pairs.
- `figures/attempts.svg`: problems solved vs attempts used, per run type.
- README: held-out transcript, results, problem-by-problem table, cost per solve, observations including the seat findings. NOTE: results against the analysis plan, disclosures. TAXONOMY: held-out section.
- `runs/superseded-0d4fe8d/`: the stopped 14:11 queues at code `0d4fe8d`, never read and not reported.

### Changed
- `runs/dev/` restored; it is the evidence for D-013 to D-016.

## [0.7.1] - 2026-10-10

### Fixed
- `agent.py`: `code_version` is the last commit that changed `checker.py`, `translator.py`, `prompts.py`, `agent.py`, `workers.py` or `problems.py`, and `+dirty` checks only those files. It used to be HEAD, which docs, STATUS and run-file commits move, so seats running identical code recorded different versions. The held-out code version is `0aaab14`.

## [0.7.0] - 2026-10-10

### Added
- `prompts.py`, `agent.py`: the ledger (D-016), with `ledger_items` logged per attempt and an agent self-test case.
- `translator.py`: R7, a declaration that Icarus rejects inside a block (Prob071_always_casez); R8, a signal declared twice (Prob109_fsm1). R1 for a port now says to change the port list, not add a declaration. An X or Z in the simulated output gets its own message instead of the Yosys counterexample's defined value (Prob093, Prob092; TAXONOMY finding 3).

### Fixed
- `translator.py`: reset messages for active-low resets (`resetn`, `rst_n`, ...) said "high"; they now say "low", and `negedge` for asynchronous ones (Prob073_dff16e). DESIGN 1.6.0.

## [0.6.0] - 2026-10-10

### Changed
- Removed every time-based milestone and code freeze (D-015): the STATUS milestones table, the "expected free" times, DESIGN 9 rule 6. Held-out runs simply name their code version.
- `agent.py`: the held-out check is now `check_heldout_list` and says why: every held-out run must use the same 40 problems.
- `report.py`: a "Code" column per run, flagged `MIXED` when a run file holds more than one code version.
- `translator.py`: a combinational module with no inputs gets "Your `<out>` is <value>, which is wrong" instead of "wrong for some inputs". Observed in Prob001_zero (dev), which repeated the old message 3 times. DESIGN 1.5.0.

## [0.5.0] - 2026-10-10

### Changed
- `translator.py`: R1 names the right declaration for ports and internal signals; new rules R5 (variable part-select) and R6 (bit-select of a 1-bit signal); unmatched compile errors quote the line (TRN-4). Every rule cites a dev problem. DESIGN 1.4.0.
- `report.py`: a "Copied repairs" column, counted from the code hashes of consecutive attempts.
- `dev_experiments/s3_rewrite.py`: no longer asserts the old S3 wording, which D-013 replaced, so it still runs.

### Measured (dev, context for the write-up)
- Attempt 1 uses the identical S1 prompt in every run, yet `C-1` and `C-s3rewrite-full-1` returned byte-identical code on only 12 of 20 problems. Reps are partly, not fully, deterministic: expect a small spread, and report it.
- `C-s3rewrite-full-1` took 1,039 s for 20 problems on one seat, about 52 s per problem, so a 40-problem B or C run is about 35 minutes per seat.

## [0.4.0] - 2026-10-10

### Changed
- `prompts.py`: S3's last line asks for a rewrite that must differ from the broken code (D-013). New S5, fresh with a hint (D-014).
- `agent.py`: copy guard in the depth loop; copies no longer count towards cycling (D-014). New self-test cases for both. DESIGN 1.3.0 (6.5, 6.6).

### Added
- `kernel-transfer/README.md` and `kernel-transfer/runs/K-raw*`, `K-docs*`: the NKI feedback ablation results. 0 of 36 level-runs solved across raw, located and docs; levels 1 and 3 scored 0.30 in all 9 runs (KT-3 to KT-5).
- `runs/dev/C-1.*` (original S3), `runs/dev/C-s3rewrite-1.*` (8-problem subset, stopped early), `runs/dev/C-s3rewrite-full-1.*` (all 20, D-013 evidence).

## [0.3.0] - 2026-10-10

### Added
- `kernel-transfer/kernel_agent.py`: the organisers' kernel agent with a `--feedback raw|located|docs` switch and `--kt-selftest` (KT-1, D-012). DESIGN 1.2.0, section 6.9.
- `kernel-transfer/runs/K-located*`: the 2026-10-10 seat-93 baseline (organisers' code at `8f1ca41`, `--samples 1 --repeat 3`), as the K-located setting (KT-2).

### Fixed
- `.gitignore`: re-include `runs/`. The repo root's `.gitignore` ignores every `runs/` folder, so no run file, held-out results included, would ever have been committed or reached the PR.
- `agent.py`: `code_version` ignores `runs/` and `kernel-transfer/runs/`, so committed run outputs no longer mark every run `+dirty`.
- `setup.sh`: the Icarus version check no longer prints a false "not 12.x" warning (`iverilog -V` exits non-zero under `pipefail`).

### Verified
- On seat-93 at `3722668`: `setup.sh` and all four self-tests (checker 6/6 planted bugs, translator, workers, agent) pass; every dev and held-out reference scores 1.0 through the checker. STATUS: CHK-2, CHK-3, CHK-5 to CHK-7 and AGT-1 marked DONE.

## [0.2.1] - 2026-10-10

### Changed
- Renamed the folder to `projects/19-verified-rtl-agent/` (team 19, D-011); every path in DESIGN, CHANGELOG and `setup.sh` updated. DESIGN 1.1.1.

## [0.2.0] - 2026-10-10

### Added
- `checker.py`: layers L0-L3, `CheckResult`, scoring, minimal VCD reader, `--selftest` (6 planted bugs plus tool-free parser tests), `--check`, `--refs` (CHK-1 to CHK-7).
- `translator.py`: `raw` and `located` modes, rules R1-R4 with `observed_in`, reference-leak check, `--selftest` (TRN-1 to TRN-5).
- `workers.py`: model client with the seed, thinking, context and per-seat guards; `SeatDown` on a dead engine; `--ping` (AGT-1).
- `prompts.py`: S1-S4 and context shrinking (AGT-2).
- `agent.py`: runs A-E plus R, depth and breadth solvers, stop rules, per-problem claim, JSONL ledger, freeze guard, `--resume`, `--selftest` with a fake model (AGT-3 to AGT-7).
- `problems.py` and `eval/dev.txt`, `eval/heldout.txt` (seed 20261010): 20 dev, 40 held-out (21 comb, 19 seq) (EXP-1, EXP-2).
- `audit.py` (EXP-7), `report.py` with paired comparison, taxonomy and chart (EXP-8), `setup.sh` (ENV-8).
- DESIGN 1.1.0: run R, extra `CheckResult` and JSONL fields, output locations (D-008 to D-010).

## [0.1.0] - 2026-10-10

### Added
- `DESIGN.md` 1.0.0: the technical specification (constraints, architecture, component interfaces, data formats, experiments, metrics, integrity rules, runbook, collaboration protocol).
- `STATUS.md`: the task board, with organiser questions, environment state on seat 93, and the exploratory measurements.
- `CHANGELOG.md`: this file, with the git workflow, versioning rules and decisions D-001 to D-007.
