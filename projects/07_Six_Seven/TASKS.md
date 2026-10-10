# Task board — who did what

This is the task board our team used during the day, copied as-is at submission time. The **Who**
column records who did each task. Links to team-only files (`TEAM.md`, `SERVER.md`) point to our
working repo, not this folder; the seat guide is [`SETUP.md`](SETUP.md) here.

---

# VeriLoop — task board

**Need the server?** See [`SERVER.md`](SERVER.md).

**How to pick up a task:** write your name in the **Who** column and set **Status** to `doing`, then
commit and push *right away* (so two people don't grab the same one). When finished, set `done`, push.
Edit only your own row. One task at a time. If GitHub says there is a conflict, `git pull --rebase`
and try again.

**Every task owns specific files** (the **Files** column). Only touch your files — that is what lets six
people work at once without breaking each other's work. Need a change in someone else's file? Ask them.

Status: `todo` · `doing` · `done` · `blocked`. 🟢 = no coding needed.

---

## Start here — six tasks that can begin right now, in parallel

| task | what | needs |
|---|---|---|
| **C1** | the checker core | the level contract below |
| **L1** | level 1 (mux) + level 2 (adder) | the level contract below |
| **L3** | level 3 (counter) | the level contract below |
| **L4** | level 4 (state machine) | the level contract below |
| **A1** | the model loop | nothing |
| **W1** 🟢 | organiser questions + checker write-up start | nothing |

Everything else unlocks as these finish.

---

## Where every file goes

```
veriloop/
├── checker.py              C1–C3   compile, simulate, compare, score, feedback A/B/C
├── selftest.py             C4      every good.v passes, every bad_*.v fails
├── agent.py                A1–A3   prompt → model → Verilog → checker → feedback → retry, log
├── run_experiment.py       A4      levels × feedback A/B/C × N runs → results/*.jsonl + summary
├── plot.py                 M2      results → the graph
└── levels/
    ├── README.md           C1      the level contract (copied from this file)
    ├── 01_mux/             L1
    ├── 02_adder/           L1
    ├── 03_counter/         L3
    ├── 04_traffic_fsm/     L4
    ├── 05_mac/             L5
    ├── 06_fifo/            L6
    ├── 07_systolic_2x2/    L7
    └── 08_pipelined_mac/   L8
results/
├── <date>_<who>_<what>.jsonl   M1   raw attempt logs (every attempt, its score, its feedback)
├── summary.csv                 M1   one row per level × feedback level × run
├── graph.png                   M2   solve rate vs feedback quality
└── TAXONOMY.md                 M3   the model's mistakes, grouped and counted
SUBMISSION.md                   W2   the one-page note (required)
CHECKER.md                      W3   what the checker accepts and rejects, and why (required)
DEMO.md                         W4   the 2-minute demo script
```

## The level contract — every level folder has exactly these files

So the levels and the checker can be written at the same time, every `veriloop/levels/NN_name/` folder
contains:

| file | what it is |
|---|---|
| `spec.txt` | what the model is told: 3–6 plain lines, **including the exact module name and port list** (names, widths, directions). Nothing else — no hints about the solution. |
| `reference.py` | the correct behaviour in Python (see below) |
| `good.v` | a correct design you wrote by hand — **must pass** |
| `bad_1.v`, `bad_2.v`, `bad_3.v` | at least three deliberately broken designs, each with a top comment saying what is wrong (e.g. `// carry out is never set`) — **must fail** |

`reference.py` defines:

```python
MODULE  = "counter8"                     # must match spec.txt and good.v
INPUTS  = {"reset": 1, "en": 1}          # name -> bit width (do NOT list clk)
OUTPUTS = {"count": 8}                   # name -> bit width
CLOCKED = True                           # True: the test bench drives clk; False: combinational

def vectors():
    """The test inputs: a list of dicts, one per step (one per clock cycle if CLOCKED).
    Combinational: every input combination if small, else edge cases + random.
    Clocked: start with reset, then cover enable off, wrap-around, reset mid-run, etc."""

def reference(vectors):
    """The correct outputs: a list of dicts, one per step.
    CLOCKED: the output value just after that cycle's rising clock edge."""
```

The checker (C1–C2) turns this into a Verilog test bench automatically — level authors never write test
benches.

---

## Phase 3 — the checker

| ID | task | done when | files | depends on | who | status |
|---|---|---|---|---|---|---|
| S1 | Install the simulator on seat 7 and write the one-line install for everyone | `iverilog -V` works on seat 7; the command is in SERVER.md | `SERVER.md` (install section only) | — | Krish (via Claude Code) | done — Icarus Verilog 12.0 on seat 7; tested a 4-bit adder |
| C1 | **Checker core**: compile a `.v` with `iverilog`; on a compile error, return the line and the message in plain words | a broken `.v` gives "line 4: …" not a raw dump | `veriloop/checker.py`, `veriloop/levels/README.md` | — | Krish (via Claude Code) | done — 12/12 compile cases pass on seat 7 and Mac (`python veriloop/tests/test_compile.py`) |
| C2 | **Test bench + compare**: generate a test bench from `reference.py`, run it with `vvp`, parse the outputs, compare with `reference()` | `checker.py levels/01_mux good.v` → pass; `bad_1.v` → fail | `veriloop/checker.py` | C1, L1 | Krish (via Claude Code) | done — 11/11 simulation cases pass on seat 7 and Mac (`python veriloop/tests/test_simulate.py`) |
| C3 | **Score + feedback A/B/C**: score 0–1 (compiles / runs / fraction of steps correct); feedback A = pass/fail, B = "x of N steps wrong", C = first wrong step: inputs, signal, expected vs got, cycle number. **Never print the full answer.** | the three messages for a `bad_*.v` read correctly | `veriloop/checker.py` | C2 | Krish (via Claude Code) | done — `checker.grade()`; 9/9 feedback cases pass on seat 7 and Mac (`python veriloop/tests/test_feedback.py`) |
| C4 | **Checker self-test**: for every level folder, `good.v` must pass and every `bad_*.v` must fail | `python veriloop/selftest.py` prints PASS for all levels written so far | `veriloop/selftest.py` | C3 | Krish (via Claude Code) | done — `python veriloop/selftest.py [LEVEL]`; fixtures 2/2 PASS; catches a missing port in the spec and an uncaught bad design |

## Levels — one person per level, all in parallel

Each: write `spec.txt`, `reference.py`, `good.v`, `bad_1..3.v` in the folder; **done when
`python veriloop/selftest.py veriloop/levels/NN_name` says PASS.** Copy the examples in `veriloop/tests/sim_fixtures/`.

| ID | level | the tricky part to test | folder | who | status |
|---|---|---|---|---|---|
| L1 | 1 — 4-to-1 multiplexer **and** 2 — 4-bit adder with carry out | adder: carry out on 15 + 1 | `veriloop/levels/01_mux/`, `veriloop/levels/02_adder/` | Krish (via Claude Code) | done — 01_mux (64 tests) + 02_adder (256 tests); selftest PASS |
| L3 | 3 — 8-bit counter, enable + synchronous reset | wrap 255 → 0; enable off holds; reset mid-count | `veriloop/levels/03_counter/` | Krish (via Claude Code) | done — 272 tests; selftest PASS |
| L4 | 4 — traffic-light state machine (green 3 cycles → yellow 1 → red 2 → …) | exact cycle of each change; reset | `veriloop/levels/04_traffic_fsm/` | Krish (via Claude Code) | done — selftest PASS (256 tests; 5 bad designs all caught in simulation); experiment running on seat 7 |
| L5 | 5 — multiply-accumulate cell: `acc <= acc + a*b`, clear input | overflow width; clear | `veriloop/levels/05_mac/` | Krish (via Claude Code) | done — signed MAC, 242 tests; 5 bad designs (unsigned multiply, no sign extension, clear ignored, saturates, en ignored) all caught; selftest PASS |
| L6 | 6 — 4-entry FIFO with full/empty flags | read + write in the same cycle; full and empty edges | `veriloop/levels/06_fifo/` | Krish (via Claude Code) | done — 174 tests; 5 bad designs (writes when full, write+read at full, dout a cycle late, if/else drops one of read+write, stale dout when empty) all caught; selftest PASS |
| L7 | 7 — 2×2 systolic array of MAC cells | data passes to neighbours each cycle; result timing | `veriloop/levels/07_systolic_2x2/` | | cut — not enough chip time; depth over breadth (6 levels done) |
| L8 | 8 — 2-stage pipelined MAC | answer arrives one cycle later; stages stay in step | `veriloop/levels/08_pipelined_mac/` | | cut — not enough chip time; depth over breadth (6 levels done) |

## Phase 4 — the model loop

| ID | task | done when | files | depends on | who | status |
|---|---|---|---|---|---|---|
| A1 | **Model call**: send `spec.txt` to Qwen3-8B on `http://localhost:8000/v1` (thinking **off**, short prompt), pull the Verilog out of the reply | prints one Verilog module for level 1 | `veriloop/agent.py` | — | Krish (via Claude Code) | done — `agent.ask()` + `extract_verilog()`; live on seat 7, thinking off, token counts logged |
| A2 | **The loop**: up to 8 rounds × 4 attempts; keep the best; send the chosen feedback level (flag `--feedback A/B/C`) with the previous design; log every attempt as one JSON line (level, round, feedback level, Verilog, score, feedback) | one level runs end to end and writes a log | `veriloop/agent.py` | A1, C3 | Krish (via Claude Code) | done — `python veriloop/agent.py --level DIR --feedback A|B|C --log FILE`; live on seat 7: counter and adder solved on round 0; `--offline` tests the loop without a model |
| A3 | **Calibration**: at the end of each level the agent states `solved` or `could not verify`; record it next to the checker's real verdict | log has both; mismatches counted | `veriloop/agent.py`, `veriloop/calibrate.py` | A2 | Krish (via Claude Code) | done — agent claims SOLVED only when the checker passed it, else COULD NOT VERIFY; `python veriloop/calibrate.py results/*_attempts.jsonl --out results/calibration.csv` asks the model its confidence per design (fresh chat, no verdict shown) and compares with the checker. Run it after the experiments **Run on 80 designs: correct 91.5 vs wrong 60.1 mean confidence; 16% confidently wrong; Brier 0.334.** |
| A4 | **Experiment runner**: levels × feedback A/B/C × N runs, interleaved; one summary + one attempt log per person (`--tag`); resumes after a dropped connection; refuses levels that fail the self-test | one command runs a full grid | `veriloop/run_experiment.py` | A2 | Krish (via Claude Code) | done — live on seat 7 (6 runs); `python veriloop/run_experiment.py --summary` prints the table |

## Phase 5 — measure

| ID | task | done when | files | depends on | who | status |
|---|---|---|---|---|---|---|
| M1 | **Run the experiments** on your own seat: `nohup python run_experiment.py --tag <you> --levels levels/NN_a levels/NN_b --runs 5 > run.log 2>&1 < /dev/null &` — split levels between seats so each seat runs different ones | `results/<date>_<you>_summary.csv` + `_attempts.jsonl` copied into the repo's `results/` | `results/` (new files only, named `<date>_<who>_<what>`) | A4 | Krish (via Claude Code) | running on 8 seats; `bash veriloop/collect.sh SEATS...` pulls results in; levels 1–3 done (72 runs) |
| M2 | **Results table + graph**: solve rate per level × A/B/C, with spread; fill the table in TEAM.md | `results/graph.png` exists; TEAM.md table filled | `veriloop/plot.py`, `results/graph.png`, `TEAM.md` (results table only) | M1 | Krish (via Claude Code) | script done (`python veriloop/plot.py` → results/graph.png + table.md); final graph after level 4 |
| M3 🟢 | **Failure taxonomy**: read the failed attempts in the logs; group the mistakes (e.g. "reset is asynchronous", "off by one cycle", "wrong bit width", "does not compile") and count them per level | `results/TAXONOMY.md` with a table of types × counts and one example each | `results/TAXONOMY.md` | first logs from A2 | Krish (via Claude Code) | script ready — `python veriloop/taxonomy.py` → results/TAXONOMY.md; run after the last runs, team checks the examples |
| M4 🟢 | **Pick the demo case**: find a run where an early round fails and a later round succeeds *because of* feedback C | case written in `DEMO.md` with the round-by-round story | `DEMO.md` | M1 | | todo |

## Phase 6 — write-up and submit

| ID | task | done when | files | depends on | who | status |
|---|---|---|---|---|---|---|
| W1 🟢 | **Ask the organisers**: submission deadline, how to submit, must the repo be public — write answers in TEAM.md | the three answers in TEAM.md "What the organisers require" | `TEAM.md` (that section only) | — | | todo |
| W2 🟢 | **One-page note** (required): what we ran, on what hardware, how many runs, the spread, the graph, the limits | `SUBMISSION.md`, one page | `SUBMISSION.md` | M2 | Krish (via Claude Code) | drafting — `SUBMISSION.md`; numbers filled when runs finish |
| W3 🟢 | **Checker write-up** (required): what it accepts and rejects and why; the three feedback levels; how we proved it (good/bad designs) | `CHECKER.md` | `CHECKER.md` | C3 | Krish (via Claude Code) | done (draft) — `CHECKER.md`; team: please read and check |
| W4 🟢 | **Demo script + rehearsal**: 2 minutes — the failure → the feedback → the fix → the graph | `DEMO.md` final; rehearsed once | `DEMO.md` | M4 | Krish (via Claude Code) | drafting — `DEMO.md`; live case picked from real runs (M4) |
| W5 | **Final check + submit**: everything in the requirements list in TEAM.md is ticked; submit | submitted | `TEAM.md` | all | | todo |

---

## Rules that keep six people from colliding

- **Only edit the files in your task's Files column.** New files are fine if they are inside your folder.
- `git pull --rebase` before you start, commit small, push often.
- Results are **new files only** in `results/` — never edit someone else's log.
- Never commit the AWS credentials. Never delete a teammate's file.
- Only real numbers from real runs, with the number of runs stated.
