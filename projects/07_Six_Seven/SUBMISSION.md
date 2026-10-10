# VeriLoop — one-page note

**Team 7, Six Seven** — Krish Mehta, Claude Code, Dhriti Vaidya, Smruthi Ramesh, Manish Reddy, Bhagavan Madala,
Ankit Singh (emails and contributions in `README.md`) · Hack the Chip, NYU × Annapurna Labs, 2026-10-10

## The question

A small model writes digital hardware in Verilog; a simulator grades every design; the grade goes back and
the model tries again. **Does the quality of that feedback decide whether the loop succeeds?** Everything is
held fixed — model, prompt, levels, rounds — except the feedback line:

| | the model is told |
|---|---|
| **A** | `FAIL: the design is not correct.` |
| **B** | `FAIL: 263 of 272 tests give a wrong output.` |
| **C** | where it first goes wrong, like a waveform: `cycle 6: en=0 -> count = 6, expected 5` |
| **D** | C plus the cause, in spec terms: `your phases last RED 4, GREEN 5, YELLOW 3; the spec says 3, 4, 2` |

## What we ran, on what

- **Model:** Qwen3-8B on **AWS Trainium2** (vLLM, one chip per run, tensor-parallel 2), thinking off,
  temperature 1.0. **Checker:** Icarus Verilog simulation against a Python reference — proof in `CHECKER.md`.
- **Six levels**, easy → hard: mux · 4-bit adder · 8-bit counter · traffic-light state machine with a
  pedestrian button · signed multiply-accumulate (MAC) cell · 4-entry FIFO. Each is proven before use: a
  correct design scores 1.0 and every one of 27 deliberately broken designs is caught.
- **One run** = up to 6 rounds × 4 attempts, stopping when a design passes every test. **141 counted runs.**

## Results — solved / runs · mean best score (min–max)

| level | A | B | C | D |
|---|---|---|---|---|
| mux, adder, counter | 24/24, round 1 | 24/24, round 1 | 24/24, round 1 | – |
| traffic light | 0/8 · 0.65 | 0/8 · 0.65 | 0/8 · 0.69 | 0/6 · 0.66 |
| traffic light, 12 rounds | 0/5 · 0.69 | 0/5 · 0.64 | 0/5 · 0.65 | – |
| MAC cell | 3/3, round 2 | 3/3, round 2 | 3/3, round 2 | – |
| FIFO | 0/3 · 0.34 | 0/3 · 0.35 | 0/3 · 0.33 | 0/6 · 0.36 |

Graph: `results/graph.png` · full table with ranges: `results/table.md` · every attempt, with the design and the
feedback sent: `results/*_attempts.jsonl`. D was added after A–C, for the two levels nobody solved.

## What we found

1. **Feedback quality did not decide success here.** Easy blocks were solved on the first try with any
   feedback (72/72); the MAC cell was fixed on round 2 with A, B and C alike. The traffic light and the FIFO
   were **never solved** — not with C (*where* it goes wrong), not with 12 rounds instead of 6, and not with D
   (*why*). C's small lead on the traffic light at 6 rounds (0.69 vs 0.65, inside the spread) vanished at 12.
   In one FIFO run the model received the same correct diagnosis four rounds running and never changed the
   flagged behaviour. **For this 8B model the limit is turning a correct diagnosis into a correct edit.**
2. **The checker is right; the model's idea of the design is wrong** (`results/TAXONOMY.md`, 1,836 failed
   attempts, 295 distinct designs). Traffic light: in 66% the first error is a light held too long; the single
   most common design counts from 0 to the full length, so **every phase runs one cycle long** (19% of A–C
   failures), and 8% skip counting the reset cycle as RED's first. MAC: every round-1 failure widened port `b`
   to 9 bits, "to handle signed multiplication" in its own comment. FIFO: 91% put the wrong byte on `dout`
   while writing; the most common design also flags `full`/`empty` a cycle late and loses a count update when
   reading and writing together.
3. **The model does not know when it is wrong.** Asked its confidence on 80 of its own designs (fresh chat,
   verdict hidden): designs it rated 80–100 were correct only **52%** of the time; **16%** of wrong designs got
   ≥ 80; Brier score 0.334, worse than always answering 50. So only the checker may say "solved" — our agent
   reports `SOLVED` only when every test passed, otherwise `COULD NOT VERIFY`.
4. **Two traps that would have invalidated the experiment.** (a) The model goes in circles: in the pilot it
   resent the identical design for 6 rounds; the loop now tells it when it repeats itself. (b) "8 runs" were
   really 2–3: the model is nearly deterministic for code, so separate chips produced byte-identical designs.
   We fingerprinted every design, added a per-run prompt tag, and count only runs made after the fix.

## Honesty notes

- Every number is a rate over independent runs with n shown — never a best run. The MAC and FIFO cells have
  small n (3 per feedback level for A–C).
- Kept in `results/excluded/` but not counted: the pilot (old loop) and the level-4 attempts made before the
  independence fix.
- "Solved" means passing our tests, not a proof of correctness; timing, area and power are not checked.

## Reproduce

```bash
python veriloop/selftest.py                    # proves all six levels
python veriloop/run_experiment.py --tag me --levels veriloop/levels/04_traffic_fsm --feedback A B C D --runs 1
python veriloop/plot.py                        # table + graph from results/
```
Seat setup (start the model, install Icarus Verilog): `SETUP.md`.
