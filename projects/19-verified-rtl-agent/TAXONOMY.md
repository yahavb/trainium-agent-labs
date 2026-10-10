# Failure taxonomy

**Status:** dev sections from the dev runs; the held-out section below is generated from the final held-out run files (all at code `0aaab14`).

## How a failure gets its label

Every failed attempt gets exactly one label, read from the checker's result in a committed run file. No model is involved, and the same run file always gives the same counts.

1. **Mode:** `report.failure_mode()`, the same labels as `runs/summary.md`.
2. **Why it is still wrong:** for the two "wrong" modes only, `taxonomy.py` adds a second label. The first match wins:

| Reason | Meaning | Read from |
|---|---|---|
| copied | The repair returned the code it was asked to fix, so nothing changed | `code_sha1` equal to the previous attempt's (same rule as `report.copies`) |
| output X or Z | The simulator saw X or Z in at least one bit of the first wrong output: not assigned on every path, not initialised, or driven twice | `first_mismatch.dut` |
| reset behaviour | The testbench printed a reset hint | `tb_hints` |
| comb, counterexample | Yosys found an input where the design is wrong | `counterexample` |
| comb, no counterexample | The formal check was unsupported or not run | `counterexample` empty |
| seq, wrong from start | First mismatch in clock cycle 0 or 1: the start or reset state | `first_mismatch.cycle` |
| seq, wrong later | First mismatch after cycle 1: the logic that updates state | `first_mismatch.cycle` |

Regenerate with:

```bash
python taxonomy.py runs/dev/C-1.jsonl runs/dev/C-s3rewrite-full-1.jsonl   # dev, below
python taxonomy.py --dir runs --out runs/taxonomy.md                     # held-out, after the final runs
python taxonomy.py runs/dev/C-1.jsonl --show Prob050_kmap1:1             # read one attempt in full
```

`taxonomy.py` takes file names because `report.py` only loads files named `<RUN>-<N>.jsonl`, so it skips `C-s3rewrite-full-1.jsonl`.

## What the dev failures show

Dev runs, one rep each, with two different S3 wordings. **These are context for the method, not results.**

1. **"Wrong output" mostly means no repair happened.** In `C-1`, 17 of the 37 "wrong" attempts are byte-identical copies of the attempt before them. With the rewrite wording (`C-s3rewrite-full-1`, D-013) it is 12 of 45. Copies hide inside compile modes too: 6 of 9 R1 attempts in `C-1` were unchanged code. Across all repairs, 30 of 42 were copies in `C-1` and 21 of 50 in the rewrite run. This is AGT-8 seen from the failure side.
2. **A named compile rule is not always applied.** R1 (`is not a valid l-value`) failed every attempt on `Prob100_fsm3comb`, `Prob109_fsm1` and `Prob142_lemmings2`, in both runs. R5 failed all 6 attempts on `Prob021_mux256to1v` in the rewrite run. The message names the fix, and the model still doesn't make it.
3. **X outputs get the wrong message.** 9 attempts on 7 problems had X or Z in the first wrong output (`Prob001`, `054`, `073`, `092`, `093`, `098`, `156`). The translator has no rule for X. In `Prob093_ece241_2014_q3` (`C-1`), the simulator saw `4'bxxxx`, but the message quoted the Yosys counterexample: "your `mux_in` is 4'b1100". `Prob092_gatesv100` (rewrite) shows the same pattern. The model is told a defined wrong value while the testbench sees an undriven one, and neither problem recovered.
4. **A single counterexample was not enough for two comb problems.** In the rewrite run, `Prob050_kmap1` and `Prob093_ece241_2014_q3` each got a counterexample on all 6 attempts, changed their code each time, and never passed.
5. **Sequential failures are in the update logic, not the start state.** No sequential attempt first went wrong in cycle 0 or 1. Excluding copies and reset hints, all 11 (`C-1`) and 13 (rewrite) went wrong later.
6. **To check (P1, CHK-4):** on `Prob156_review2015_fancytimer`, the message reports the inputs as `data=x, ack=x` at cycle 13. Either the testbench drives X there, or the VCD reader misses those signals.

## Read by hand

The labels above say *where* an attempt failed. These rows are read by a person to say *what* went wrong in the design. Use `--show PROBLEM:ATTEMPT` on the named run file.

| Run file | Attempt | Label | What actually went wrong (reader) |
|---|---|---|---|
| `C-1` | `Prob093_ece241_2014_q3:1` | output X or Z | |
| `C-1` | `Prob066_edgecapture:1` | seq, wrong later | |
| `C-1` | `Prob100_fsm3comb:1` | compile R1 | |
| `C-s3rewrite-full-1` | `Prob050_kmap1:1` | comb, counterexample | |
| `C-s3rewrite-full-1` | `Prob021_mux256to1v:1` | compile R5 | |

## Dev counts

#### `runs/dev/C-1.jsonl`

62 attempts, 59 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (seq) | 21 | 7 |
| wrong output (comb) | 13 | 8 |
| compile: R1 is not a valid l-value | 9 | 6 |
| compile: R5 is not allowed in a constant expression | 3 | 1 |
| compile: R3 break statements not supported | 3 | 2 |
| compile: R6 can not select part of scalar | 3 | 2 |
| wrong: reset behaviour | 3 | 2 |
| compile: syntax error | 3 | 2 |
| compile: R2 always process does not have any delay | 1 | 0 |

**Inside the 37 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 17 |
| output X or Z | 5 |
| reset behaviour | 1 |
| comb, counterexample | 2 |
| comb, no counterexample | 1 |
| seq, wrong later | 11 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob001_zero` (comb): R2 → X → copy → copy  **FAIL, cycling**
- `Prob012_xnorgate` (comb): PASS  **PASS, passed**
- `Prob021_mux256to1v` (comb): R5 → R5 (copy) → R5  **FAIL, cycling**
- `Prob027_fadd` (comb): PASS  **PASS, passed**
- `Prob035_count1to10` (seq): PASS  **PASS, passed**
- `Prob050_kmap1` (comb): cex → cex → copy → copy  **FAIL, cycling**
- `Prob054_edgedetect` (seq): seq@later → seq@later → copy → copy  **FAIL, cycling**
- `Prob066_edgecapture` (seq): seq@later → seq@later → seq@later → copy → copy  **FAIL, cycling**
- `Prob071_always_casez` (comb): R3 → R3 (copy) → R3 (copy)  **FAIL, cycling**
- `Prob073_dff16e` (seq): R6 → R6 (copy) → R6 (copy)  **FAIL, cycling**
- `Prob086_lfsr5` (seq): reset → copy → copy  **FAIL, cycling**
- `Prob092_gatesv100` (comb): syntax → syntax (copy) → syntax (copy)  **FAIL, cycling**
- `Prob093_ece241_2014_q3` (comb): X → copy → copy  **FAIL, cycling**
- `Prob098_circuit7` (seq): X → seq@later → seq@later → seq@later → seq@later → seq@later  **FAIL, budget**
- `Prob100_fsm3comb` (comb): R1 → R1 (copy) → R1 (copy)  **FAIL, cycling**
- `Prob109_fsm1` (seq): R1 → R1 (copy) → R1 (copy)  **FAIL, cycling**
- `Prob112_always_case2` (comb): comb? → copy → copy  **FAIL, cycling**
- `Prob128_fsm_ps2` (seq): seq@later → copy → copy  **FAIL, cycling**
- `Prob142_lemmings2` (seq): R1 → R1 (copy) → R1 (copy)  **FAIL, cycling**
- `Prob156_review2015_fancytimer` (seq): X → X → copy  **FAIL, cycling**

#### `runs/dev/C-s3rewrite-full-1.jsonl`

70 attempts, 66 failed. Every failed attempt has exactly one mode.

| Mode | Failed attempts | of which copied repairs |
|---|---|---|
| wrong output (seq) | 22 | 7 |
| wrong output (comb) | 18 | 3 |
| compile: R1 is not a valid l-value | 9 | 5 |
| compile: R5 is not allowed in a constant expression | 6 | 2 |
| wrong: reset behaviour | 5 | 2 |
| compile: syntax error | 3 | 2 |
| compile: R2 always process does not have any delay | 1 | 0 |
| compile: R3 break statements not supported | 1 | 0 |
| compile: R6 can not select part of scalar | 1 | 0 |

**Inside the 45 "wrong" attempts:**

| Why it is still wrong | Attempts |
|---|---|
| copied | 12 |
| output X or Z | 4 |
| reset behaviour | 2 |
| comb, counterexample | 12 |
| comb, no counterexample | 2 |
| seq, wrong later | 13 |

**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, `R1`... = translator rule, `(copy)` = the same compile error from unchanged code):

- `Prob001_zero` (comb): R2 → PASS  **PASS, passed**
- `Prob012_xnorgate` (comb): PASS  **PASS, passed**
- `Prob021_mux256to1v` (comb): R5 → R5 (copy) → R5 → R5 (copy) → R5 → R5  **FAIL, budget**
- `Prob027_fadd` (comb): PASS  **PASS, passed**
- `Prob035_count1to10` (seq): PASS  **PASS, passed**
- `Prob050_kmap1` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob054_edgedetect` (seq): seq@later → X → seq@later → seq@later → copy  **FAIL, cycling**
- `Prob066_edgecapture` (seq): seq@later → seq@later → copy → copy  **FAIL, cycling**
- `Prob071_always_casez` (comb): R3 → syntax → syntax (copy) → syntax (copy)  **FAIL, cycling**
- `Prob073_dff16e` (seq): R6 → X → copy → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob086_lfsr5` (seq): reset → reset → copy  **FAIL, cycling**
- `Prob092_gatesv100` (comb): X → copy → copy  **FAIL, cycling**
- `Prob093_ece241_2014_q3` (comb): cex → cex → cex → cex → cex → cex  **FAIL, budget**
- `Prob098_circuit7` (seq): X → seq@later → seq@later → seq@later  **FAIL, cycling**
- `Prob100_fsm3comb` (comb): R1 → R1 (copy) → R1 (copy)  **FAIL, cycling**
- `Prob109_fsm1` (seq): R1 → R1 (copy) → R1 (copy)  **FAIL, cycling**
- `Prob112_always_case2` (comb): comb? → comb? → copy  **FAIL, cycling**
- `Prob128_fsm_ps2` (seq): seq@later → copy → copy  **FAIL, cycling**
- `Prob142_lemmings2` (seq): R1 → R1 → R1 (copy)  **FAIL, cycling**
- `Prob156_review2015_fancytimer` (seq): seq@later → copy → copy  **FAIL, cycling**

_Generated by `taxonomy.py` from 2 run file(s). Counts only; no model was used._

## Held-out runs

Every failed attempt in every finished held-out run except A (one attempt, no loop), summed per run type.
The run types have different numbers of reps, so compare shares, not raw counts. The attempt-by-attempt
detail for each run is `runs/taxonomy.md` (`python taxonomy.py --dir runs --out runs/taxonomy.md`).

<!-- gen:taxonomy -->
| Failure mode | R (3 reps) | Q (4 reps) | B (3 reps) | C (3 reps) | D (3 reps) | All | Share |
|---|---|---|---|---|---|---|---|
| compile: is not a valid l-value | 160 | 171 | 107 | 89 | 110 | 637 | 29% |
| wrong output (comb) | 121 | 123 | 96 | 111 | 106 | 557 | 25% |
| wrong output (seq) | 102 | 132 | 68 | 90 | 106 | 498 | 22% |
| compile: has already been declared in this scope | 18 | 37 | 26 | 24 | 33 | 138 | 6% |
| compile: syntax error | 18 | 28 | 21 | 13 | 25 | 105 | 5% |
| compile: other | 19 | 18 | 28 | 21 | 14 | 100 | 5% |
| wrong: reset behaviour | 12 | 17 | 10 | 9 | 16 | 64 | 3% |
| compile: requires an explicit cast | 20 | 7 | 16 | 13 | 6 | 62 | 3% |
| other | 5 | 20 | 12 | 16 | 6 | 59 | 3% |
| **failed attempts** | 475 | 553 | 384 | 386 | 422 | 2220 | |
<!-- /gen:taxonomy -->

**Never solved by any run**, and how their final attempts failed, counted across every run:

<!-- gen:never -->
| Problem | Kind | Final attempt across all runs |
|---|---|---|
| `Prob058_alwaysblock2` | seq | compile error ×19 |
| `Prob063_review2015_shiftcount` | seq | wrong output ×19 |
| `Prob070_ece241_2013_q2` | comb | wrong output ×18, compile error ×1 |
| `Prob080_timer` | seq | wrong output ×19 |
| `Prob121_2014_q3bfsm` | seq | compile error ×16, wrong output ×3 |
| `Prob125_kmap3` | comb | wrong output ×19 |
| `Prob133_2014_q3fsm` | seq | compile error ×18, wrong output ×1 |
| `Prob135_m2014_q6b` | comb | compile error ×12, wrong output ×7 |
| `Prob144_conwaylife` | seq | compile error ×18, wrong output ×1 |
| `Prob145_circuit8` | comb | wrong output ×18, compile error ×1 |
| `Prob149_ece241_2013_q4` | seq | wrong output ×19 |
| `Prob152_lemmings3` | seq | wrong output ×15, compile error ×4 |
<!-- /gen:never -->

**What the held-out failures show**
- **The l-value error dominates.** A signal assigned inside `always` but declared as a wire. Rule R1 names
  the exact fix, and the model often still doesn't apply it: `Prob058_alwaysblock2` failed to compile in
  every run, most often on this error.
- **The never-solved problems are mostly sequential**: FSMs (`2014_q3fsm`, `2014_q3bfsm`, `lemmings3`),
  timers and counters, and Conway's Life. Some fail on Verilog the model can't write (an enum too narrow
  for its states, syntax errors); others compile and compute the wrong thing.

---

`taxonomy.py` and this draft were written with Claude from the committed dev run files. The reading notes in "Read by hand" are filled in by a team member.
