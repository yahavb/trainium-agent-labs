# The checker: what it accepts, what it rejects, and why

The checker decides whether a Verilog design does what its spec says. It has two jobs, and each has its own rules:
**build a testbench it can trust**, then **judge the design with it**. Code: `vagent.py` (testbench engine),
`verify.py` (runs it, writes the hand-offs), `mutate.py` (measures testbench strength).

The guiding rule: *a wrong checker is worse than no checker.* If the checker is wrong, the loop "fixes" correct
designs into broken ones. So every rule below exists to stop the checker from lying in one direction or the other.

## 1. Where the expected answers come from

| Rule | Why |
|---|---|
| The AI writes a **golden model in Python** (`def model(...)` / `def step(state, ...)`), not in Verilog | Verilog reference models failed on width truncation (`(a+b)>>16` is 0), signedness, and illegal part-selects. Python has none of these traps. Moving to Python took the exam from 6/9 to 7/9. |
| **3 independent models; 2 must agree on every test vector** | A random misreading of the spec rarely happens twice the same way. Disagreement is sent back with the exact input where they differ. |
| The model sees **only the spec and the port list, never the design** | Otherwise the reference copies the design's bug. |
| Python runs in a **sandbox**: separate process, no imports, restricted builtins, 10 s timeout | It is AI-written code. Crashes, infinite loops and `import os` are rejected, not executed. |
| Expected values are **baked into the testbench** | The testbench is a fixed, self-contained file. It is reused unchanged across fix rounds and after the optimize step, so the target never moves. |

Fallbacks, in order: a Verilog reference template (AI writes only `exp_* = ...` statements), then an AI-written whole testbench.

## 2. Which inputs are tested

| Design | Stimulus | Complete? |
|---|---|---|
| Combinational, ≤ 12 input bits | **every** input combination | yes |
| Combinational, > 12 input bits | 4000 random vectors, fixed seed | strong evidence, not proof |
| Clocked (has a `clk` input) | 300 cycles of random inputs; reset held for cycles 0–1, again mid-run, and at random (~1 in 29) | strong evidence, not proof |

## 3. A testbench is accepted only if it passes all of these

| Check | Rejects |
|---|---|
| Compiles against an empty copy of the design's ports | wrong port names/widths, syntax errors (the error and the broken line go back to the AI) |
| Finishes within 10 s | infinite loops, a missing `$finish` |
| Prints `ERRORS=<n> TOTAL=<m>` with `TOTAL > 0` | testbenches that test nothing |
| **Fails an empty design** (outputs never driven) | testbenches that never really compare outputs (e.g. `!=` instead of `!==`) |
| **Input toggle coverage:** every input bit is driven to both 0 and 1 | e.g. an adder testbench that never sets `cin = 1`, which would let a carry-in bug pass |

## 4. Verdict on the design

| Outcome | When | Hand-off |
|---|---|---|
| **PASS** | 0 mismatches on every vector | `for_optimize.json`, listed in `designs/PASSED.txt` |
| **FAIL** | any mismatch: each one reports inputs, expected, got | `for_feedback.json` → fix step |
| **FAIL (compile)** | the design does not compile: its own error + the line on/just before it | `for_feedback.json` |
| **REVIEW** | the judge (below) sides with the design twice | `for_review.json` → a human; **never** sent to be "fixed" |
| **TESTBENCH ERROR** | a user-supplied testbench does not compile | nothing; the design is not blamed |
| **NOT TESTED** | no valid testbench could be built | stated plainly; never reported as a pass |

**The judge.** When a design fails a testbench that was *just* built, the golden model might be the one that is wrong.
For up to 3 mismatches the AI is asked a concrete question ("spec says X; input req=4; A) idx=2, B) idx=1 — which?"),
3 votes each, with the last few clock cycles shown for clocked designs. If the design wins, the testbench is rebuilt
once with a warning; if it still wins, the case goes to human review. The judge never runs on a testbench that was
already trusted (a user's, or one reused across fix rounds), so a real bug cannot argue its way out.

**User testbenches** are used as-is, but only if they instantiate the design under test (a pasted design is rejected).

## 5. How strong is the testbench? (mutation testing)

`mutate.py` plants one bug at a time in a design that passes (`+`→`-`, `&`→`|`, `*`→`+`, constant +1, negated `if`,
dropped term, …) and counts how many the testbench catches. Mutants that do not compile are excluded.
v14 exam: **45/45 mutants killed** across the 7 designs that passed.

## 6. Known limits (measured, not guessed)

- **Same-model blind spots.** If Qwen3-8B misreads a spec, all 3 golden models and the judge can share the mistake.
  Seen on `prienc8` ("highest set bit": models said idx=1 for req=4) and `det101` (models kept history across reset).
  Fix: a different, larger model as the judge.
- Random stimulus is evidence, not proof, for wide or clocked designs. Formal equivalence (Yosys) would close this.
- Small designs give few mutants (e.g. 1/1), so their mutation score says little.
- Clocked designs assume registered outputs.
- A vague spec produces a vague golden model. Name the ports, widths, reset and edge cases.
