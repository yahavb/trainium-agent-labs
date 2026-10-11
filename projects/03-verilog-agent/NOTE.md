# AI-Powered Verilog Generation, Verification, and Optimization: one-page note

**What it is.** Describe a circuit (the AI writes it) or submit your own Verilog. The design is checked by an automated
testbench; failures go through a feedback engine that tells the AI what to fix; passing designs go to a gate-count
optimizer that only keeps a smaller circuit if it still behaves identically. Entry point: `python vagent.py`.

**On what.** Qwen3-8B served by vLLM-Neuron 0.24 on one AWS Trainium chip (NYU × Annapurna Labs seat pod, TP=2),
thinking mode off, ~14 tokens/s per request. Simulation and checks run on the pod's CPU: Icarus Verilog 12, Yosys, Python.

## The loop

1. **Generate / submit** → one JSON hand-off: `{prompt, verilog, testbench?}` (a user testbench is used as-is).
2. **Verify.** The AI writes the expected behaviour as a *Python golden model* (3 independent models, 2 must agree);
   Python bakes every expected value into a self-contained testbench. It is accepted only if it compiles, fails an
   empty design, and drives every input bit to 0 and 1. Exhaustive up to 12 input bits, else 4000 random vectors;
   clocked designs get 300 cycles with resets.
3. **Feedback.** On failure, `for_feedback.json` (inputs, expected, got) → the feedback engine classifies the error
   (syntax / functional / testbench / timeout), points at source lines, and writes suggestions the AI reads before fixing.
   A judge step stops the loop from "fixing" a correct design when the golden model itself looks wrong.
4. **Optimize.** The verified design is `vprev`; the optimizer proposes `voptimized`. Kept only if Yosys counts
   fewer gates **and** its outputs match `vprev`; otherwise `vprev` is returned.

## What we ran and what came out

**Testbench exam** (`make_tests.py`): 9 circuits, each with a correct and a buggy version (multiplier, barrel shifter,
16-bit adder, priority encoder, popcount, BCD-to-7-segment, signed comparator, shift register, 101-detector FSM).
"Perfect" = testbench built, correct version passes, buggy version fails.

| Version | Change | Perfect |
|---|---|---|
| v10 | AI writes a Verilog reference model, 2-of-3 voting | 3/9 |
| v11 | + clocked template, signed ports | 6/9 |
| v12 | + golden model in **Python** | 7/9 |
| v14 | + judge (design vs golden model) | **7/9, 7/9** (2 runs) |

- **Spread:** the two v14 runs are identical, down to the failing inputs, so the 7 passes are stable and the 2
  failures are systematic, not luck.
- **Bugs caught:** 9/9 planted bugs in each v14 run.
- **Testbench strength:** 45/45 mutants (planted single bugs) killed on the 7 passing designs.
- **Feedback engine:** 263/263 unit tests pass. In the loop it located a missing `;` by line and flagged `|` used where `^` was needed.
- **End-to-end** (menu, real pod runs): buggy popcount → score 0.5 → feedback → AI fix → PASS 1.0 (14/14 mutants, 144 s);
  gt8 comparator → PASS first try (13/14 mutants, 35 s).
- **Optimizer:** gt8 (redundant 8-bit comparator) **61 → 38 Yosys cells (38% smaller)**, smaller version passed the same
  testbench (comparator.py). Best possible is 33 (`a > b`). 1 run so far; more runs needed for a spread.

## What we learned

1. **The checker's correctness is the bottleneck.** Every false failure came from a wrong golden model, never from the
   simulator. A wrong checker sends correct designs to be "fixed".
2. **Ask the small model for less, in a language it is good at.** Full AI testbenches mostly failed to compile;
   Python writes the testbench, the AI writes only the expected-value function. 3/9 → 7/9.
3. **Voting fixes random mistakes, not shared ones.** On the priority encoder all 3 models and the judge agreed on the
   same wrong reading of "highest set bit". An independent, larger judge model is the next step.
4. **Optimization must be gated by verification.** A smaller circuit only counts if it still matches `vprev`.

**Next.** A larger, independent judge model (shared gpt-oss-20b); formal equivalence (Yosys) to prove `voptimized ≡ vprev`
for all inputs instead of tested vectors; more exam runs.

*Artifacts:* `CHECKER.md` (verification rules + reasons) · `ATTEMPT_LOG.md` (every attempt with its score) ·
`results/scorecard_*.log` (exam runs).
