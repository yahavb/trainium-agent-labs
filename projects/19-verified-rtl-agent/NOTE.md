# One-page note: team 19, verified RTL agent

> Every number comes from a committed run file in `runs/`; all held-out runs record code version `0aaab14`.

**What we ran.** An agent loop around Qwen3-8B that writes Verilog for VerilogEval spec-to-RTL
problems. Each attempt is graded by a deterministic checker (Icarus compile, the benchmark's testbench,
and a Yosys equivalence proof for combinational designs). A failure becomes one instruction, and the
model repairs its latest attempt, up to 6 attempts per problem.

**On what.** Qwen3-8B served with vLLM on AWS Trainium2, one chip per seat (seats 90 to 93),
thinking off, temperature 0.6. VerilogEval at commit `c498220`; a fixed split of 20 dev problems for
tuning and 40 held-out problems (21 combinational, 19 sequential) for the results.

**The question.** At an equal budget of 6 attempts, does a translated, located message (run C) solve
more problems than the raw tool output (run B)? Run R, six fresh attempts with no feedback, separates
"feedback helps" from "more attempts help". Run A is the model alone.

**What came out (held-out, solved of 40, each rep).** A, the model alone: <!--g:A-->14, 13, 14<!--/g-->. R, six
same-prompt retries: <!--g:R-->14, 13, 14<!--/g-->. Q, six fresh attempts that rotate the prompt, no feedback:
<!--g:Q-->19, 20, 18, 18<!--/g-->. B, raw-error feedback: <!--g:B-->20, 18, 22<!--/g-->. C, located-fix feedback: <!--g:C-->20, 20, 21<!--/g-->.
D, three strategies then a repair round: <!--g:D-->22, 22, 22<!--/g-->. Per-rep detail, with medians, copies and
escapes, is in `README.md` and `runs/summary.md`.

**Against the analysis plan:**
- **Primary, C vs B: not supported.** No rep shows a significant difference; the discordant counts are
  near-even.
- **Secondary, B and C vs R: supported.** Feedback beats same-prompt retries in every rep.
- **Added control, B and C vs Q: no significant difference in any rep.** Rotating the prompt with no
  feedback gets most of the way; feedback is ahead by about 1–2 problems on average, leaning its way in
  rep 3. So most of the R effect is prompt variety, not the feedback's content.
- **Exploratory, D vs C and D vs Q:** D ties or beats every run in every rep (22, 22, 22) and beats Q in
  every rep (3–0, 3–1, 4–0). No single rep is significant.

<!-- gen:paired -->
| Comparison | Rep 1 | Rep 2 | Rep 3 | Rep 4 |
|---|---|---|---|---|
| **C vs B**: translated vs raw (primary) | 3–3, p = 1.000 | 4–2, p = 0.688 | 3–4, p = 1.000 |  |
| **C vs R**: feedback vs same-prompt retries | 7–1, p = 0.070 | 7–0, p = 0.016 | 8–1, p = 0.039 |  |
| **B vs R**: raw feedback vs same-prompt retries | 7–1, p = 0.070 | 5–0, p = 0.062 | 8–0, p = 0.008 |  |
| **Q vs R**: rotating vs same-prompt retries | 5–0, p = 0.062 | 7–0, p = 0.016 | 4–0, p = 0.125 |  |
| **C vs Q**: feedback vs rotating prompts (added control) | 3–2, p = 1.000 | 3–3, p = 1.000 | 5–2, p = 0.453 |  |
| **B vs Q**: raw feedback vs rotating prompts | 3–2, p = 1.000 | 2–4, p = 0.688 | 5–1, p = 0.219 |  |
| **D vs C**: breadth vs depth | 4–2, p = 0.688 | 3–1, p = 0.625 | 3–2, p = 1.000 |  |
| **D vs Q**: breadth + repair vs rotating prompts | 3–0, p = 0.250 | 3–1, p = 0.625 | 4–0, p = 0.125 |  |

Each cell: problems solved only by the first run – only by the second, then the two-sided exact sign test on those problems. Same 40 problems, same rep number.
<!-- /gen:paired -->

**How many runs, and the spread.** A, B and C: 3 reps; R, D: 3; Q: 4 (fewer if a run in the README table
is missing). Rep-to-rep spread is 0–4 problems per run. The seats are only partly random: every run's first
attempt uses the same prompt and gets 12–14 passes, mostly the same problems. So we report every rep, never a
mean alone, and test problem by problem.

**Simulator vs formal.** <!-- gen:audit -->
Across 19 audited runs, 235 of 245 combinational passes are proven equivalent to the reference by Yosys. The rest: `Prob079_fsm3onehot` in 10 runs, where the only counterexample is a state the spec rules out (non-one-hot in a one-hot FSM), so it is outside the spec rather than a wrong design. 99 sequential passes are checked by simulation only.
<!-- /gen:audit -->

**Analysis plan, fixed at 15:50 EDT, before any rep-2 result was read.**

| Role | Comparison | When it was fixed |
|---|---|---|
| Primary | C vs B: translated vs raw feedback | DESIGN 1.0.0, before any held-out run |
| Secondary | C vs R and B vs R: feedback vs retries without feedback | D-008, before any held-out run |
| Added after viewing rep 1 (disclosed) | C vs Q and B vs Q. Q is 6 fresh attempts with no feedback that rotate prompts S1, S2, S4 (`controls/prompt_rotation.py`) | 15:50. R-1's retries returned byte-identical code 63% of the time, so R alone cannot tell "information" from "any new prompt" |
| Exploratory | D vs C (breadth vs depth), and a rep 3 of B and C with seats swapped | 15:50 |

- **Test:** for each rep, an exact two-sided sign test on the problems where exactly one of the two runs passed. Every rep is reported, p-values are not pooled, and problems solved in both reps are counted separately.
- **Calibration:** every combinational pass is re-checked by Yosys. Escapes are reported. A mismatch that occurs only on inputs the spec excludes (such as a non-one-hot state in a one-hot FSM) is reported as that, not as a failure.

**Disclosures.**
- Rep-1 results of A, R, B and C were viewed at 15:22 and 15:38, before rep 2 finished. No code changed afterwards, and every held-out run records code version `0aaab14`.
- An earlier set of held-out runs at code `0d4fe8d` was stopped at 14:50 when the team moved to `0aaab14`. Those runs were never read and are not reported.
- Dev check C-4 (seat 90) was stopped at 14:50 so that seat 90 could run D.
- `runs/dev` was removed from the repo at 15:00 and remains in history at `b73760c`.

**What we learned building it (dev, context, not results).**
- The original repair prompt ("keep everything else identical") made 71% of repairs return the broken
  code unchanged, rescuing 0 of 20 problems (`runs/dev/C-1.jsonl`). A rewrite request plus a fresh
  restart after a copy cut copies to 35% and rescued 2 (`runs/dev/C-2.jsonl`).
- Each translator rule came from a dev failure and names one fix. Two of our own messages were wrong
  and were found by reading transcripts: R1 told internal signals to become outputs, and the reset
  message said "high" for active-low resets.
- On the organisers' NKI kernel ladder, no feedback setting solved a level (0 of 36 level-runs). A
  better message helps when the model knows the language and made a mistake, not when it lacks the
  knowledge.
