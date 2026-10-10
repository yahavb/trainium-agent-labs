# Heat-Rod PDE Agent — One-Page Note

**Scope and system.** We evaluated a real Qwen3-8B agent that proposes heat-equation
solutions, requests integrals from a calculator, receives checker feedback, and revises
answers within a fixed budget. Recorded hardware: seat-85, one allocated Trainium2 chip
on a trn2.48xlarge host, tensor parallelism 2, context 8192 (historical RESULTS.md).
No new inference was run for the latest runtime fixes.

**Experiment.** Reference c3c16f6, batch 20261010T170025Z: Level 0.1–0.3 and
Level 1.1–1.3 at problem seed 0, plus Level 1.3 seeds 1 and 2. Each variant used
two samples/round, at most three rounds, 512 output tokens/request, one calculator
exchange/attempt and a 180-second case deadline; concise mode was off.
The historical command can be reconstructed as python benchmark.py --timeout 180
at that revision; a captured invocation file is not present in this batch.

**Run count and variability.** There were eight paired configurations, one execution
per variant/configuration: 16 executions, including one timeout. There were no repeated
trials of an identical variant/configuration in this reference batch. Independence of
stochastic model draws is not established; candidates within rounds are not independent
replicates. Repeat-to-repeat variability therefore cannot be estimated.

| Historical metric | Original | Previous improved |
|---|---:|---:|
| Validated solved | 3/8 | 3/8 |
| Level 0 / Level 1 | 2/3; 1/5 | 3/3; 0/5 |
| Total elapsed seconds | 813.5 | 687.2 |
| Timeouts / service failures | 0 / 0 | 1 / 0 |

Cross-case elapsed ranges were 39.6–128.2 s and 36.6–180.0 s, respectively;
these are differences across problems, not repeated-trial variability.

**Findings.** Total success did not improve. Level 0 improved but Level 1 regressed.
The 15.5% reduction in total time is not evidence of better mathematical reasoning.
All cases, including the timeout, remain in the denominator.

**Latest engineering changes.** Runtime revision 2d1a68a isolates checker/calculator
exceptions, preserves partial logs, stops failed batches and corrects request accounting.
Its 35 core/regression tests, 11 workflow tests and both official selftests passed.
Its real-model solve rate is unmeasured; no speed or accuracy gain is claimed.

**Limitations and status.** Historical validated scores required an extra numerical
validator; current code uses the unchanged original checker only. These policies must
not be pooled. Numerical checking is not a universal proof. The 80 preserved candidate
records include 42 original-agent records without contemporaneous checker feedback;
the timeout trace is partial. Missing historical data is not reconstructed as original
evidence. Final status: NOT READY, with engineering review also required.

**Artifacts.** Checker: ../pdecheck.py; rationale: CHECKER.md. Reference logs and
case statuses: ../benchmark-results/20261010T170025Z/. Audit: reference-log-audit.json.
Tests: ../evidence/runtime-resilience/. Full gate and blockers: READINESS.md.
