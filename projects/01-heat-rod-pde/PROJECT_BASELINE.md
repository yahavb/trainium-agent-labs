# Project 1 — Fixed Baseline and Submission Requirements

Established by the project owner on 2026-10-10. This is the standing policy for future
Project 1 changes, experiments and submissions. “Official deliverables” below follows
the owner's supplied requirements; the additional engineering gate is labeled separately.

## Objective and required functionality

Improve real-model PDE solving, especially Level 1 validated solve rate.
Preserve the full loop: model candidate → independent mathematical/physical checker →
score and actionable feedback → model correction → repeat until acceptance or budget
exhaustion. Preserve real inference, problem generation/loading, model-directed calculator
calls, candidate evaluation, attempt logs, reliable termination and failure reporting.

Do not claim improved performance merely because incorrect answers are faster.

## Official deliverables

A. **Checker**: deliver the actual implementation, explain PDE, boundary and initial
condition acceptance/rejection criteria, and document known limitations. Preserve the
original rules, weights and tolerances; never relax them to raise solve rate.

B. **Attempt Log**: preserve every real attempt, including problem identifier, problem
seed, round/attempt number, generated answer, checker score and feedback, and final case
status. A linked case summary may supply final status if the relationship is explicit.
Also preserve prompt, model/configuration, calculator trace, latency, errors and truncation
where available. Unknown scores are null with a reason. Do not fabricate missing feedback
or relabel regenerated feedback as historically recorded.

C. **One-Page Note**: explain what ran, model/hardware, configuration, independent run
count, results and observed variability, findings and limitations. Include unsuccessful
and interrupted outcomes. Do not select only the best run.

## Additional engineering criteria — not official requirements

1. Official selftests pass.
2. Existing regression tests pass; new behavior has appropriate regression coverage.
3. Compare every proposed optimization against a fixed baseline with matched problems,
   seeds, samples, rounds, token/tool budgets, verification, and recorded resource differences.
4. No scoring-rule changes to manufacture gains.
5. No hidden exact answer, series_answer, hardcoded benchmark answer or other answer leak
   as a runtime solving shortcut. Known answers may be used by the official selftests.
6. Offline tests, controlled responses and recorded-candidate replay are not live-model evidence.
7. Report timeout, service failure and mathematical unsolved separately; retain partial data.
8. Preserve logs, actual commands, code revision, source snapshot and reproducible configuration.
9. Explain each change's benefits, costs, limitations and possible regressions.
10. Never merge main/master without the owner's confirmation.

Before an optimization, check compliance, fix the comparison protocol and state the expected
benefit and possible regression. Afterward, run required tests and matched real-model evaluation
when authorized and available. Record unmeasured effects as unmeasured. Stop real experiments
on service unavailability, permission failure or budget exhaustion; never substitute fake data.

## Fixed performance reference

Reference commit: c3c16f6.
Reference batch: benchmark-results/20261010T170025Z/RETEST.md.

| Metric | Original Agent | Previous Improved Agent |
|---|---:|---:|
| Historical validated solved | 3/8 | 3/8 |
| Level 0 | 2/3 | 3/3 |
| Level 1 | 1/5 | 0/5 |
| Total seconds | 813.5 | 687.2 |
| Timeouts | 0 | 1 |

This is the specific 8-case team experiment, not the README's historical 6/6 example.
It used the historical additional validation policy. Preserve these reported values and
their provenance; do not silently relabel them as scores from a different acceptance policy.

The current runtime uses the original checker only. Future comparisons must label the
original checker result and any separate additional validation independently, apply the
same policy to both variants, and record skipped/unknown validation explicitly.
Additional analysis must not alter the original checker score, problem requirements or
stopping rule. Do not pool runs with different budgets or acceptance policies.

## Submission gate

Before declaring submission ready, check checker code and rationale, all attempt logs,
the one-page note, official selftests, regression tests, accurate real-model results,
run count and spread, limitations, and committed/backed-up code and deliverables.

- Missing official deliverable or required official content: **NOT READY**.
- Official deliverables complete but engineering checks unresolved: **ENGINEERING REVIEW REQUIRED**.
- Both sets satisfied with evidence: **READY**.
- When both have gaps, show NOT READY as the overall status and list engineering issues separately.

Every final readiness report must contain official completeness, engineering checklist,
benchmark comparison, outstanding blockers, commit/artifact locations and final status.
If an official requirement conflicts with an additional engineering goal, prioritize the
official requirement and document the conflict explicitly.

Current evidence-based assessment: [submission/READINESS.md](submission/READINESS.md).
