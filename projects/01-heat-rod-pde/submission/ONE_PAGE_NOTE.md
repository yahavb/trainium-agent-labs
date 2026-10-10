# Team 18 — Project 1: Heat-Rod Agent

**10 October 2026**

**Task and approach.** We used Qwen3-8B on AWS Trainium to solve three Level 1
heat-equation problems through a model → checker → feedback → revision loop.
Level 0 was warm-up. We report one baseline and three subsequent suite runs;
preliminary and interrupted runs remain in the detailed logs.

**Acceptance and changes.** The original checker awards 0.4 for the PDE, 0.2 for
each boundary and 0.2 for the initial condition; only 1.0 counts as solved. The
parabolic initial shape permits less than 0.5% relative error. Attempt 1 used
symbolic decay repair. Attempt 2 (`cd16f7b`) used calculator caching, model-driven
repair, targeted feedback and adaptive budgets. The latest run (`31a150d`) kept
that agent but enabled optional enhanced verification: precision audits, exact
boundary evaluation, initial-condition uncertainty estimates and richer diagnostics.
Weights and tolerances stayed fixed; acceptance and feedback paths changed.

| Level 1 suite | Solved | Best scores: 1.1 / 1.2 / 1.3 | Time | Full-score candidates |
|---|---:|---|---:|---:|
| Initial baseline | 2/3 | 1.0 / 0.8 / 1.0 | 1,408.0 s* | Incomplete record |
| Attempt 1 | 3/3 | 1.0 / 1.0 / 1.0 | 410.65 s† | 6/12 |
| Attempt 2 | 3/3 | 1.0 / 1.0 / 1.0 | 228.24 s† | 9/16 |
| Latest: enhanced checker | 3/3 | 1.0 / 1.0 / 1.0 | 187.00 s† | 7/16 |

*Sum of round durations. †Sum of per-problem elapsed times, including failures.
Attempts 2/latest used seed 0, four candidates/workers, at most four rounds,
1200 tokens/request, one calculator exchange and 900-second deadlines. Attempt 1's
full configuration is unverified. The latest run used 1/1/2 rounds, 29 requests,
15.19/59.00/112.82 seconds and had no candidate request/evaluation errors.

**What the loop corrected.** In Attempt 2's third problem, coefficients and waves
were already correct. For `A*exp(-lambda*t)*sin(mu*x)`, the PDE requires
`lambda=k*mu**2`. With `k=2` and `mu=n*pi/2`, rates `2,18,50` times `pi**2`
became `0.5,4.5,12.5` times `pi**2`: the original answer omitted division by the
rod length squared. All four scores rose 0.6→1.0; initial error stayed 0.3364%.
Targeted feedback may preserve correct structure while localizing repair.

**Limits and next steps.** Latest candidate scores agreed with the independently
recorded original checker, but changed feedback, sampling and service variation
prevent causal speed claims. Unchanged agent code needed one versus four rounds
for problem 1.2 across complete/interrupted runs. Decay repair cannot fix wrong
amplitudes or missing modes; recompute frequencies rather than memorize “divide
by four.” A pre-core checker-timeout defect was reproduced and fixed; 70 regression tests
and both official selftests passed. The 187-second run predates this fix. Preserve
all traces and repeat matched runs with individual features isolated before
claiming stable improvement.
