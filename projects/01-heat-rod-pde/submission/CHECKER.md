# Checker implementation, acceptance and limitations

Implementation: [pdecheck.py](../pdecheck.py). Problem definitions:
[level0_heatrod.py](../level0_heatrod.py) and [level1_heatrod.py](../level1_heatrod.py).
The runtime exception wrapper is [checker_runtime.py](../checker_runtime.py).
These explanations describe the unchanged original checker, not a new grading rule.

## Accepted solution

The candidate is parsed as an expression in x and t. The checker accepts its supported
Python/SymPy syntax, including its existing implicit multiplication behavior. Unknown free
symbols or unparseable expressions receive the original checker's format feedback.
No new AST restriction is imposed on grading.

A score of 1.0 requires all four original checks to pass:

| Constraint | Weight | Evaluation |
|---|---:|---|
| Heat equation | 0.4 | Residual u_t - k*u_xx |
| Left boundary | 0.2 | u=0 for Dirichlet, or u_x=0 for an insulated boundary |
| Right boundary | 0.2 | Same boundary-type-specific rule |
| Initial condition | 0.2 | Relative L2 difference from the specified initial shape |

The PDE/boundary checks use 24 problem-seeded numerical sample points/times. Times are
drawn from [0, 0.05 L²). Residuals must be finite and their maximum absolute magnitude,
normalized by the initial-shape scale, must be strictly below 1e-6. Neumann boundary
values are multiplied by L as in the original implementation.

The initial condition is evaluated on an 801-point grid including both ends. The
relative L2 error uses trapezoidal integration and must be strictly below the problem's
tol. For Level 1.3 that is 0.005 (0.5%); it must not be increased. The published example
rejects one/two Fourier terms and accepts three/five when their coefficients and decay
rates are correct. The actual requirement is the error threshold, not a hardcoded term count.

## Rejection and feedback

Wrong decay can fail the equation while preserving boundaries and initial shape. Wrong
wave frequencies can violate the insulated-end condition. Wrong coefficients or omitted
modes can fail the initial-shape threshold despite satisfying the PDE. Feedback identifies
failed constraints, residual examples and qualitative coefficient/decay errors.

REVEAL_COEFFICIENTS remains false. Runtime solving must not consume exact solutions or
series_answer. The checker tests the submitted expression against the stated problem.

A normal checker rejection retains its numerical partial score. If the checker itself
raises, checker_runtime records evaluation_error and null reward/original_reward; this is
not a new zero score. The agent may correct its answer within the remaining budget.

## Known limitations

Finite samples and grids are not a proof over all space and time. High-frequency aliasing
and rapidly decaying errors can escape the sampled points; existing parity tests retain
that original behavior. Numerical tolerances and floating-point evaluation also limit
precision. A finite Fourier approximation can meet the stated tolerance without being
an exact infinite-series solution.

A score of 1.0 means original-checker acceptance only. The historical extra numerical
validator is not active and must not be reported as passing. Symbolic parsing/evaluation
can fail or run slowly; exception isolation does not preempt a hung calculation.

Official selftests passed on runtime commit 2d1a68a; see
[checker-tests.log](../evidence/runtime-resilience/checker-tests.log).
