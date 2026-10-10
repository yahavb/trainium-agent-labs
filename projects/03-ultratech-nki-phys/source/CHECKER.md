# Checker Acceptance and Rejection

Current scope: normalized frictionless contact snapshots, one world per check.
The checker is `contact.diagnostics`; fixtures and the FP64 reference are trusted
inputs owned by the evaluator. Candidate code must not receive oracle outputs.

The binary score is 1 only if every implemented gate passes, otherwise 0. The log
also keeps individual errors so failed scores remain useful for diagnosis.

| Gate | Reason |
| --- | --- |
| Exact impulse-vector shape and finite values | Malformed output and NaN/Inf are invalid results, regardless of apparent speed |
| Impulses >= -1e-6 | Frictionless normal impulses must not pull bodies into contact; tiny normalized rounding error is allowed |
| Projected-gradient residual <= 1e-4 | Tests constrained optimality, rather than whether a solver reports convergence |
| Normalized objective discrepancy <= 1e-5 | Independently checks proximity to the reference optimization result |
| Normalized velocity discrepancy <= 1e-4 | Tests the physical response, which may still be wrong despite an acceptable stopping residual |

The independent FP64 oracle solves the QP through nonnegative least squares after
Cholesky transformation. Every reference must have projected residual <= 1e-9.
Analytical one-contact solutions and two-body inelastic momentum provide checks
separate from the iterative candidate. These validate the implemented restricted
model, not equivalence to MuJoCo or real-world contact behavior.

All norms, normalizations and formulas are in `contact.py` and the project contract.
Tolerances are normalized engineering thresholds for FP32 iterative solutions;
they must be frozen before final evaluation and must not be relaxed to pass a
candidate. Residual, objective and velocity gates are conjunctive: good residual
alone is insufficient. No reward is given merely for compilation or low runtime.

`test_contact.py` checks analytic answers and rejects planted wrong signs, zero
responses, perturbed impulses, nonfinite values and incorrect shapes. The
development manifest records every failed candidate rather than discarding it.

Pending before final submission: actual padded-contact invariance and input
mutation tests, generated-code isolation and rule enforcement, held-out fixtures,
device output validation, and trajectory tests. The current checker has not
implemented these broader contract requirements. It must not certify those claims.

The CPU candidate now supports a reference-free velocity-bound stopping rule.
For positive-definite A with smallest eigenvalue mu and largest L, the projected
gradient map with step 1/L contracts by 1-mu/L. Consequently the impulse error
is bounded by ||G||2/mu. Multiplying by the largest row 2-norm of M-inverse J.T
bounds absolute velocity error. Requiring that bound <= 1e-4 is sufficient for
the normalized velocity gate because its denominator is at least one. This bound
uses FP64 host checks and spectral preparation; its cost is included in CPU solve
time. It is conservative and does not guarantee termination within 4096 updates.
The independent objective gate still applies; the bound is not an oracle answer.
The NKI implementation and its device-check precision remain to be established.

Repeated CPU attempts currently test a deterministic baseline. They are recorded
as such, not presented as an agent repair loop. Future model attempts must also
record prompt, response, model/decoding settings, candidate hash, parent attempt,
feedback, tokens, compile status, all correctness gates and hardware measurements.
The parent process is the only log writer, avoiding interleaved parallel records.
