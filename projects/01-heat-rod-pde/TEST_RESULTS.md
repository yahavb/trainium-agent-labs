# Verification record

Executed in `seat-85` with the lab's Python/SymPy/NumPy environment on October 10, 2026.

- Original `level0_heatrod.py --selftest`: PASS, including exact solutions for 15 generated problems and deliberately wrong decay/boundary/format cases.
- Original `level1_heatrod.py --selftest`: PASS, including insulated boundaries and the 0.5% initial-shape tolerance (one/two Fourier terms rejected; three/five accepted).
- `python -m unittest test_improvements -v`: 9 tests passed, including the regression that omits the Neuron-incompatible per-request sampling seed.

New test cases:

1. Best candidate survives a later worse round.
2. HTTP 500 retries succeed without sending a per-request seed.
3. Unreachable model is labeled infrastructure failure, without an invented zero math reward.
4. Calculator trace retains the model's chosen integral and its result.
5. Exact solutions pass held-out validation over multiple seeds after removing the known-answer field.
6. A high-frequency grid-aliasing wrong answer passes the original checker but fails the added validation.
7. Fourier truncation thresholds remain correct.
8. Attribute access, comprehensions, unknown function calls and non-arithmetic expressions are rejected.
9. Wrong decay and insulated-boundary errors are rejected.

These are code/physics checks, not measured model-solving success rates. Real-model results are reported separately.
