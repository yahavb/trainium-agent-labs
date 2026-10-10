# Adaptive Restart Experiment and Fallback

The six-attempt fixed-momentum loop at 1024 updates scored 10, 10, 10, 6, 6,
and 4 out of 16 for beta 0.9, 0.95, 0.98, 0.99, 0.995 and 0.999 respectively.
Every attempt remains in data/grip-loop-6d62f48aef734d9bbdf4824307e2ec71.

## Implemented Mathematics

For each world, initialize x=y=initial and t=1. At each iteration:

1. x_next = max(0, y - alpha * (A*y + b)).
2. r = dot(y - x_next, x_next - x).
3. t_next = (1 + sqrt(1 + 4*t*t))/2; beta = (t - 1)/t_next.
4. If r > 0, set beta=0 and t_next=1.
5. y_next = x_next + beta*(x_next - x); then update x, y and t.

Each world has independent state, schedule and restart decisions. Returned
forces are x, not extrapolated y. The objective and checker remain unchanged.
Restart is a heuristic; no universal speedup, monotonicity or finite-budget
accuracy guarantee is claimed. This is human-written algorithm/code, not Qwen
discovery. Public case IDs and oracle solutions do not enter the solver.

Sources: [adaptive restart research](https://arxiv.org/abs/1204.3982),
[FISTA](https://epubs.siam.org/doi/abs/10.1137/080716542), and
[NKI ISA reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_modules/nki/isa.html).

## Observed Results

All 16 public gripping fixtures, one CPU execution per case/method/budget:

| Working Arithmetic | Method | 1024 Updates | 4096 Updates |
|---|---|---:|---:|
| FP32 | Fixed beta 0.9 | 10/16 | 10/16 |
| FP32 | FISTA + restart | 10/16 | 10/16 |
| FP64 force/gradient work | FISTA + restart | 16/16 | 16/16 |

The FP64 diagnostic uses the same exported FP32 inputs and FP32 schedule
scalars, but higher precision for vector/matrix computations. It is not an
eligible FP32 kernel or a device result. These results implicate iterative
arithmetic precision in the remaining failures; they do not establish the exact
source of rounding error or guarantee that a different FP32 algorithm cannot pass.

Artifacts: data/adaptive-reference-f815f6efbd704d3ba094bba230859290 (1024),
data/adaptive-reference-4413d3109b7c4646a5e0a13a1c8f35c1 (4096).
Both contain every score, saved output, failure metric and restart history.

adaptive_reference.py is tested locally. nki_contact_adaptive.py implements the
same equations, but cannot yet be called SDK-validated: installed NKI simulation
and device compilation have not run. Its extra dot-product, square-root,
reciprocal and broadcasting operations add cost. Simulator/CPU results are not
Trainium latency or throughput measurements. The old beta-only automatic loop
intentionally rejects this broader algorithm change; do not bypass that gate.

## Hackathon Time-Limit Decision

On October 10, 2026, we decided to shift the active hackathon deliverable from
frictional gripping to simpler frictionless plane/pair contact problems because
of the hackathon's time limit. The remaining time is being prioritized toward
a reproducible, correctness-gated Trainium throughput experiment rather than
further investigation of gripping's unresolved FP32 numerical limitations.

This is a scope reduction, not a successful completion of the gripping task.
The fixed-momentum search did not exceed 10/16, and adaptive restart remained
at 10/16 in the tested FP32 CPU reference at both iteration budgets. Higher-
precision CPU arithmetic passed 16/16, but that diagnostic is not a working
FP32 Trainium solution. It does not prove that gripping is impossible in FP32.
All gripping attempts, failures, deviations and diagnostic results are retained.
Gripping remains a future extension; its accuracy gates are not relaxed.

Return to frictionless synthetic plane/pair contact snapshots, starting with
33 contacts, batch 8 and 32 projected-gradient updates. Local CPU rechecks passed
8/8 for each scene: data/batch-cpu-478b0bc24de54d58ba4b47202c01a163 and
data/batch-cpu-d6930a022b1f44d6b5f8d7e15a32663e.

These are synthetic contact impulse QPs, not gripping/MuJoCo force fixtures or
validated rollouts. Keep that distinction explicit. The next optimization target
is measured device throughput with all physics gates passing. Previous passing
results establish a baseline, not a new Qwen improvement. Broader NKI proposals
still need manual review or isolation before execution.
