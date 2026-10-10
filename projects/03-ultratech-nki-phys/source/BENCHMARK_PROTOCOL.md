# Physics Benchmark and Model Evaluation Protocol

## Status

No Qwen training, fine-tuning or model-generated solver attempts have run yet.
The existing synthetic and engine-backed cases are development data. None is an
private evaluation has occurred. A separate frozen split is now generated in
`data/engine-sets-v1`; its private cases have only undergone reference certification,
not candidate evaluation. Subsequent prompt/feedback iterations are agent
development, not weight training. Keep these distinctions in the submission.

## Two Different Mathematical Tasks

The original synthetic task is a regularized, instantaneous contact-impulse QP:
`A = J M^-1 J.T + 0.001 I`, `b = J v_free`, `impulse >= 0`.
It uses normalized units with no restitution or position stabilization.
The candidate produces impulses; the checker examines the resulting velocities.

The engine-backed task is MuJoCo's frictionless soft-contact force QP:
`A = J M^-1 J.T + diag(efc_R)`,
`b = J qacc_smooth - efc_aref`, `force >= 0`.
The reference acceleration is `qacc_smooth + M^-1 J.T force`.
Use the engine's regularizer and reference acceleration, not the synthetic ones.
Outputs are forces, not impulses. Do not reuse the velocity checker or compare
these forces directly against outputs of the old impulse fixtures.

Both are convex nonnegative QPs, so the same numerical solver family can apply;
their physical inputs, output interpretation and acceptance checks differ.

## Current Engine Development Cases

| Family | Configuration | What It Checks |
| --- | --- | --- |
| Plane | 1 or 8 independent free spheres touching a plane | Uncoupled normal contacts; one-contact closed-form check |
| Pairs | 1 or 8 independent sphere pairs, away from floor | Pair impulses are replaced here by forces; equal/opposite internal momentum rates |
| Stack | 1 or 8 vertically arranged free spheres | Coupled contact equations and force propagation |

Each configuration uses seeds 0 and 1: 12 snapshots, not 12 trajectories.
Masses are seeded in [0.5, 2]; radius is 0.2. Small initial penetration creates
contacts. Initial translational velocities are recorded; angular velocities are
zero. Pair scenes have zero gravity; floor/stack scenes use -9.81 along z.
Full mass matrices include rotational degrees of freedom.
The scene XML records every engine setting, including normal-only `condim=1`,
zero friction, soft-contact parameters, dense Jacobian, Newton solver, iteration
limit 200, solver tolerance 1e-12 and disabled warmstarts. MuJoCo is pinned to
3.15.0. Extracted contacts, not requested counts, define the actual QP size.

A one-sphere stack is physically the same as a one-sphere plane case with the
same seed. Problem fingerprints identify these duplicates. Do not count them
as independent evidence or put equivalent snapshots in different splits.

## Independent Validation

The engine supplies geometry, contacts, Jacobians, mass matrix, regularization,
reference acceleration, contact forces and generalized accelerations.
SciPy NNLS independently solves the extracted QP in FP64 after a Cholesky
transformation. Certify the reference residual <=1e-9. Compare engine and NNLS
forces and generalized accelerations with normalized max-error <=1e-7.
The existing feasibility, objective and projected-optimality gates also apply.
These tight extraction/reference tolerances are not a new FP32 candidate gate.
FP32 force and acceleration acceptance is separately frozen in `force_checker.py`
and each suite's `public/contract.json`: normalized force error <=1e-4,
normalized generalized-acceleration error <=1e-4, feasibility >=-1e-6,
projected residual <=1e-4, objective discrepancy <=1e-5. Shape and finite checks
are mandatory. The old impulse velocity threshold does not substitute.

Tests independently check the one-contact closed-form answer and conservation
of total linear momentum rate for a pair with no external forces. Invalid
frictional constraints and deliberately corrupted engine outputs are rejected.
This validates a restricted engine model, not agreement with physical experiments,
frictional manipulation, articulated robots or long-horizon stability.

## Artifact Verification

For each case preserve scene XML, qpos/qvel, M/J/R/A/b, contact geometry,
engine force/acceleration and independent reference force. The manifest records
engine/runtime versions, source hashes, input/output file hashes and each check.
`engine_validation.py --replay DIRECTORY` checks hashes, reruns the saved scene
and state, re-extracts the matrices and resolves the reference problem.
Any mismatch is an explicit failure. SHA256 identifies bytes; it does not prove
correct physics. Engine agreement, analytical checks and replay provide evidence.
Generation appends every certification success/failure to
`data/engine-validation-attempts.jsonl` and writes a run note.

## Public, Validation and Private Sets: Required Next Work

1. Public development cases can drive detailed model feedback. Never pass
   reference outputs into candidate solver arguments.
2. Split by complete scenes/trajectory groups and problem fingerprints, not
   filenames alone. Reserve unseen configurations and contact counts; fresh
   seeds alone do not establish topology generalization.
3. Validation may guide candidate selection. If private failures guide revisions,
   that set becomes validation and a fresh final test set is required.
4. Freeze candidate source hash, checker version, thresholds and public report
   before final private evaluation. Public correctness gates must all pass.
5. Final private cases and detailed results stay evaluator-side, inaccessible
   to candidate code and the model. A folder named private is not a security
   boundary; use separate access-controlled execution/storage.

The initial 12-case extraction validation remains development-only.
`build_engine_sets.py` now creates a separate split: 24 deduplicated public cases
(plane/pairs/stack; counts 1/4/8; seeds 2/3/4) and 33 deduplicated private cases
with reserved counts and seeds plus three disconnected vertical stacks.
These do not cover arbitrary branching contact graphs, friction or trajectories.
Both sets are independently certified and replayable. Exact problem fingerprints
are disjoint, and frozen manifests/references are hash-committed. Equivalent
problems under different coordinate transforms are not comprehensively detected.
Certification of private references is not tuning candidate code on private data.
The developer can access the private artifacts; they are model/candidate-private,
not claimed to be organizer-blind. Private candidate execution is not wired yet.

Only `public/` is distributable to candidate environments. `evaluator-only/`
contains all reference answers and every private input/state/scene. The public
contract, public manifest, private manifest and reference index are bound in
`commitment.json`. Local restrictive permissions are not an isolation guarantee.
The `public_gate` helper checks a trusted controller's complete, all-pass report
against candidate hash, suite ID and contract hash. It must not trust a report
written or altered by candidate code. It does not yet execute private evaluation.

## Agent Feedback and Performance

Separate input validity, compilation, numerical execution, physics convergence
and hardware timing failures. For development report per-gate errors and their
progress over iteration budgets. A valid alternative algorithm need not match
every projected-gradient update; it must solve the same physical problem.
Use fixed-update equivalence only when preserving that specific algorithm.
Never use compilation success or simulator runtime as a hardware speed reward.

Primary future performance measure: time to an independently checker-approved
solution, including candidate-specific preprocessing and production stopping
checks. Report device-only and end-to-end timings separately. Compute oracles
offline; do not hide candidate CPU work outside the timing. Freeze warmstart,
reuse and batching assumptions equally across candidates. Record unsuccessful
solves rather than averaging only fast successful cases.

Log every model attempt: prompt, raw response, endpoint/model and decoding settings,
candidate/parent hash, suite/checker hash, failures, physics scores, feedback,
token usage where available, and measured timing samples. Repeat model loops and
hardware measurements, reporting counts and spread. No such model result exists yet.

## Sources

- MuJoCo contact computation: https://mujoco.readthedocs.io/en/stable/computation/
- MuJoCo data fields: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
- Project Chrono APGD implementation: https://github.com/projectchrono/chrono/blob/main/src/chrono/solver/ChSolverAPGD.cpp
- Adaptive restart research: https://arxiv.org/abs/1204.3982
- AWS agent tools: https://github.com/aws-neuron/neuron-agentic-development
