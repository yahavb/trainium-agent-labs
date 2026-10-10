# Gripping Contact-Solver Task

## Physics and Scope

A free sphere is pinched between two fixed box pads. Gravity acts downward.
We vary mass (0.5/2 kg), sliding friction (0.2/0.8), initial penetration
(0.0002/0.001 m), and downward velocity (0/0.2 m/s): 16 development cases.
Pads are prescribed fixed geometry, not controlled finger joints. This is not
a robotic policy, a stable-hold test, a trajectory, or a measured physical grasp.
Sphere geometry and condim=3 omit torsional and rolling friction. Rotation about
the grip axis is not resisted by a torsional contact model.

MuJoCo 3.15.0 uses its regularized soft-contact model with pyramidal friction,
condim=3, Newton solver, 200 iterations, tolerance 1e-12, disabled warm-start,
dense Jacobian, timestep 0.002, solref .02 1 and solimp .9 .95 .001.
The chosen penetrations prescribe squeeze indirectly, not a measured actuator force.
Pyramidal and elliptic soft-contact models are different; we do not assert
equivalence to an elliptic solver or ideal Coulomb sticking.

Source: https://mujoco.readthedocs.io/en/3.15.0/computation/#friction-cones

## Candidate Interface

Input: FP32 matrix A and vector b; output: eight nonnegative pyramid-edge forces.
Each of two contacts has four edges. Minimize 0.5*f.T*A*f + b.T*f subject to f>=0.
A = J*M^-1*J.T + diag(efc_R); b = efc_b. Positive regularization makes A SPD.
Edge values are not directly normal/tangential force components. A separately
verified decoding matrix reconstructs contact-frame forces for the checker.
For this model: normal=sum(edges), t1=mu1*(edge0-edge1),
t2=mu2*(edge2-edge3); no contact torque components.

Existing NKI matmul/project/clamp mechanics are structurally reusable, but have
not been executed on these gripping cases. Existing fixed-step timing is not
evidence of gripping correctness or convergence. Qwen generation has not run.

## Checker Reasoning

Independent FP64 Cholesky plus SciPy NNLS solves the same force QP. Certification
requires normalized force, acceleration and reconstructed-wrench discrepancies
against MuJoCo <=1e-7 and oracle projected residual <=1e-9. Extraction additionally
checks b=J*qacc_smooth-efc_aref, generalized force J.T*f, positive definiteness,
constraint type/dimension, and decoded wrenches against mj_contactForce.

Candidate acceptance: shape and finite outputs; edges >=-1e-6; normalized
projected residual <=1e-4; normalized objective discrepancy <=1e-5; normalized
edge-force, generalized-acceleration and contact-wrench errors <=1e-4;
normalized pyramidal cone violation <=1e-6. Normalization denominators are
max(1, reference maximum absolute value); residual uses max(1, max(abs(b))).
The cone check uses |t1|/mu1+|t2|/mu2<=normal. Passing frictionless tests does
not substitute for these checks. Rejection tests include removing tangential
forces while preserving normal force, negative/zero/NaN/wrong-shape outputs,
unsupported elliptic contacts, and artifact corruption.

## Logs and Reproduction

Run generate_grip.py --out data/gripping-development-v1 --runs 5 in the physics
environment. Existing directories are never overwritten. It creates scene XML,
public FP32 A/b inputs, evaluator-only full reference arrays, saved baseline
outputs, certification-attempts.jsonl, attempts.jsonl (every scored baseline
attempt), source/artifact hashes, commitments, and a one-page run-note.md.
Run generate_grip.py --replay data/gripping-development-v1 to re-extract all
physics, certify references, and independently reproduce every saved score.

The initial CPU baseline uses FP32 projected gradient from zero, residual-only
stopping at 1e-7 and an 8192-step cap. Returning does not imply acceptance;
failed attempts remain logged. No oracle is used by its stopping rule.
CPU timings include eigenspectrum preparation and stopping checks. They are
diagnostic, not controlled Trainium speed comparisons. Accelerator benchmarks
must measure baseline and candidate at the same accuracy, batch sizes and
allocation, with repeated device/host timings and separately reported compilation.

All 16 cases are public DEVELOPMENT cases, separate from engine-sets-v1. Do not
merge them into its previously frozen private split. Only expose public/ to
candidate code; references and evaluator files must be outside that environment.
Local file permissions do not isolate same-user generated code.
Before final evaluation, freeze a separate unseen gripping test suite and checker.
Next: execute NKI baseline on these inputs, then Qwen generation/public-check/
feedback/revision. No fine-tuning or end-to-end MuJoCo speedup is claimed.
