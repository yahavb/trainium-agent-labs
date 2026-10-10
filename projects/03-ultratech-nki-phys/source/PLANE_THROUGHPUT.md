# Qwen Plane-Contact Throughput Task

The hackathon time-limit pivot targets synthetic frictionless contact impulse
QPs. Primary workload: plane scene, 33 active contacts padded to 64, eight
independent worlds, exactly 32 projected-gradient updates. Pair scenes provide
additional coverage. These are not MuJoCo gripping exports or full rollouts.

Qwen now uses qwen_grip.py --task plane-throughput, despite the historical
script name. The gripping default is unchanged. plane-feedback.txt supplies
human guidance to try copy removal and supported instruction fusion; this is
not an autonomous discovery claim. Every generation preserves its task name,
request, feedback, SDK metadata, response, source and hashes in data/qwen-plane-*.

Compare candidate and baseline on the same scene, contacts, seeds, batch, steps,
core allocation, warmups and repeated timing protocol. Do not credit increased
batching against an already B=8 baseline as a kernel improvement. Optimize the
same recurrence, not a new solver algorithm. Report compilation separately.

Generated source is not automatically executable. Review its imports, entry
point, side effects, tensor operations and recurrence before providing its hash
to check_batch.py --candidate ... --candidate-sha256 ... . Candidate execution
automatically enables the full-physics gate. Baseline comparisons must use
--require-physics as well. Existing beta-only grip_loop.py is not this task's
controller and should not be used for unrestricted throughput candidates.

check_batch.py records independent oracle physics checks, fixed-update
equivalence, padding and input preservation before and after timing. It now
snapshots source files and saves pre/post outputs plus CPU expected values.
All attempts and timing samples are logged. The hash identifies reviewed source,
not a security sandbox or remote attestation.

Local preparation: strict plane CPU baseline passed 8/8 at 32 steps; strict
stack control failed before benchmarking, as intended. The task prompt dry run
succeeded. No Qwen candidate has been generated or benchmarked for this scope
yet. There is no new performance improvement to claim at this stage.

Only confirmed free cores may be benchmarked. If stopping the Qwen server to
free cores, generate candidates first, then stop it; do not time kernels against
an active model on the same allocation. Device timing excludes compile/load;
host timing uses preallocated tensors and excludes packing/transfers/readback.
Use at least five repeats and report median/min/max plus stored raw samples.

## Restricted Automatic Loop

Run plane_loop.py --minutes 10 --attempts 4 from the laptop physics environment.
It uploads its dependencies, requests Qwen revisions, rejects unreviewed AST
forms and runs the candidate against both plane and pair scenes in the NKI CPU
simulator. It stops at a fully passing candidate or the budget. Every attempt
is retained. Accepted means correctness passed, not performance improved.

Only the unchanged baseline, reviewed copy-removal form and reviewed
multiply/add subtraction form are executable automatically. Local-variable
renaming and comments do not bypass the gate or count as a new program.
Qwen is given the human-reviewed template explicitly; this is a constrained
implementation experiment, not autonomous discovery. Broader optimizations
still require review or genuine isolation. Device timing is a subsequent stage.

An initial review-gate bug rejected otherwise approved forms when Qwen omitted
the module docstring. Canonical comparison now ignores module/function docstrings,
comments and local variable names, but retains executable statements. Saved loop
c81eecaf61d8444ab11378d6a8bf2329 attempts 0 and 1/2 match the reviewed fused and
copy-removal forms respectively. Original rejection records remain unchanged;
this static correction does not supply missing physics or timing measurements.
