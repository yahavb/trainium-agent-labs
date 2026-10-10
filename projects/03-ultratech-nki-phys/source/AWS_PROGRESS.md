# AWS Hackathon Progress Walkthrough

## One-Sentence Project

We are building a checker-guided agent that uses Qwen to propose Trainium NKI
physics-computation kernels, rejects incorrect solutions, measures device throughput,
and uses correctness/performance feedback to guide the next proposal.

The deliverable is the agent workflow and trusted checker, not a newly trained
model. Qwen weights are unchanged. This is inference-time code revision/search.

## Problem and Current Scope

Example: bodies collide with a flat surface; compute nonnegative contact impulses
so the contact solve satisfies its mathematical and physical conditions. We test
synthetic frictionless plane and pair snapshots, not complete moving simulations.
Current workload: eight independent worlds per scene, 33 contacts per world,
padded to 64, with exactly 32 projected-gradient updates.

The shared solver repeatedly computes x_next=max(0,x-alpha*(A*x+b)). Different
worlds have their own matrices, biases and step sizes. Optimization must preserve
this calculation, inputs, output padding and accuracy requirements.

We have now added two different equations, not merely more parameter values:
spring-damper forces F=-(k*x+c*v), and net-force accumulation F_net=sum(F_i).
These are elementwise and reduction workloads, respectively, whereas contact
solving is iterative matrix-vector work. Each new task has 16 deterministic
public cases and a separate frozen-input FP64 reference checker. These are small
physics calculations, not full simulator or robotics-task demonstrations.

## What We Built

1. Explored seat-260: Trainium2 hardware, installed SDK, quotas and compile/run smoke test.
2. Built CPU reference solvers and deterministic physics checks, including planted-failure tests.
3. Created a handwritten NKI baseline; verified fixed-update agreement separately from physics convergence.
4. Measured independent-world batching; the original worlds run serially within one launch.
5. Created MuJoCo-backed frictional gripping fixtures with independent reference certification.
6. Ran Qwen revisions with measured checker feedback and an auditable attempt history.
7. Added reviewed-code gates, automatic generation/checking, and documentation retrieval.
8. Added device throughput feedback, paired baseline timing, repeat spread and best-candidate retention.
9. Added model-driven operation-graph proposals across distinct math tasks; trusted code lowers them to NKI.

## Why We Changed Scope

Because of the hackathon time limit, we shifted the active deliverable from
frictional gripping to the simpler passing plane/pair cases. Gripping's fixed-
momentum search reached 10/16, then regressed as momentum increased. A human-
implemented adaptive-restart CPU reference also reached 10/16 in FP32 at 1024
and 4096 updates. Higher-precision working arithmetic passed 16/16 on the same
exported inputs. This implicates numerical precision but does not prove that an
FP32 gripping solver is impossible. The new adaptive NKI kernel is not yet
seat-validated. Gripping failures remain recorded; we did not relax their gates.

## Agent Flow

Task + baseline + prior proposal + retrieved NKI guidance + previous measured feedback
-> Qwen proposal -> source review -> correctness checks -> device benchmark
-> record score/errors/timing -> next proposal -> retain best verified implementation.

The checker is deterministic code, not an LLM judging its own output. The
controller is the agent: it coordinates generation, retrieval, tests, timing and
revision. Incorrect or duplicate proposals receive no performance reward.
Qwen is stopped during timing on the shared chip and restarted for new proposals.

## Current Measured Results

| Implementation | Correctness | Device Throughput | Ratio to Paired Baseline |
|---|---|---:|---:|
| Original baseline, copy-removal comparison | All required checks pass | 26,218.53 worlds/s | 1.00000x |
| Qwen copy-removal proposal | All required checks pass | 26,573.52 worlds/s | 1.01354x |
| Earlier fused-subtraction proposal | All required checks pass | 21,211.84 worlds/s | 0.80905x |
| Copy-removal independent repeat | All required checks pass | 26,572.31 worlds/s | 1.01349x |
| Scaling/update fusion | All required checks pass | 26,496.74 worlds/s | 1.01058x |

The copy-removal proposal measured a 1.354% throughput gain in one experiment.
Plane and pair gains were 1.312% and 1.396%. Five candidate timing repeats per
scene, 200 samples per repeat; paired baseline drift was 0.0130% and 0.0403%.
This is an observed kernel-level gain, not independent-run replication,
statistical significance, global optimality or full-simulator speedup.

Update: independent run 9f23118d8ff945a9a99c892d07271528 repeated copy-removal's
gain at 1.349%, closely matching the earlier 1.354%. Scaling/update fusion passed
but measured 0.99703x relative to its paired copy-removal reference (0.297%
lower throughput), so the agent retained copy-removal. The table's ratios are
to each experiment's paired original baseline, not one shared timing sample.
Replication strengthens the observation; statistical significance and broader
workload generalization remain unestablished. The first-result caveat above
describes the evidence available before this independent repeat.

Device timing excludes compile/load. Host timing uses preallocated tensors and
excludes packing, transfers and readback. Baseline and candidate use the same
workload/cores; correctness is checked before and after timing. More than 10%
baseline latency drift invalidates the performance reward.

## What Copy Removal Changed

The baseline copies the matrix-vector product from PSUM to a temporary SBUF
tile before adding the bias. The candidate adds the SBUF bias directly to the
PSUM result, removing that intermediate copy without changing the mathematics.
Qwen was explicitly given human optimization guidance and reviewed templates;
we do not claim it independently discovered the technique.

## New Model-Driven Math Results

Run: data/math-agent-f6b07d70f80843839970831aee0139b4, seat-260, cores 0,1.
This loop gives Qwen the equation, shapes, original baseline and prior feedback;
it does not prescribe the next exact template. Qwen chooses and arranges
operations and states a hypothesis. A bounded operation language is validated
and lowered by trusted code; generated Python is not executed directly.

| Attempt | Task | Model Proposal | Correctness | Throughput Versus Original |
| --- | --- | --- | --- | --- |
| 0 | Spring-damper force | Fuse negative signs, then an extra final negation | 0/16; rejected | Not measured |
| 1 | Net-force accumulation | Replace serial additions with native sum | 16/16; pre/post gates pass | 1.07042797x (+7.043%) |
| 2 | Spring-damper force | Restore correct sign; explicit Vector Engine placement | 16/16; pre/post gates pass | 1.03981028x (+3.981%) |
| 3 | Net-force accumulation | Repeat the earlier native-sum graph | Duplicate; not evaluated again | Not measured |

The spring proposal computed -(k*x) and -(c*v), summed them, and then negated
again. Its output had the opposite sign to the required force. This was a real
mathematical failure, not small platform rounding. The checker rejected it and
the controller withheld timing; the failed proposal and numerical errors remain
available as feedback for the next spring attempt.

Attempt 2 returned to spring with its earlier failure feedback and previous
proposal. Qwen changed the first two operations from mul_neg to mul, retained
the final negation, and explicitly selected Vector Engine for arithmetic. The
result now evaluates -(k*x+c*v) correctly: all 16 cases passed. Its source SHA256
is 68872332a2fcf9d5f03506f749e9d43482fbf131079706ff5c19c8c4e17cba20.
Observed paired-original throughput ratio was 1.0398102810x (+3.981%); baseline
drift was 0.05059%. This is a recorded failure-to-correct revision in the agent
loop. It uses the same four arithmetic steps as the original, not a new physics
algorithm or fewer-operation fusion. Explicit placement is a source difference,
but an isolated ablation/profile is needed to attribute the speed gain to it.

The net-force proposal used one sum operation, lowered to nisa.tensor_reduce,
instead of the original serial-addition loop. Its source SHA256 is
cc18d809141ee45b7fbe81c8358c95d2e63824cdf8031e7c4ee1fbffe153bfe3.
Candidate mean device latency was 0.01734891 ms; the pooled paired original
baseline mean was 0.01857076 ms. The observed throughput ratio was 1.0704279692x.
Baseline-before/after drift was 0.6061%, below the unchanged 10% validity guard.
Five candidate repeats, 200 samples per repeat and 20 warmups were used.
Candidate repeat means ranged from 0.01731500 to 0.01738581 ms.
All 16 cases were checked; only seed 3 was timed. Qwen was stopped for timing.

This is one model-selected, hardware-measured proposal on one reduction workload.
It is stronger than an unmeasured hypothesis, but is not independent replication,
statistical significance, private-test generalization or full-simulation speedup.
The 7.043% figure is a throughput gain, not an equal percentage latency decrease.
The four-attempt run is complete. Attempt 3 returned the same executable native-sum
graph and source SHA256 as attempt 1, so the controller rejected the duplicate
without new execution or timing. Its null correctness/throughput fields mean
unmeasured for that attempt, not a failed equation check. Final retained winners
are spring at 1.03981028x and net-force at 1.07042797x versus their respective
paired original baselines. Two measured correct proposals, one mathematical
failure, and one duplicate are all retained; the duplicate is not replication.

## Current Limitations

The contact loop accepts a small set of human-reviewed AST forms. The new math
loop accepts model-proposed graphs in a finite operation language, not arbitrary
generated kernels or unrestricted algorithm discovery. Retrieval in the contact
loop is keyword search over curated,
source-linked documentation cards, not live autonomous browsing or embeddings.
No weight training, unrestricted optimization or isolated execution sandbox is
implemented. The current throughput scenes are public development cases; we
have not demonstrated private-test generalization or integrated a simulator.

## Evidence and Hand-In

Checker/reasoning: CHECKER.md, contact.py, check_batch.py and the gripping checker docs.
Agent/provenance: full_plane_loop.py, qwen_grip.py, optimization_knowledge.py.
New math agent/checkers: math_agent_loop.py, qwen_math.py, math_program.py,
math_tasks.py, math_harness.py and DISTINCT_MATH.md.
New math attempts: data/math-agent-f6b07d70f80843839970831aee0139b4/attempts.jsonl.
Native-reduction evidence: that run's attempt-001/device/net-force/timing.json,
results.json and baseline/agent-proposal/post-timing-suite/checks.json.
Copy-removal experiment: data/full-plane-loop-ca51b1815ec0403caf90df1234bb9aa6.
Replication/new fusion: data/full-plane-loop-9f23118d8ff945a9a99c892d07271528.
Fused slowdown/duplicates: data/full-plane-loop-fb1b0f0aa75c4645ae36731a27f7181a.
Gripping history and pivot: QWEN_EXPERIMENT.md and ADAPTIVE_RESTART.md.
Each run retains attempts.jsonl, source/output snapshots, detailed reports,
raw timing samples and run-note.md. Data directories are ignored by Git and
must be included explicitly in the final evidence bundle; local presence is not backup.

## Next Milestone

Final evaluation completed successfully on seat-260, cores 0,1: frozen bundle
data/final-math-428948fb229d4c648bae14c17f9235b1 contains the exact selected
spring and net-force sources, their hashes, pinned checker/lowerer sources, and
32 unique unseen input snapshots per task (64 total). Development used seeds
0..15; holdout generation starts at 10000 and excludes exact input duplicates
of development and other holdout cases. Equations, shapes, distributions and
accuracy gates remain unchanged. This tests new values, not unseen algorithms.
frozen_math_eval.py stops Qwen, performs one fixed-kernel invocation per case,
downloads outputs and grades locally. It makes no model calls, sends no feedback,
and refuses a second assessment of the same bundle. Spring passed 32/32 and
net-force passed 32/32: 64/64 measured held-out cases passed, with unchanged
accuracy thresholds and no generation or feedback during assessment. Report:
data/final-math-428948fb229d4c648bae14c17f9235b1/assessment/grading/results.json.
Manifest SHA256: 2de67353ccb88bce02cd76d311ed37b56c2b7543e0145b71bd701f034509ce59.
Development speedups were not remeasured by this correctness-only assessment.

The four-attempt math-agent run and frozen-case assessment are complete. Next
independently replicate the spring and native-reduction performance gains and
package the checker/reasoning, complete attempt history and one-page run note.
Compare model-driven decisions with a fixed-template script under the same
budget for a stronger agent-specific evaluation. Local tests pass (95); completed
hardware evidence above is distinct from mock/controller tests.

Broaden the agent's reviewed optimization choices using documentation and device
profiles. Scaling/update fusion has now been tested and did not beat copy-removal;
the winning benchmark has been independently repeated. Next obtain a device
profile before choosing an engine/layout or scheduling transformation.
The final held-out cases are now assessment evidence, not a new revision set.

## Independent Benchmark Replication

Completed on the same frozen kernel sources, without Qwen or held-out reruns.

| Task | Initial Throughput Ratio | Independent Ratio | Independent Candidate Repeat Means (Min/Median/Max ms) |
| --- | ---: | ---: | --- |
| Spring | 1.03981028x | 1.03873494x (+3.873%) | 0.015456995 / 0.015461330 / 0.015469085 |
| Net-force | 1.07042797x | 1.07546770x (+7.547%) | 0.017342015 / 0.017345200 / 0.017361155 |

Each independent comparison used five repeats, 200 device samples per repeat,
20 warmups, the original seed-3 workload and paired original baseline measurements.
Baseline drift was 0.01799% for spring and 0.06042% for net-force. Source hashes
match the selected kernels and final-evaluation manifest. Both earlier gains
replicated; this is repeatability evidence, not a statistical-significance claim.
Raw timings and pre/post checks are retained in the independent-repeat directory.

## Submission Packaging

submission_math.py is prepared to independently repeat both frozen winners on
the original development benchmark and then produce an evidence ZIP. It does
not call Qwen or re-execute held-out inputs. Each repeat uses five runs and 200
samples with paired original baseline timings, checks and candidate source hashes.
Replication is now complete with the observed results above.

A provisional package is available at
data/submission-math-30ed602c35bc4a4f917c214263480ede.zip, with a 339-word run note,
checker/source/docs, all four math-agent attempts, frozen final evaluation and
file checksums. ZIP integrity was checked. Compiled NEFFs are excluded; source
and SDK requirements are retained for reproduction. The package labels the
independent repeats as pending, and its complete-history claim is limited to
the four-proposal distinct-math run; older contact/gripping history is documented
but not all earlier raw artifacts are bundled. A successful --repeat invocation
will create a new package with the independent results and repeat spreads.

Completed package: data/submission-math-1737b37499f845ca801b1f58b161cef0.zip.
It includes RUN_NOTE.md, checker and acceptance reasoning, all four math-agent
attempts, 64/64 held-out evaluation, independent repeat reports, raw samples,
sources and file checksums. ZIP integrity was verified. The repeat evidence is
independent-repeat/summary.json within that package. Local tests pass (96).
Submission claims should remain limited to these two bounded math workloads,
held-out input values and replicated kernel-only throughput measurements.
