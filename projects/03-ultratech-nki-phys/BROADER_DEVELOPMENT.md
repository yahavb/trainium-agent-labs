# Broader Development: Toward Physics-Engine Primitives

**We developed this broader prototype as Team Ultratech; source: our [six-page engine report](submission-assets/teammate-engine-report.pdf).** This document summarizes that report. Its underlying engine implementation and full raw logs were not independently inspected or reproduced in this submission package.

## What Was Developed

| Primitive | Physics computation | Intended checker evidence |
| --- | --- | --- |
| Semi-implicit Euler integration | Update velocity from force, then position from updated velocity | Discrete reference and convergence behavior |
| Coupled spring-chain forces | Neighbor-dependent restoring forces | Force/energy-gradient consistency |
| Floor-contact projection | Prevent a step from crossing the floor using a velocity constraint | Contact feasibility and constrained reference |
| Fused rollout | Keep state on chip across repeated force/integration/contact steps | Rollout accuracy and transfer/launch accounting |

The coupled spring chain is different from our final independent spring-damper force primitive. The report describes a `World` interface for a force/contact step and a fused 32-step rollout, with independent worlds batched across partition rows. These reference primitives explore the larger simulation direction; they are not presented as fully solved Qwen-generated levels.

## Reported Validation And Performance

- Four SciPy reference certifications, four accepted reference kernels and 12 correctly diagnosed planted physics bugs in NKI simulation.
- A 128-world, 16-mass, 64-step test compared with a float64 discrete reference: reported RMS error approximately `4e-5`, no floor penetration and no energy gain in that particular tested scenario. This is not a theorem that the integrator can never create energy.
- Approximately **51x lower simulation-counted DMA traffic** for the reported fused comparison. This is transfer accounting, not a measured hardware-bandwidth speedup.
- Approximately **52x host-to-host speedup** in a reported 32-step rollout comparison on seat-263, NeuronCore 2: about 102 seconds unfused versus 1.96 seconds fused. The report attributes large standalone-SDK launch overhead to the result; device-only execution time was not isolated.

Do not compare the 52x directly with the narrow agent's independent 1.038735x/1.075468x device-throughput ratios: the workload, baseline and timing boundaries differ. The report also needs clarification of a listed 1.5x traffic-floor gate versus a 2.0x floor in a rollout result before declaring that particular performance gate passed.

## Where The Broader Agent Failed

The report states that Qwen3-8B did not fully solve the physics levels in the reported search. Feedback iterations produced partial progress but unresolved output errors remained, including a stray velocity offset and an incorrect rollout slice. The working reference engine must therefore not be described as an entirely agent-generated successful engine.

## How The Two Tracks Fit Together

The narrow track supplies a verified model-proposal/checker/timing loop and frozen evaluation. The broader track supplies a prototype set of simulation primitives, richer physical diagnostics and an integration direction. Together they motivate an agent that selects and verifies optimizations for more of the simulation pipeline. They do not yet establish an agent that optimizes a complete MuJoCo-equivalent engine.

To promote this appendix to a separately reproducible broad-track deliverable, provide the engine source, input builders, checker registrations, full attempt history and raw timing/traffic records described in the report. Only the report itself is currently packaged for this track.
