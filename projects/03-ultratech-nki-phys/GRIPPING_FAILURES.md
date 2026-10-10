# Gripping Failures And Why We Narrowed The Agent Experiment

## Original Physics Task

Compute normal/friction contact-force coefficients for a simplified gripping configuration, using exported MuJoCo contact snapshots and a trusted higher-precision solution. This is a coupled constrained solve, not the final independent spring equation or force sum. Acceptance checks included optimality residual, force accuracy, reconstructed contact wrench, generalized acceleration and feasibility. Passing an update-equivalence test did not imply convergence to a physically acceptable solution.

## Six Recorded Trials

Source run: `grip-loop-6d62f48aef734d9bbdf4824307e2ec71`; NKI simulator on seat-260; 16 public cases, batch 2, 1024 updates, one execution per group. These are correctness trials, not device throughput measurements.

| Attempt | Momentum beta | Cases passing all physics gates | Fraction | Accepted throughput |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.9 | 10/16 | 0.625 | None |
| 1 | 0.95 | 10/16 | 0.625 | None |
| 2 | 0.98 | 10/16 | 0.625 | None |
| 3 | 0.99 | 6/16 | 0.375 | None |
| 4 | 0.995 | 6/16 | 0.375 | None |
| 5 | 0.999 | 4/16 | 0.25 | None |

The [original six-row log](gripping-evidence/ATTEMPTS.jsonl) is preserved separately from the final math experiment. Each `gripping-evidence/attempt-NNN/results.json` contains every case's before/after checks and diagnostic feedback. Available candidate files and next prompts are included. This is a curated record of this six-trial run, not every earlier gripping generation, output array or benchmark. Historical absolute paths in the log point to the original workspace; portable grading reports are beside this document.

## Concrete Failures From Attempt 0

Saved diagnostic report: [attempt-000/results.json](gripping-evidence/attempt-000/results.json).

- **grip-000:** projected residual `9.136249e-6` and objective error `6.673453e-11`; numerical subcheck passed, but generalized-acceleration error `0.0004568481444` exceeded the `0.0001` gate. Thus, a small optimization error was not sufficient to accept the physical force output. Feedback explicitly requested reduced force/acceleration error.
- **grip-002:** projected residual `0.0016767216604`, edge-force error `0.0028473360839`, acceleration error `0.0048339843688` and contact-wrench error `0.0029021835937` exceeded their `0.0001` gates. Feasibility was true and cone violation zero; those properties alone did not make the solution accurate. Feedback identified convergence/step-size/FP32 stagnation and coupled force accuracy as investigation targets.

These were measured physics failures, unlike an API/compilation error whose score remains `null`. Increasing momentum did not repair the failures and eventually reduced the number of passing cases.

## Precision Investigation And Decision

A human-written adaptive-restart CPU solver also plateaued at 10/16 in FP32 at the tested 1024/4096 update budgets. Higher-precision working arithmetic passed 16/16 on the same exported inputs. That points toward numerical precision/conditioning rather than merely different MuJoCo platforms, but it does not prove that FP32 cannot solve the task or isolate one universal failure cause. The investigation is documented in [source/ADAPTIVE_RESTART.md](source/ADAPTIVE_RESTART.md); its full separate raw history is not bundled here.

**Because of the hackathon time limit, we shifted to simpler physics primitives to complete and evaluate a working correctness-plus-throughput agent. We did not loosen checker tolerances or call the gripping task solved.** Gripping remains future work requiring a more accurate solver, suitable conditioning/precision choices and new independent validation before any accepted speedup claim.
