# CHIPBOOST Qwen results — October 10, 2026

The deliverable comparison uses Qwen/Qwen3-8B only, with expert-template random search as the non-model control. P1 acceptance and the original 72-evaluation comparison are complete. Neither original Qwen arm found a verified speedup; the template control did. The separate consolidated Qwen-only v2 run remains in progress, with a provisional single-candidate `faster` observation around 1.517x. Its final results and replay evidence remain pending here.

## Evidence by experiment

| Experiment | Current evidence | Interpretation |
| --- | --- | --- |
| P1 throughput acceptance | Nine targeted checks passed; pinned A/A retry on core 3 returned `no_gain`, 1.000049x | Referee execution/acceptance evidence, not an optimization discovery |
| Original Qwen comparison | Complete: 72 records, 24 per arm, three repeats of eight; strict report passed | Original feedback only; no Qwen `faster` outcomes; keep separate from later treatments |
| Qwen recovery pilot v3 | Complete: 8 graded attempts, all `wrong`, zero `faster` | Full-feedback plus repair-controller exploratory treatment; no valid improvement found |
| Qwen reasoning run | Canceled for the deadline with zero graded attempts | No optimization result; do not count cancellation as a kernel failure |
| Queued DMA-only v2 | Canceled as superseded; no comparison results | Its earlier message-selection replay is not a model outcome |
| Consolidated Qwen-only v2 | Running on core 2 with integrated P3, `--tag v2`, budget 8; PID `884773`; output `/tmp/p1-qwen-v2-20261010-1`; attempt 3 reported `faster` around 1.517x | Provisional single observation; final batch and unchanged-source replay pending |

The recovery pilot's collected evidence is currently in the repository workspace staging directory `.integration-prep/recovery-pilot-complete/`: `state.json`, `acceptance.json`, and `pilot.jsonl`. Preserve those records in the integrated experiment artifacts before publishing a portable report.

## Completed original Qwen comparison

The [strict report](experiments/qwen-v1-comparison/report.md) and [machine-readable report](experiments/qwen-v1-comparison/report.json) validate all nine runs. The passed acceptance gate precedes 72 evaluations on core 3, using one worker (`worker_starts=1`). The historical busy-core interruption is a separate infrastructure event, excluded from attempt and failure counts.

| Arm | Attempts | `wrong` | `no_gain` | `slower` | `faster` | Median best verified speedup across runs |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen + original referee feedback | 24 | 22 | 2 | 0 | 0 | 1.000000x |
| Qwen model alone | 24 | 2 | 22 | 0 | 0 | 1.000000x |
| Expert-template random-search control | 24 | 0 | 0 | 2 | 22 | 3.125919x; run-best range 2.525760–3.323522x |

No records have `rules` or `heldout_fail` verdicts. Runs with no verified improvement retain the 1x baseline in the best-speedup summary. Kernel rejection rates were 22/24 and 2/24 for the two Qwen arms, respectively; these are descriptive results from three small runs, not proof that feedback generally harms performance. This older referee supplied the feedback later corrected. The template control starts from a substantially different candidate prior, so its gains are not Qwen-generated improvements or an isolated feedback effect.

## Feedback and integration scope

Current work includes outer-loop reuse diagnosis, conservative fallback feedback, generalized DMA mismatch guidance, and PSUM/compiler repair guidance, pushed in `7da33ee`. Local verification with `python -m unittest discover -s projects/03-chipboost/tests -p 'test_*.py'` passed all 36 tests; this is implementation coverage, not evidence of model improvement. The earlier four representative chip checks are in `feedback_validation.json`. Final seat/core-2 validation in [final_feedback_validation.json](research/final_feedback_validation.json) confirms the baseline `no_gain` and all three named failure instructions: DMA mismatch, NKI tile-list representation, and PSUM lifetime. Failure cases remain `wrong`; these instruction-selection checks are not optimized-kernel results.

The original comparison remains pinned to the older referee/agent snapshots and does not measure these later feedback fixes. Recovery v3 changes both feedback and the repair controller; its unsuccessful eight-attempt pilot does not isolate either change's causal effect. Keep all experiment cohorts separate.

## Final comparison handoff

The original comparison files are collected in `experiments/qwen-v1-comparison/`, including all nine logs, state, acceptance, infrastructure, and strict reports. Strict validation passed for 72 records, 24 per arm, three completed runs per arm, matching pinned versions, and the acceptance gate. Keep the running consolidated v2 results in a separate directory and report; do not overwrite or extend the completed v1 records.

Build the dashboard with explicit comparison log paths and separately verified adversarial results. Infrastructure interruptions and canceled jobs are not kernel attempts. P2 searches over an expert-template prior while agent arms start from the reference; preserve that limitation in the final report. The earlier dashboard inspection was static only because browser preview was blocked by the URL security policy.

## Archive appendix — excluded from the Qwen deliverable comparison

These prior side experiments are retained for provenance only. They are excluded from the Qwen headline results, comparison aggregates, and dashboard inputs. Their different generation context and workflow do not support an equal-budget model comparison or a model ranking.

| Archived case | Recorded speedups |
| --- | --- |
| Astra-labeled A | Initial 1.5170524x; unchanged-source repeats 1.5172862x, 1.5173145x |
| Astra-labeled B | Initial 1.2033504x; unchanged-source repeat 1.2027148x |
| Astra-labeled C | 2.55416996x and 2.551280807x |
| Syntax-only repair diagnostic | 1.1469457x following an earlier compilation failure |

The A/B records are in [initial_results.json](experiments/astra-trial/initial_results.json) and [replication_results.json](experiments/astra-trial/replication_results.json); C records are in [candidate_c_results.json](experiments/astra-trial/candidate_c_results.json). [GENERATION.md](experiments/astra-trial/GENERATION.md) records generation provenance. The Astra label was supplied by delegation and is not independently attested by these artifacts. Initial A/B checks used referee SHA-256 `429309eb48c515ead0d6bd50b28c432f2c69c089a1992eeee9b9b87333714a5b`; A/B replications used original referee SHA-256 `5f366558ae933c88262bbde806afbcd4a568fdbfab4fc3395951f1b3799d71be`.
