# CHIPBOOST Qwen results — October 10, 2026

The deliverable comparison uses Qwen/Qwen3-8B only, with random search as the non-model control. P1 acceptance passed. The original three-arm comparison is still in progress at this documentation snapshot, and the consolidated Qwen-only v2 run has launched; neither is claimed complete here. The completed recovery pilot found no valid improvement.

## Evidence by experiment

| Experiment | Current evidence | Interpretation |
| --- | --- | --- |
| P1 throughput acceptance | Nine targeted checks passed; pinned A/A retry on core 3 returned `no_gain`, 1.000049x | Referee execution/acceptance evidence, not an optimization discovery |
| Original Qwen comparison | P1 `434e5f9`, P2 `919c6be`, P3 `2ce9416`; planned 3 arms × 3 repeats × 8 evaluations; still running | Wait for strict final report; do not combine partial data with later treatments |
| Qwen recovery pilot v3 | Complete: 8 graded attempts, all `wrong`, zero `faster` | Full-feedback plus repair-controller exploratory treatment; no valid improvement found |
| Qwen reasoning run | Canceled for the deadline with zero graded attempts | No optimization result; do not count cancellation as a kernel failure |
| Queued DMA-only v2 | Canceled as superseded; no comparison results | Its earlier message-selection replay is not a model outcome |
| Consolidated Qwen-only v2 | Launched on core 2 with integrated P3, `--tag v2`, budget 8; PID `884773`; output `/tmp/p1-qwen-v2-20261010-1` | Separate treatment using the consolidated feedback; results pending |

The recovery pilot's collected evidence is currently in the repository workspace staging directory `.integration-prep/recovery-pilot-complete/`: `state.json`, `acceptance.json`, and `pilot.jsonl`. Preserve those records in the integrated experiment artifacts before publishing a portable report.

## Feedback and integration scope

Current work includes outer-loop reuse diagnosis, conservative fallback feedback, generalized DMA mismatch guidance, and PSUM/compiler repair guidance, pushed in `7da33ee`. Local verification with `python -m unittest discover -s projects/03-chipboost/tests -p 'test_*.py'` passed all 36 tests; this is implementation coverage, not evidence of model improvement. The earlier four representative chip checks are in `feedback_validation.json`. Final seat/core-2 validation in [final_feedback_validation.json](research/final_feedback_validation.json) confirms the baseline `no_gain` and all three named failure instructions: DMA mismatch, NKI tile-list representation, and PSUM lifetime. Failure cases remain `wrong`; these instruction-selection checks are not optimized-kernel results.

The original comparison remains pinned to the older referee/agent snapshots and does not measure these later feedback fixes. Recovery v3 changes both feedback and the repair controller; its unsuccessful eight-attempt pilot does not isolate either change's causal effect. Keep all experiment cohorts separate.

## Final comparison handoff

After the original comparison finishes, collect its nine arm JSONL files together with `state.json`, `acceptance.json`, and `infrastructure.jsonl`. Run the strict summarizer without `--allow-partial`. Require 72 records, 24 per arm, three distinct completed runs per arm, matching pinned versions, and the passed acceptance gate before replacing this provisional status.

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
