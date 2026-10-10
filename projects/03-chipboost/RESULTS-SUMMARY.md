# CHIPBOOST Qwen results — October 10, 2026

The deliverable comparison uses Qwen/Qwen3-8B only, with expert-template random search as the non-model control. P1 acceptance and the original 72-evaluation comparison are complete. Neither original Qwen arm found a verified speedup; the template control did. The separate consolidated Qwen-only v2 run is complete: 7 `wrong`, 1 `faster`; its 1.517123964x winner passed two unchanged-source replays at 1.516961251x and 1.516817767x.

Two later P1-fix runs produced 2 `faster` and 14 `wrong` outcomes in 16 attempts. A four-attempt continuation produced no new valid kernel; the verified 1.517x seed remains the deliverable.

## Evidence by experiment

| Experiment | Current evidence | Interpretation |
| --- | --- | --- |
| P1 throughput acceptance | Nine targeted checks passed; pinned A/A retry on core 3 returned `no_gain`, 1.000049x | Referee execution/acceptance evidence, not an optimization discovery |
| Original Qwen comparison | Complete: 72 records, 24 per arm, three repeats of eight; strict report passed | Original feedback only; no Qwen `faster` outcomes; keep separate from later treatments |
| Qwen recovery pilot v3 | Complete: 8 graded attempts, all `wrong`, zero `faster` | Full-feedback plus repair-controller exploratory treatment; no valid improvement found |
| Qwen reasoning run | Canceled for the deadline with zero graded attempts | No optimization result; do not count cancellation as a kernel failure |
| Queued DMA-only v2 | Canceled as superseded; no comparison results | Its earlier message-selection replay is not a model outcome |
| Consolidated Qwen-only v2 | Complete: 8 attempts, 7 `wrong`, 1 `faster`; attempt 3 at 1.517123964x; two unchanged-source `faster` replays at 1.516961251x and 1.516817767x | One exploratory multi-change run; replays confirm its candidate, not a general model success rate |

The unsuccessful recovery pilot is preserved in [its experiment directory](experiments/qwen-recovery-pilot/), including `state.json`, `acceptance.json`, and all eight records in `pilot.jsonl`.

## Completed original Qwen comparison

The [strict report](experiments/qwen-v1-comparison/report.md) and [machine-readable report](experiments/qwen-v1-comparison/report.json) validate all nine runs. The passed acceptance gate precedes 72 evaluations on core 3, using one worker (`worker_starts=1`). The historical busy-core interruption is a separate infrastructure event, excluded from attempt and failure counts.

| Arm | Attempts | `wrong` | `no_gain` | `slower` | `faster` | Median best verified speedup across runs |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen + original referee feedback | 24 | 22 | 2 | 0 | 0 | 1.000000x |
| Qwen model alone | 24 | 2 | 22 | 0 | 0 | 1.000000x |
| Expert-template random-search control | 24 | 0 | 0 | 2 | 22 | 3.125919x; run-best range 2.525760–3.323522x |

No records have `rules` or `heldout_fail` verdicts. Runs with no verified improvement retain the 1x baseline in the best-speedup summary. Kernel rejection rates were 22/24 and 2/24 for the two Qwen arms, respectively; these are descriptive results from three small runs, not proof that feedback generally harms performance. This older referee supplied the feedback later corrected. The template control starts from a substantially different candidate prior, so its gains are not Qwen-generated improvements or an isolated feedback effect.

## Completed consolidated Qwen v2 and unchanged-source replay

The [v2 state](experiments/qwen-v2-feedback/state.json) confirms completion of one eight-attempt
Qwen/Qwen3-8B referee-guided run on core 2: **7 `wrong`, 1 `faster`**. Attempt 3 achieved
**1.517123964x**, passing simulator, chip correctness, and five undisclosed held-out shapes.
The model response and prompt are preserved in [pilot.jsonl](experiments/qwen-v2-feedback/pilot.jsonl).
The candidate stages all rhs K tiles in distinct SBUF slots before the m loop and allocates a fresh
PSUM accumulator for each output tile.

Two unchanged-source checks on core 3 both returned `faster`: **1.516961251x** and **1.516817767x**.
See [replay evidence](experiments/qwen-winner-replication/results.json) and the
[unchanged candidate](experiments/qwen-winner-replication/first_winner.py). Candidate SHA-256:
`18a59e54ce2d293c5b68651d50fd2dcaab2aab4e2876570e0893c71b991165d6`.
These two checks validate this candidate; they are not additional model attempts or independent
model runs. The run's pinned source hashes and replay referee hash remain in their respective artifacts.

This is one exploratory consolidated treatment, changing more than DMA feedback. It is separate
from the original three-repeat v1 comparison, the failed recovery pilot, and the canceled DMA-only
treatment. One successful run does not establish a general success rate or isolate the causal effect
of any individual feedback change. This cohort contributes 8 model attempts alongside the 72 original v1 records; later cohorts are reported separately below. V1 and v2 stay separate; replay and archived side experiments
are excluded from those attempt counts.

## Final P1-fix repeats and winner continuation

Two fresh Qwen runs using P1-fix source `d05cd7e` completed 16 evaluations: **14 `wrong`, 2 `faster`**.
Core-2 run r0 produced both verified candidates (1.516970430x and 1.517210463x);
core-3 run r1 had eight `wrong` results. Raw records remain separate in
[r0](experiments/qwen-v2-p1fix-r0/pilot.jsonl) and [r1](experiments/qwen-v2-p1fix-r1/pilot.jsonl).
These exploratory repeats used the original timing baseline and do not replace the v1 comparison.

The [four-attempt winner continuation](experiments/qwen-continuation/summary.json) used a later
referee snapshot and began from the frozen 1.517x Qwen candidate. All four new candidates were `wrong`;
none improved the seed. Its separate startup check returned `faster` at **1.517291745x**
against the original reference baseline. The existing verified winner is preserved. Startup/replay
checks are excluded from the generation budget; continuation has a different starting prior.

The dashboard now includes **100 graded model/control attempts**: 72 original v1 + 8 consolidated v2
+ 16 P1-fix repeats + 4 continuation. These cohorts have separate display groups; no archived non-Qwen
side trials, startup checks, or replication records enter the attempt totals.

## Feedback and integration scope

Current work includes outer-loop reuse diagnosis, conservative fallback feedback, generalized DMA mismatch guidance, and PSUM/compiler repair guidance, pushed in `7da33ee`. Local verification with `python -m unittest discover -s projects/03-chipboost/tests -p 'test_*.py'` passed all 36 tests; this is implementation coverage, not evidence of model improvement. The earlier four representative chip checks are in `feedback_validation.json`. Final seat/core-2 validation in [final_feedback_validation.json](research/final_feedback_validation.json) confirms the baseline `no_gain` and all three named failure instructions: DMA mismatch, NKI tile-list representation, and PSUM lifetime. Failure cases remain `wrong`; these instruction-selection checks are not optimized-kernel results.

The original comparison remains pinned to the older referee/agent snapshots and does not measure these later feedback fixes. Recovery v3 changes both feedback and the repair controller; its unsuccessful eight-attempt pilot does not isolate either change's causal effect. Keep all experiment cohorts separate.

## Final comparison handoff

The original comparison files are collected in `experiments/qwen-v1-comparison/`, including all nine logs, state, acceptance, infrastructure, and strict reports. Strict validation passed for 72 records, 24 per arm, three completed runs per arm, matching pinned versions, and the acceptance gate. Completed consolidated v2 results and winner replays are preserved in separate experiment directories; do not overwrite or extend the completed v1 records.

Build the dashboard with explicit comparison log paths and separately verified adversarial results. Infrastructure interruptions and canceled jobs are not kernel attempts. P2 searches over an expert-template prior while agent arms start from the reference; preserve that limitation in the final report. The earlier dashboard inspection was static only because browser preview was blocked by the URL security policy.

## Archive appendix — excluded from the Qwen deliverable comparison

These prior side experiments are retained for provenance only. They are excluded from the Qwen headline results, comparison aggregates, and dashboard inputs. Their different generation context and workflow do not support an equal-budget model comparison or a model ranking.

| Archived case | Recorded speedups |
| --- | --- |
| Astra-labeled A | Initial 1.5170524x; unchanged-source repeats 1.5172862x, 1.5173145x |
| Astra-labeled B | Initial 1.2033504x; unchanged-source repeat 1.2027148x |
| Astra-labeled C | 2.55416996x and 2.551280807x |
| Syntax-only repair diagnostic | 1.1469457x following an earlier compilation failure |

The A/B records are in [initial_results.json](experiments/astra-trial/initial_results.json) and [replication_results.json](experiments/astra-trial/replication_results.json); C records are in [candidate_c_results.json](experiments/astra-trial/candidate_c_results.json). GENERATION.md records generation provenance. The Astra label was supplied by delegation and is not independently attested by these artifacts. Initial A/B checks used referee SHA-256 `429309eb48c515ead0d6bd50b28c432f2c69c089a1992eeee9b9b87333714a5b`; A/B replications used original referee SHA-256 `5f366558ae933c88262bbde806afbcd4a568fdbfab4fc3395951f1b3799d71be`.
