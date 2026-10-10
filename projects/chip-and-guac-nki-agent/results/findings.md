# Chip and Guac (#26): One-Page Findings

## Problem and Approach

Qwen3-8B must generate a valid NKI tiled matmul across the four official Level 4
shapes. The original agent repeatedly passed only the smallest shape. We changed
feedback ordering with a shape curriculum, then added SDK-proven stationary and
moving free-dimension repair guidance. The generator, scorer, references,
shapes, tolerances and stopping algorithm remain the organizers' originals.

## Achievement

Historical best scores are **Level 2 (Transpose): 1.000**, from earlier unchanged
original-agent baselines (3/5 runs solved), and **Level 4 (Tiled Matrix
Multiplication): 1.000**, from enhanced Arm C. Arm C was evaluated on Level 4
only; we do not attribute the earlier Level 2 result to Arm C.

The enhanced agent generated a Level 4 kernel that achieved **official reward 1.000**,
passing all four shapes. It was independently verified in a fresh process using
the unchanged original agent.grade; only temporary candidate-file I/O was isolated.

| Fresh matched Level 4 arm | Reward | Shapes | Calls | Tokens | Wall |
|---|---:|---:|---:|---:|---:|
| Curriculum B | 0.625 | 1/4 | 32 | 35,413 | 646.71 s |
| Enhanced C | **1.000** | **4/4** | **20** | **26,238** | **484.84 s** |

B reached its eight-round cap; C solved in five rounds. Both kept samples=4,
context=8192, max_tokens=2500 and give-up-after=4. The model was Qwen3-8B on
Trainium2, with temperature=.6, top_p=.95 and thinking disabled.

## What Changed After Guidance

C first reached 0.750 without the new targeted guidance. A selected stationary
free-dimension-256 error then received a repair plan. The next response added
M/N output tiles, tile-sized PSUM, explicit K accumulation and offset writeback,
and scored 1.000. Moving guidance did not activate in this live run. The actual
source diff is in guided_repair.diff.

## Verification and Scope

Five original baseline runs were recorded; 404 baseline candidates regraded with
exact reward, parts and feedback parity. Arm C's offline replay matched rewards
and scoring components on 448/448 saved records; only 15 selected stationary
errors changed relative to B, with all 433 other messages unchanged. The moving
trigger was separately tested using the installed SDK. This portable submission
passed 17/17 tests in the Neuron environment; the benchmark self-test and direct
original-checker verification passed. No new inference run was needed to package it.

The matched results are one pair, not repeated-run averages or a statistical
causal estimate. Device compilation and latency were not measured. Full-K SBUF
allocations remain suspicious despite CPU correctness. The roofline feedback is
an estimate based on counted traffic, not measured device utilization.
