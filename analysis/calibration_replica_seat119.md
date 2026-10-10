# Calibration after the fact

Source: /h/trainium-agent-labs/runs/seat-119/replica/attempts.jsonl. 20 (run, level) pairs. For each, the run's best kernel went through `agent.verdict()`: the confidence from `agent.confidence()` (stated from the loop's evidence only), then the held-out set (`nkibench.evaluate`, new shapes x 4 value kinds) in the NKI 0.6.0 CPU simulator.

Loop rewards were **re-graded**: every logged attempt scored again with `agent.grade()`, each from a fresh path, and the best kernel chosen by the new score. Every attempt scored the same as logged.

| run | level | loop reward | confidence | why | held-out | claim | first held-out failure |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 1 | 2 | 1.00 | 0.90 | passed every loop shape and nothing in the code is shape-specific | 16/16 | VERIFIED |  |
| 1 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and |
| 1 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 2 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 2 | 2 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | shape=(128, 128) as 16x8 normal: raised AssertionError: dma_copy requires src and dst to have the same number of element |
| 2 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and |
| 2 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 3 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 3 | 2 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | shape=(128, 128) as 16x8 normal: NON-FINITE OUTPUT: 16256 NaN and 0 Inf, first at (0, 8). Usually an uninitialised PSUM  |
| 3 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got s |
| 3 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 4 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 4 | 2 | 1.00 | 0.90 | passed every loop shape and nothing in the code is shape-specific | 16/16 | VERIFIED |  |
| 4 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and |
| 4 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 5 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 5 | 2 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | shape=(128, 128) as 16x8 normal: raised AssertionError: dma_copy requires src and dst to have the same number of element |
| 5 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and |
| 5 | 4 | 0.50 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |

## Calibration

| confidence said | verdicts | mean confidence | passed held-out |
|---|---|---|---|
| 0.00–0.01 | 18 | 0.00 | 0/18 |
| 0.80–1.00 | 2 | 0.90 | 2/2 |

**Brier score 0.001** over 20 verdicts (0 is perfect; always saying 0.5 scores 0.25). **Confident (≥ 0.5) but failed held-out: 0.**

Unsolved runs are confidence 0 and fail held-out by construction, so they pull the Brier score toward 0. Over the 2 solved run(s) alone, where the confidence is a real prediction, the Brier score is **0.010**.

## Solved runs: did they hold up?

- run 1, level 2 (best kernel from round 0): confidence 0.90; held-out 16/16, **passes every case**.
- run 4, level 2 (best kernel from round 0): confidence 0.90; held-out 16/16, **passes every case**.

## Failure modes, logged vs re-graded

- logged: `copy_size_mismatch` 100, `invented_name` 81, `partition_over_128` 55, `reshape` 48, `tile_1d` 33, `out_of_bounds` 32, `wrong_signature` 21, `wrong_buffer` 21, `numeric_mismatch` 13, `broadcast` 9, `other` 4
- re-graded: `copy_size_mismatch` 100, `invented_name` 81, `partition_over_128` 55, `reshape` 48, `tile_1d` 33, `out_of_bounds` 32, `wrong_signature` 21, `wrong_buffer` 21, `numeric_mismatch` 13, `broadcast` 9, `other` 4
- per run, best loop reward logged -> re-graded: run 1 L1 0.30 -> 0.30, run 1 L2 1.00 -> 1.00, run 1 L3 0.30 -> 0.30, run 1 L4 0.62 -> 0.62, run 2 L1 0.30 -> 0.30, run 2 L2 0.30 -> 0.30, run 2 L3 0.30 -> 0.30, run 2 L4 0.62 -> 0.62, run 3 L1 0.30 -> 0.30, run 3 L2 0.30 -> 0.30, run 3 L3 0.30 -> 0.30, run 3 L4 0.62 -> 0.62, run 4 L1 0.30 -> 0.30, run 4 L2 1.00 -> 1.00, run 4 L3 0.30 -> 0.30, run 4 L4 0.62 -> 0.62, run 5 L1 0.30 -> 0.30, run 5 L2 0.30 -> 0.30, run 5 L3 0.30 -> 0.30, run 5 L4 0.50 -> 0.50

## Notes

- Input: runs/seat-119/replica/attempts.jsonl -- a teammate re-ran the baseline on seat-119 with upstream 8f1ca41 (--all --repeat 5, same settings as the seat-116 baseline). That code predates c39c0ce, so every attempt was re-graded here from a fresh path with the current checker (allocation audit on), NEURON_PLATFORM_TARGET_OVERRIDE=trn2, NKI 0.6.0 in Docker.
- Confidence weights fixed in 7868c08, before this run; confidence() unchanged since.
- The 2 level-2 solves are 2 distinct kernels; one of them (sha1 97daf585c6) is also a seat-116 baseline solve. All 420 attempts re-grade exactly as logged, so the replica's numbers (L1 0/5, L2 2/5 [1.00 0.30 0.30 1.00 0.30], L3 0/5, L4 0/5 [0.62 x4, 0.50]) stand.
