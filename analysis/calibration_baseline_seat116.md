# Calibration after the fact

Source: /h/trainium-agent-labs/runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl. 20 (run, level) pairs. For each, the run's best kernel went through `agent.verdict()`: the confidence from `agent.confidence()` (stated from the loop's evidence only), then the held-out set (`nkibench.evaluate`, new shapes x 4 value kinds) in the NKI 0.6.0 CPU simulator.

| run | level | loop reward | confidence | why | held-out | claim | first held-out failure |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 1 | 2 | 1.00 | 0.90 | passed every loop shape and nothing in the code is shape-specific | 16/16 | VERIFIED |  |
| 1 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and |
| 1 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 2 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 2 | 2 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | shape=(128, 128) as 16x8 normal: raised AssertionError: dma_copy requires src and dst to have the same number of element |
| 2 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got s |
| 2 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 3 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 3 | 2 | 1.00 | 0.90 | passed every loop shape and nothing in the code is shape-specific | 16/16 | VERIFIED |  |
| 3 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and |
| 3 | 4 | 0.50 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 4 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 4 | 2 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | shape=(128, 128) as 16x8 normal: raised AssertionError: dma_copy requires src and dst to have the same number of element |
| 4 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and |
| 4 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |
| 5 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 5 | 2 | 1.00 | 0.90 | passed every loop shape and nothing in the code is shape-specific | 16/16 | VERIFIED |  |
| 5 | 3 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/4 | NOT SOLVED | K=128 M=64 N=512 normal: raised AssertionError: dma_copy requires src and dst to have the same number of elements, got s |
| 5 | 4 | 0.62 | 0.00 | it still fails shapes the loop tested | 0/16 | NOT SOLVED | K=384 M=256 N=512 normal: raised AssertionError: dma_copy dst partition dimension 384 exceeds maximum 128 |

## Calibration

| confidence said | verdicts | mean confidence | passed held-out |
|---|---|---|---|
| 0.00–0.01 | 17 | 0.00 | 0/17 |
| 0.80–1.00 | 3 | 0.90 | 3/3 |

**Brier score 0.001** over 20 verdicts (0 is perfect; always saying 0.5 scores 0.25). **Confident (≥ 0.5) but failed held-out: 0.**

Unsolved runs are confidence 0 and fail held-out by construction, so they pull the Brier score toward 0. Over the 3 solved run(s) alone, where the confidence is a real prediction, the Brier score is **0.010**.

## Solved runs: did they hold up?

- run 1, level 2 (best kernel from round 0): confidence 0.90; held-out 16/16, **passes every case**.
- run 3, level 2 (best kernel from round 0): confidence 0.90; held-out 16/16, **passes every case**.
- run 5, level 2 (best kernel from round 0): confidence 0.90; held-out 16/16, **passes every case**.

## Notes

- The confidence weights (0.9; x0.7 single loop shape; x0.5 hard-coded test size; x0.6 hard-coded output dtype) were fixed in commit 7868c08 (13:40), before this retroactive run was made; confidence() is unchanged since then (diffed). Nothing was tuned to these results.
- The 3 level-2 solves are 2 distinct kernels (runs 1 and 5 produced the same code). scripts/reaudit.py (4350038) separately confirmed they pass the loop's shapes under the allocation audit.
- Input: runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl (baseline, 2026-10-10 10:50-12:43, --all --rounds 8 --samples 4 --context 8192 --repeat 5; old log format without a run field, split by order). Held-out check in the NKI 0.6.0 CPU simulator in Docker; nothing ran on the device.
