# Calibration after the fact

Source: /h/trainium-agent-labs/runs/seat-116/expA_L1/attempts_expA_L1.jsonl. 5 (run, level) pairs. For each, the run's best kernel went through `agent.verdict()`: the confidence from `agent.confidence()` (stated from the loop's evidence only), then the held-out set (`nkibench.evaluate`, new shapes x 4 value kinds) in the NKI 0.6.0 CPU simulator.

Loop rewards were **re-graded**: every logged attempt scored again with `agent.grade()`, each from a fresh path, and the best kernel chosen by the new score. Every attempt scored the same as logged.

| run | level | loop reward | confidence | why | held-out | claim | first held-out failure |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 2 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 3 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 4 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |
| 5 | 1 | 0.30 | 0.00 | it still fails shapes the loop tested | 0/20 | NOT SOLVED | C,H,W=(128, 32, 32) pool=2 normal: raised AssertionError: dma_copy requires src and dst to have the same number of eleme |

## Calibration

| confidence said | verdicts | mean confidence | passed held-out |
|---|---|---|---|
| 0.00–0.01 | 5 | 0.00 | 0/5 |

**Brier score 0.000** over 5 verdicts (0 is perfect; always saying 0.5 scores 0.25). **Confident (≥ 0.5) but failed held-out: 0.**

## Failure modes, logged vs re-graded

- logged: `copy_size_mismatch` 60, `wrong_signature` 60, `wrong_buffer` 20, `invented_name` 20
- re-graded: `copy_size_mismatch` 60, `wrong_signature` 60, `wrong_buffer` 20, `invented_name` 20
- per run, best loop reward logged -> re-graded: run 1 L1 0.30 -> 0.30, run 2 L1 0.30 -> 0.30, run 3 L1 0.30 -> 0.30, run 4 L1 0.30 -> 0.30, run 5 L1 0.30 -> 0.30
