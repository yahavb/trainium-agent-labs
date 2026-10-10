# Kernel transfer: the NKI feedback ablation

**Result: no feedback setting solved any NKI level, 0 of 36 level-runs.** Neither a better message nor the real API documentation moved Qwen3-8B past the organisers' walls. That fits the hypothesis in DESIGN 6.9: on NKI the model lacks knowledge, in particular the tiling idiom, and neither the error message nor the docstrings supply it.

| Level | K-raw: simulator error verbatim | K-located: organisers' `enrich()`, as shipped | K-docs: located plus real NKI docstrings |
|---|---|---|---|
| 1 average pooling | 0/3 · 0.30, 0.30, 0.30 | 0/3 · 0.30, 0.30, 0.30 | 0/3 · 0.30, 0.30, 0.30 |
| 2 transpose | 0/3 · 0.30, 0.62, 0.30 | 0/3 · 0.30, 0.30, 0.50 | 0/3 · 0.30, 0.30, 0.30 |
| 3 matmul, one tile | 0/3 · 0.30, 0.30, 0.30 | 0/3 · 0.30, 0.30, 0.30 | 0/3 · 0.30, 0.30, 0.30 |
| 4 matmul, tiled | 0/3 · 0.50, 0.30, 0.62 | 0/3 · 0.62, 0.30, 0.62 | 0/3 · 0.62, 0.30, 0.62 |

Cells show levels solved out of 3, then the best reward in each run. The reward is 0.1 for code that parses, plus 0.2 if it follows the rules, 0.2 if it runs, and 0.5 if it's correct, so 0.30 means the kernel parsed and followed the rules, then crashed in the simulator.

## Setup

- **Model and hardware:** Qwen3-8B on seat 93 (one Trainium2 chip), 2026-10-10.
- **Agent:** the organisers' `projects/02-kernel-agent/agent.py` at `8f1ca41`, imported unchanged by `kernel_agent.py`.
- **Settings:** `--all --rounds 8 --samples 1 --context 8192 --repeat 3`.
- **Only the feedback differs.** K-raw switches `enrich()` off. K-docs appends the signature and first docstring lines of every real NKI call in the failing kernel, up to 2,400 characters.
- **K-located is the morning baseline**, run with the same code and settings.

## What the numbers show

- **Levels 1 and 3 are fixed walls.** They scored 0.30 in all 9 runs across the three settings, as in the organisers' 5 runs.
- **Levels 2 and 4 vary more within a setting than between settings.** At 3 runs each, no setting effect is detectable.
- **Level 2 depends on sampling.** The organisers report it solved 4 of 5 times with 4 samples per round. Here, with 1 sample per round, it solved 0 of 9 times. One sample per round was forced because identical prompts sent in parallel return identical answers on these seats.
- **The same failures dominate every setting.** Most common is a runtime `AssertionError` from the NKI simulator (37 to 52 attempts per setting), followed by invented attributes (`AttributeError`, 12 to 15).

## Limits

- 3 runs per level.
- Model and sampling are as the seats behaved on 2026-10-10.
- K-docs costs about 600 extra prompt tokens per repair.
- `--samples 1` differs from the organisers' default of 4, for the reason above.

## Files

- `runs/K-<setting>.log`: the organisers' per-level summary.
- `runs/K-<setting>-attempts.jsonl`: one line per attempt, with the feedback text the model saw.
