# Preliminary live results

These observations come from participant-supplied terminal output and trace statistics.
The raw server artifacts have not yet been downloaded for independent verification.
Each profile has only one completed live run; the table does not establish population solve rates
or a causal effect of the prompt change.

Both runs used level 1.3, seed 0, four samples, at most four rounds, a 1200-token reply limit,
one calculator exchange per candidate, and baseline repair feedback. The experimental setting
changed only the calculator-request prompt. The checker and its tolerances were retained.

| Metric | Baseline | Concise calculator protocol |
|---|---:|---:|
| Solved | Yes | No |
| Rounds used | 1 | 4 (budget exhausted) |
| Best reward | 1.0 | 0.6 |
| Full-run wall time | 106.3 s | 173.7 s |
| Model requests | 8 | 32 |
| Total completion tokens | 2313 | 3632 |
| Calculator expressions evaluated | 21 | 80 |
| Truncated replies | 0 | 0 |

## Interpretation

The baseline spent 105.5 seconds in generation/calculator exchanges and 0.2 seconds in checking.
Its successful candidate made two API calls lasting 97.8 and 7.4 seconds. Those calls are on the
same candidate's sequential path; calls across candidates run concurrently and must not be summed
as batch wall time. The first successful reply contained 959 completion tokens.

The concise trial took approximately 43 seconds per round, but all 16 candidates scored 0.6.
The best expression printed in each round repeated the same incorrect decay rates despite checker
feedback. Additional rounds increased full-run latency and total output tokens without solving
the problem. The terminal summary alone cannot establish that every candidate was byte-identical.

The concise trial is **not a demonstrated optimization** and remains opt-in. Baseline stays the
default. Longer output may contain useful reasoning; this single comparison cannot establish
that removing it caused the failure. The structured repair profile remains unmeasured.

## Artifact locations

- Baseline: `runs/live-baseline-20261010T162848Z-4c2v19jf/` (initial experiment commit `888d1a9`).
- Concise trial: `runs/live-baseline-tools-concise-20261010T163842Z-t5e788wg/` (configuration introduced in `f8287ec`).

## Next validation

Run the structured repair profile with the original calculator instructions. Inspect whether it
repairs the decay-rate error after a failed attempt. A first-round success does not exercise repair
feedback. Repeat comparable profiles serially and retain unsuccessful or interrupted runs.
