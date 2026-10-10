# Verified symbolic decay repair

The integrated agent now has an opt-in `--decay-repair` tool. In a matched real-model
comparison on three parabola problem seeds, it solved 2/3 cases versus 0/3 without the
tool. Total measured time fell from 210.595 to 120.110 seconds (43.0%), and completion
tokens fell from 5012 to 2812 (43.9%). These are observed results on this fixed small
test set, not a population solve-rate estimate.

## What the tool changes

For each submitted separable term `A*exp(-lambda*t)*wave(x)`, SymPy checks that
`wave''(x)/wave(x)` is a negative constant and sets the exponential rate from
`u_t = k*u_xx`. Thus `lambda = -k*wave''/wave`; for a sine or cosine with frequency
`mu`, this is `k*mu**2`. It uses the model's existing waves and amplitudes plus the
problem's conductivity. It does not read known exact solutions, the problem's basis
or frequency helpers, or `series_answer`.

The agent invokes the tool only if the original candidate passes both boundaries
and the initial-shape test but fails the equation. The tool preserves the initial
expression exactly and rejects unsupported forms. A proposal replaces the executed
answer only after the original checker and held-out validator both accept it. No
weights or tolerances were changed. Missing modes, wrong coefficients, malformed
answers, and token truncation remain the model/feedback loop's responsibility.

This extends the arithmetic performed by the agent's tools. It demonstrates an
improvement to the tool-assisted system, not stronger unaided model reasoning.

## Matched live experiment

Qwen3-8B on seat-87; problem 1.3 with seeds 0, 1, 2; two samples per round, at most
three rounds, 1200 completion tokens per request, one calculator exchange per attempt,
two client workers, no transport retries, thinking disabled, a 600-second case limit.
Both variants use the same improved agent, prompts, sampling parameters and verifier.
The sole treatment is `--decay-repair`. All six runs completed normally, without
service errors or case timeouts. Variant order alternates between problem cases.

| Seed | Without repair: solved / rounds / seconds | With repair: solved / rounds / seconds |
|---|---|---|
| 0 | No / 3 / 58.698 | Yes / 1 / 20.216 |
| 1 | No / 3 / 86.792 | Yes / 1 / 34.649 |
| 2 | No / 3 / 65.104 | No / 3 / 65.245 |

| Aggregate | Without repair | With repair |
|---|---:|---:|
| Validated solved cases | 0/3 | 2/3 |
| Total rounds | 9 | 5 |
| Model requests | 36 | 20 |
| Completion tokens | 5012 | 2812 |
| Total measured seconds, including unsuccessful runs | 210.595 | 120.110 |

For each seed, the two variants' first-round raw answers and actual model request
messages were identical. For seeds 0 and 1, the raw model answers failed the equation;
the tool changed their decay factors and the executed answers passed both checks.
This recorded correspondence directly locates the improvement in the tool, rather
than attributing it to a lucky correct model sample.

Seed 2 remains unsolved: its amplitudes are about half the required values, while
the equation already passes. The decay tool correctly does not change coefficients.
The failure is retained in all aggregates. Repeating similar model outputs does not
provide independent statistical trials.

Evidence: [`live-three-seeds/comparison.json`](evidence/decay-repair/live-three-seeds/comparison.json),
all per-case attempt traces, commands, console logs and the executed Python source snapshot
in the same directory. Settings include source SHA-256 hashes. `wall_seconds` includes
the client process and common post-hoc validation; `process_wall_seconds` is also retained.

Reproduce the exact case list by creating `three-seeds.json` containing
`[[1,3,0],[1,3,1],[1,3,2]]`, then running inside a seat with `HEATROD_BASE_URL` configured:

```bash
python compare_agents.py --cases three-seeds.json --variants improved decay \
  --samples 2 --rounds 3 --max-tokens 1200 --tool-steps 1 --case-timeout 600
```

## Recorded-candidate replay

A separate replay of eight previously completed runs examined 80 recorded candidates
(45 unique problem/seed/answer combinations). All 39 eligible candidate occurrences
were repaired and independently accepted. The fixed recorded-run set had a passing
candidate in 5/8 runs before repair and 8/8 afterward. One interrupted batch was
excluded explicitly because it was incomplete.

This is a counterfactual replay, not eight new real-model runs and not a latency
measurement. Later logged rounds would sometimes not have happened if an earlier
repair had stopped the run. Duplicate candidate occurrences are not independent trials.
Full input expressions, grades, proposed changes and input-file hashes are retained in
[`recorded-replay.json`](evidence/decay-repair/recorded-replay.json).

## Negative trials and limits

The earlier concise calculator prompt failed its initial live trial. A later generic
spectral-method checklist also failed its pilot: 0/1 solved, four rounds, 527.7 seconds,
26 model requests, 10345 completion tokens and one truncated reply. Both options remain
experimental and are not the basis of the improvement claim above.

The new tool's mathematical transformation is tested across multiple frequencies,
lengths and conductivities, with regression checks for unsupported expressions,
unchanged initial shapes, rejection of incorrect coefficients, and distinct model/tool
outcomes. Seventeen physics/agent/comparison/repair tests and eleven workflow tests passed,
as did both original checker selftests. An additional six-problem coverage batch was
stopped at the user's request to avoid further testing; its partial results are not
included in the improvement claim. Finite
held-out numerical checks strengthen validation but do not prove every possible answer.
The remaining coefficient-normalization failure needs a separate, isolated experiment.
