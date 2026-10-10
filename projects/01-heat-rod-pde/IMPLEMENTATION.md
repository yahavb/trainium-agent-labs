# Heat-Rod PDE Agent — seat 85

Upstream: https://github.com/yahavb/trainium-agent-labs

Starting revision: `8f1ca41` (the seat checkout). Original files are retained unchanged.

## Run

Inside your own pod:

```bash
cd /workspace
./serve.sh
cd projects/01-heat-rod-pde
python level0_heatrod.py --selftest
python level1_heatrod.py --selftest
python -m unittest test_improvements -v
python improved_agent.py --level 1 --all --samples 2 --workers 2 --rounds 4 --repeat 3
```

`python benchmark.py` runs the real-model comparison sequentially. Each variant uses two samples per round, three rounds, 512 output tokens and one calculator exchange. All six subproblems use problem seed 0; the hardest parabola case additionally uses seeds 1 and 2. These seeds change the generated PDE problems; model sampling remains stochastic and is NOT fixed by a per-request seed. The current Neuron backend crashes when that parameter is supplied, so the improved client deliberately omits it.

## Changed behavior

- `improved_agent.py` retries transient connection/HTTP errors, keeps infrastructure failures separate from mathematical failures, retains the best candidate across rounds, and shows the model a short ledger of distinct failures. Truncation-aware feedback addresses derivations consuming the token budget. The experimental concise output contract (`--concise`) is disabled by default because it regressed on measured Level 0 cases. Calculator exchanges use assistant/user message roles and are recorded, including the integral the model chose. Final-version traces also record the exact request messages.
- `validation.py` checks expression syntax before parsing and independently rechecks full-score candidates at 384 held-out space/time points, including near-zero times, and a refined 1601-point initial-shape grid. It uses only the equation, boundaries and initial data. It does not read `exact` or `series_answer`.
- `test_improvements.py` checks exact solutions over multiple problem seeds, Fourier truncation thresholds, boundary/decay failures, invalid expressions, retry behavior, tool traces, service-error labeling and preservation of the best candidate.
- `benchmark.py` saves isolated attempt files, console logs, summaries and a cross-variant comparison. A 180-second case timeout bounds stalled requests and is reported as an inconclusive budget timeout, separately from service errors.

## Why the extra checker matters

The original checker accepts a constructed wrong answer: a correct solution plus `exp(-1000000*t)*sin(1600*pi*x/L)`. The extra sine vanishes on the original initial-shape grid, and its exponential disappears at the original random time points. It still violates the PDE near time zero. The new early-time checks reject it. This regression is tested, rather than just asserted.

This is strengthened numerical validation, not a proof for every possible function. Finite grids can still miss adversarial expressions. The syntax guard is not a universal computational sandbox; SymPy and the inherited calculator can be expensive on complicated inputs. Run only in the isolated lab environment.

## Reading the results

Each `results/<run-id>/attempts.jsonl` line contains the prompt, final answer, full model/calculator trace, infrastructure error if any, and grade. `summary.json` records validated solutions, rounds, requests, errors, timings and settings. Benchmark baselines use the upstream log format, which does not record intermediate calculator exchanges.

The first benchmark batch (`20261010T155031Z`) hit the Neuron per-request-seed compatibility bug after the first baseline solution. Keep it as an infrastructure incident; do not treat its failed cases as evidence about mathematical capability. Compare only the subsequent healthy batch, with service failures explicitly counted.

A second preliminary batch (`20261010T155450Z`) exposed derivations consuming the output budget. It was stopped before completing the matrix, to add the concise output contract and truncation-aware feedback. It is retained separately and is not the final comparison.

A higher solve rate on three problem seeds for the hardest case is exploratory evidence, not a statistically established improvement. The other cases have only one run per variant. Equal samples and rounds do not make wall-clock/request costs identical: the improved client may retry network requests. The comparison has no hardcoded analytic-answer fallback and never uses `--offline`.

## Where the work is saved

Remote files: `/workspace/projects/01-heat-rod-pde` in `seat-85`, branch `challenge1-improvements`.

Local files: this deliverable folder and the downloaded experiment archive beside it.

Target fork: https://github.com/ChenYujunjks/trainium-agent-labs, branch `challenge1-improvements`. The official upstream remains untouched. Publication depends on repository authorization and must be confirmed from the actual remote branch; a local commit alone is not an upload. Files in a pod disappear if that pod is replaced. The workshop asks for the checker, attempt log, and a one-page note; it does not specify a submission website. An organizer must provide that destination.
