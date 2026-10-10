# Preliminary live results

These are historical baseline and prompt trials. The subsequent matched tool-repair
experiment is documented separately in [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md).

The original JSONL traces, configurations, source snapshots, and summaries have now been downloaded
directly from seat-87 and inspected. The initial comparison contains one run per profile; a subsequent
batch repeats baseline three times. These small, fixed-seed samples do not establish population solve
rates or a causal effect of the prompt change.

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
All 16 final replies and all 16 first replies were byte-identical in the downloaded traces.
Repair prompts contained the previous answer and the checker's decay-rate diagnosis, but subsequent
answers retained the same error. Calculator results were correct. Additional rounds increased
full-run latency and total output tokens without solving the problem.

The concise trial is **not a demonstrated optimization** and remains opt-in. Baseline stays the
default. Longer output may contain useful reasoning; this single comparison cannot establish
that removing it caused the failure. The structured repair profile remains unmeasured.

## Repeated baseline: three additional runs

These runs used the baseline configuration with both prompt styles set to `baseline`, seed 0,
and the same budgets as the initial comparison. They ran sequentially using source revision
`f8287ec`; no solver or checker changes were made during measurement.

| Run | Solved | Rounds | Wall time | Model requests | Completion tokens | Truncated replies |
|---|---|---:|---:|---:|---:|---:|
| 1 | Yes | 2 | 205.325 s | 15 | 3962 | 0 |
| 2 | No | 4 | 777.336 s | 25 | 16167 | 8 |
| 3 | Yes | 2 | 257.795 s | 15 | 5534 | 1 |

The observed batch result is 2/3 solved, with median full-run wall time 257.795 seconds, including
the unsuccessful run. Baseline itself can stall at reward 0.6. Its third run also started with four
0.6 candidates in a 43.7-second round, so that behavior is not exclusive to the concise profile.

Trace inspection found five candidates whose first reply used Markdown-wrapped `**COMPUTE:**`
requests that the original strict calculator-request parser did not recognize. Nine of the 32 final
answers were unparseable. The nine truncated replies affected eight candidates; truncation and
unparseable output are distinct measurements and should not be added as disjoint failure counts.
The failed run spent 776.355 seconds in generation/tool exchanges and 0.424 seconds in checking.

These observations separate protocol handling, reply truncation, and mathematical errors. They do
not demonstrate a successful optimization. In one initial baseline failure, the model wrote the
correct general decay formula before using the calculator, then used incorrect decay rates in its
final answer. Correct intermediate reasoning and correct integrals did not guarantee a correct
assembled answer.

## Other level 1 cases: initial coverage

Levels 1.1 and 1.2 were then run sequentially, once each, with seed 0, the same baseline prompt
styles, and the same per-run budgets. Original artifacts were downloaded and inspected.

| Case | Solved | Rounds | Wall time | Model requests | Completion tokens | Truncated replies | Unparseable final answers |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.1 | Yes | 2 | 430.303 s | 8 | 8958 | 7 | 7/8 |
| 1.2 | Yes | 1 | 218.206 s | 5 | 4632 | 2 | 3/4 |

Case 1.1 initially scored zero for all four candidates. An inspected reply derived the correct
frequency and first coefficient, but was truncated before a complete final answer. Both cases
eventually solved, with one passing candidate each. These are coverage checks, not reliable
per-case success-rate estimates. Level 0 was not rerun against the live model in this session.

There is also a concrete latency opportunity to investigate: the passing candidate in case 1.1's
second round returned after 100.174 seconds, while the round waited 214.860 seconds for all
candidates before grading. Case 1.2's passing candidate was on the slowest path, so early grading
would not offer the same benefit there. Incremental grading and request cancellation have not
been implemented or benchmarked.

## Artifact locations

- Baseline: `runs/live-baseline-20261010T162848Z-4c2v19jf/` (initial experiment commit `888d1a9`).
- Concise trial: `runs/live-baseline-tools-concise-20261010T163842Z-t5e788wg/` (configuration introduced in `f8287ec`).
- Repeated baseline: `runs/live-baseline-20261010T171446Z-pihufasc/` (three completed runs, `f8287ec`).
- Other level 1 cases: `runs/coverage-level1-seed0-20261010T173742Z/` (one run each of 1.1 and 1.2, `f8287ec`).

## Next validation

Baseline now covers all level 1 subproblems at seed 0. Next isolate one protocol or repair change
and compare it across problem types and additional problem seeds with equal budgets. A first-round
success does not exercise repair feedback. Retain unsuccessful,
unparseable, truncated, and interrupted outcomes in the records.
