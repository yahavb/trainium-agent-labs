# Team 18 — New optional checker test

**10 October 2026 · seat: 85-88 · Qwen3-8B**

## Changes and actual test path

The commit changes the **optional enhanced checker**, not the public scoring path.
`pdecheck.py`, `checker_runtime.py`, `algorithm_agent.py` and `algorithm_tools.py`
are byte-identical to selected Attempt 2 (`cd16f7b`). Changed optional modules are
`checker_math.py`, `checker_syntax.py`, `checker_worker.py` and
`checker_enhanced_runtime.py`. Changes include bounded precision audits, exact
boundary substitution, scaled initial-profile norms and uncertainty estimates,
projection/decay diagnostics, tighter parser bounds, and a flushed core checkpoint.
Original weights and tolerances are not relaxed.

At the owner's request, the initial original-only run was interrupted and its
partial records retained. The new run invokes existing **`validation.grade`** for
candidate ranking, model feedback and stopping, through an external test adapter.
No repository source was edited. This opt-in API requires its baseline numerical
protocol to score 1.0 and enhanced verification to accept before reporting solved.
That baseline protocol uses a different parser from public `pdecheck.check`;
the adapter therefore also records the actual public original grade independently.
All 16 recorded candidate rewards agree between the two paths in this run.

Configuration: Level 1 cases 1–3, problem seed 0, four candidates/four workers,
at most four rounds, 1200-token request ceiling, one calculator exchange,
900-second request/per-problem deadlines; cache/final/feedback/adaptive enabled.
Adaptive requests may use lower token caps. Level 0 model warm-up was skipped.

## Live-model results

| Problem | New checker best | Public original best | Rounds | Requests | Seconds |
|---|---:|---:|---:|---:|---:|
| level1.1 | 1.0 | 1.0 | 1 | 8 | 15.19 |
| level1.2 | 1.0 | 1.0 | 1 | 7 | 59.00 |
| level1.3 | 1.0 | 1.0 | 2 | 14 | 112.82 |
| Total | **3/3 solved** | **3/3 solved** | 4 | 29 | **187.00** |

Candidate scores, indexed from round 0:

- 1.1: `[1.0, 1.0, 1.0, 1.0]`.
- 1.2: `[1.0, 0.0, 0.0, 0.0]`.
- 1.3: `[0.6, 0.6, 0.6, 0.6]` then `[1.0, 0.6, 0.6, 1.0]`.

Seven of 16 candidates earned full marks; no transport/evaluation errors occurred
in the live candidate records. Total output: 3755 tokens. Durations sum complete
per-problem elapsed times, including checks and unsuccessful candidates. The
194.50-second wrapper duration also includes the preceding optional offline tests
and process startup, so it is not the same timing definition.

The run was 41.24 seconds shorter than selected Attempt 2 (228.24s), with 29 rather
than 31 model requests. It had seven rather than nine full-score candidates.
This is one run under a different grading/feedback path and a test adapter that
also evaluates public grades; it does not establish a causal speed benefit,
improved reliability, or lower checker cost. The enhanced checker did not change
any recorded candidate's score relative to the public checker in this run.

## Offline verification and confirmed defect

Both official selftests and 62 existing regression tests passed. New checks
initially passed 21/23. One failure was a test-fixture issue: the enormous public
profile simplified into a numeric literal outside the parser's allowed budget.
Expressing the same profile as a bounded `10**200` syntax tree made the scaled
initial-condition test pass. The original failed record is retained.

The remaining failure is a **confirmed implementation defect**, reproduced both
with a mocked timeout and a real controlled subprocess timeout, in `verify` and
`grade`. If the worker times out before emitting a completed core record:

1. `validation = worker_failure` aliases the same dictionary (line 126).
2. `validation['worker_issue'] = worker_failure` inserts that dictionary into itself
   (line 128).
3. `clean_json` recursively walks the cycle and raises `RecursionError`, instead
   of returning an inconclusive result.

A controlled timeout after a completed core preserved its pass and worker issue;
a failed completed core also remained failed. These are offline controlled tests,
not observed service timeouts or model performance. Focused follow-up passed 2/3;
the pre-core failure was additionally confirmed on the exact live-test API `grade`.

Recommendation for the implementation owner: keep the outer validation record
separate from the worker-failure dictionary, then verify pre-core timeouts,
nonzero worker exits, and completed-core preservation. No fix was applied here.
The live 3/3 result does not clear this runtime defect. Do not merge based solely
on these three successes; engineering review remains required.

## Artifacts and scope

- `test-settings.json`: full settings and test-adapter hashes.
- `source/`: frozen source snapshot of the requested commit.
- `level1.*/.../attempts.jsonl`: all candidate answers, scores and feedback.
- `level1.*/checker-calls.jsonl`: every grader invocation, including in-attempt checks.
- `optional-checker-checks.*`, `optional-checker-followup.*`,
  `grade-precore-reproduction.json`: offline results and reproduction traces.
- `VERIFIED_TEST_SUMMARY.json`: verified totals and per-case results.

The stopped original-only batch remains in adjacent directory
`team18-31a150d-20261010T214245Z-d8f50c`, including selftest/regression logs and
interruption metadata. It completed 1.1 and 1.2; 1.3 was interrupted, not failed.
This test is additional evidence, not a replacement for formal Attempt 2. No new
formal attempt number, source update, GitHub push or merge was made.
