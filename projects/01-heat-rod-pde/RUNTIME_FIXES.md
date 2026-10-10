# Runtime resilience fixes

Base: master `6ea9f98583c1c01db92705903c9cab167096b882`.
Scope: the five failures reproduced during the master review, including the legacy
benchmark entrypoint. These are runtime and reporting fixes, not evidence of a new
Qwen3-8B solve rate.

## Before and after

| Reproduction | Before | After |
|---|---|---|
| Final answer `u(x,t) = [1,2]` or `(1,2)` | Agent raises AttributeError before recording the answer | Answer, trace and evaluation error are recorded; subsequent rounds can recover within the existing budget |
| Model service unavailable | Comparison starts the next case; can report batch_complete=true and exit 0 | Stop after saving partial results, batch_complete=false, exit 1 |
| Valid JSONL row followed by a truncated row | Summary raises JSONDecodeError | Preserve the valid candidate, record the damaged line and mark artifacts incomplete |
| Calculator receives a tuple/list | Unhandled AttributeError | Return a scalar-expression error in the calculator trace; a later model reply can correct it |
| Two HTTP failures | model_requests=0, completion_tokens=0 | attempted_requests=2, failed_requests=2, completion_tokens=null |

The same review probe was rerun; see
[evidence/runtime-resilience/runtime-probes.json](evidence/runtime-resilience/runtime-probes.json).
A regression test additionally verifies a malformed first answer followed by a valid
answer can finish as solved in two rounds. Controlled responses in tests are not
real model performance measurements.

## Scoring and budget contract

- `agent.py`, `pdecheck.py`, `level0_heatrod.py` and `level1_heatrod.py`
  are byte-for-byte unchanged from the base commit's tested snapshot.
- A successful call to the official checker keeps its reward, parts and acceptance.
  As before, an infinite start_error is encoded as null for strict JSON only.
- If the checker raises, both reward and original_reward are null. The record
  contains evaluation_error with the exception type/message. No numerical score is invented.
- No extra physics validator is introduced. A score of 1.0 still means original
  checker acceptance, not an independent validation pass.
- A candidate evaluation error is distinct from a model service failure. The former
  can use remaining configured rounds; exhausted request failures stop further
  rounds/cases after already in-flight samples have been recorded.
- Samples, rounds, tool steps, tokens, timeouts and retry defaults are unchanged.
  No extra model call outside those budgets is added.
- After all rounds, a case with no scored candidate has status evaluation_error.
  If a prior candidate was scored, its best original score is preserved.
- Complete candidates from damaged logs remain visible, but damaged evidence is
  never labeled as a complete batch. Timeouts retain budget_timeout, even when a
  partial summary exists. A timeout with incomplete evidence makes the batch incomplete.

## Files

- `checker_runtime.py`: shared exception isolation around the original checker.
- `improved_agent.py`: log evaluation failures, bounded recovery, retain best candidates,
  stop on exhausted model request errors, save failure summaries and return nonzero.
  Its existing grade_candidate import remains available to replay callers.
- `tool_calc.py`: reject unsupported non-scalar objects as tool responses without
  changing valid scalar expressions or integral semantics.
- `attempt_records.py`: recover valid JSONL records with file/line diagnostics and
  count model requests, responses, failures and known/unknown token usage.
- `compare_agents.py`: safely regrade both raw and executed answers, preserve partial
  evidence, stop failed batches, and atomically replace its comparison summary.
- `benchmark.py`: apply the same aggregation and failure handling to the legacy
  8-case comparison; add an optional output-root path for isolated runs/tests.
- `test_runtime_resilience.py`: 22 regression tests covering recovery, interrupted
  runs, corrupt logs, batch completion, calculator errors, score parity and accounting.
- `README.md`: document the runtime/reporting contract and link this report.

## Reporting fields

`model_requests` now aliases `attempted_requests`, including model, http_error and
transport_error events. `successful_responses` and `failed_requests` are separate.
Legacy model traces without a type remain supported.

`known_completion_tokens` sums reported usage. `unknown_usage_requests` counts
requests without usable usage data. If any usage is unknown, `completion_tokens`
is null, not a partial sum presented as the total. Counts describe recorded traces;
they cannot recover requests whose log records were lost.

`log_errors` records missing, malformed or truncated evidence. `artifacts_complete`
does not imply every answer solved or every checker evaluation succeeded.
`evaluation_failures`/`evaluation_errors` record candidate regrading exceptions
separately from infrastructure failures.

## Validation

Run from `projects/01-heat-rod-pde/`:

```bash
python -m unittest discover -v
python tests/test_experiments.py
python level0_heatrod.py --selftest
python level1_heatrod.py --selftest
PYTHONPATH=. python evidence/runtime-resilience/review_probes.py
```

Results on seat-85, in isolated directory `/tmp/heatrod-runtime-resilience`:

- 35 core/regression tests passed (13 existing + 22 new).
- 11 experiment-workflow tests passed.
- Both official level selftests passed.
- All five original review reproductions now have the expected failure handling.
- Python 3.13.7, SymPy 1.14.0, NumPy 2.4.6, httpx 0.28.1.
- No real inference was run. The previously paused experiment was not resumed.

Logs, the reproduction script, environment versions and tested source SHA-256 hashes
are saved under [evidence/runtime-resilience/](evidence/runtime-resilience/).
Historical experiment artifacts were not edited.

## Remaining limits

These changes do not establish higher mathematical success rates. Once real experiments
are resumed, a matched fixed-set comparison is needed. The unchanged original agent can
still terminate on malformed final expressions; the runners now preserve and flag that
process failure instead of presenting a complete comparison. Exception isolation also
does not interrupt a symbolic calculation that hangs; external case timeouts remain the
existing protection.
