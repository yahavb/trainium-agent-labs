# Optional checker timeout repair — 10 October 2026

A worker failure before a completed core record aliased `validation` and
`worker_failure`. Attaching the latter as `worker_issue` made the result recursive,
so JSON normalization raised RecursionError. Copy the outer validation dictionary
before attaching failure evidence. This changes failure handling only: unfinished
checks remain inconclusive, and completed core outcomes survive optional-analysis
failures. Public scoring, tolerances, prompts and agent code are unchanged.

Validation: eight focused runtime regressions plus 62 existing regression tests
passed; both official selftests passed. Coverage includes verify/grade pre-core
timeouts, worker exits, incomplete checkpoints, valid/invalid/inconclusive completed
cores, original-score preservation, successful completion and real subprocess failures.
The new timeout regression failed on 31a150d and passed with this fix.

Earlier real-model evidence on 31a150d: Level 1 3/3, 187.00 seconds, 29 requests;
7/16 candidates earned full marks. Actual public checker rewards matched all 16
candidate rewards. This used an external test adapter opting into validation.grade;
the normal agent entrypoint still uses the public checker. This observation predates
the repair and is not a post-fix model benchmark or a controlled speedup claim.
Weights stayed unchanged, but enhanced uncertainty/acceptance checks and feedback
make comparisons with original-only runs observational. Missing historical traces
remain a submission limitation. No Chinese experiment guide is included.
