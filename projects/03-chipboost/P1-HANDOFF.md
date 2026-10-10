# P1 validation and handoff — October 10, 2026

Implementation under test: `referee-timing` at `434e5f9`; latest referee implementation `76b2227`.
The final commit only adds worktree references. Validation runs use an isolated copy under
`/tmp/p1-signoff-434e5f9` on seat-100, preserving the existing seat checkout and running agent loop.

## Evidence

- CPU checker: `python projects/02-kernel-agent/nkibench.py --selftest` passed locally.
- All P1 Python sources parsed successfully; `git diff --check` passed.
- Trainium timer: `python timing.py --selftest --shapes qwen` passed on core 2. q_proj: 268.3 us,
  IQR 0.1%; gate/up: 691.7 us, IQR 0.0%; 3x work took 2.58x time; A/A 1.000, threshold 1.010.
- Hardware acceptance: **all 9 checks passed on core 3**, plus deterministic threaded inputs.
  See `p1_acceptance.json` for records, wall times and the exact referee SHA-256.
  `validate_p1.py` exercises deterministic threaded inputs, isolated/worker A/A, worker reuse/recycling,
  rules and wrong-output rejection, watchdog recovery, a faster kernel with held-out checks, and a slower kernel.
- Isolated A/A: 32.75 s, 0.99993x. Cold worker A/A: 27.75 s; warm worker: 21.90 s, 0.99995x.
  Faster case with held-out: 2.94890x, 53.68 s. Slower case: 0.33899x, 24.86 s.
  Watchdog recovery: valid `no_gain`, 1.00007x. These compare the reference to itself or an intentionally
  narrow reference, rather than claiming a newly discovered optimized kernel.

Repeat on a Linux Trainium seat with the Neuron SDK, root, `setpriv`, vLLM on cores 0-1 and a free core:

```bash
cd projects/03-chipboost
python timing.py --selftest --shapes qwen
CHIPBOOST_SEAT=100 python validate_p1.py --core 3 --out /tmp/p1-acceptance.json
```

## Integration

- P2 `origin/kernels-search` at `919c6be` already contains identical `speedcheck.py` and `timing.py`.
  Its search uses `RefereeWorker`; its schema adds the `copy` op for bandwidth probes. Preserve that extension.
- P3 `origin/redteam-agent` at `2ce9416` already contains identical referee, timer and schema.
  Its loop uses supported `check_isolated`, writes candidates outside the protected tree, retries `None`,
  and sends only `instruction_given` to the model. Switching P3 to a persistent worker is an optional optimization.
- P4 consumes the existing schema, including `no_gain`; no P1 schema change is required.
- Full team integration into master and end-to-end three-arm experiment results are separate from P1 acceptance.
  The pre-existing `merge-test` worktree has uncommitted P2 path/Windows fixes; they were not overwritten.

## Scope and limitations

`results_p1.json` preserves the earlier 33-kernel adversarial regression, not a rerun of every attack on
the throughput implementation. The new acceptance suite is targeted at the changed execution paths.
The timer figures above and acceptance wall times are individual runs, not a sustained throughput claim.

The existing device-access limitation remains: the sandbox relies on vLLM holding core 0 and environment
configuration; device-node permissions do not enforce isolation. `check_isolated` still has the documented
child-cleanup limitation on whole-check timeout; use the worker for its descendant cleanup and restart path.
Model/input caching across checks and a shorter child timeout are not implemented. Current limits are
600 s child wall time, 1200 s child CPU time, and 1800 s per whole check.
