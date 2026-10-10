# P1 acceptance retry and agent comparison

## Launch snapshot — October 10, 2026

The pinned A/A retry **passed** on core 3 with `no_gain`, 1.000049x speedup. The first `referee-r0`
comparison started after that gate. Runner PID: `751076`; worker PID: `751140` at launch.
This is launch evidence, not a claim that all 72 comparison evaluations have completed.

Live output on seat-100: `/tmp/p1-comparison-20261010-2`. The process is detached from the client session.
Inspect `state.json`, `runner.log`, and the per-arm logs there for current progress. Local
`comparison_launch.json` is a snapshot only; `comparison_acceptance.json` records the passed retry and
`comparison_infrastructure.jsonl` records the historical busy-core interruption separately.

The earlier `/tmp/p1-comparison-20261010-1` launch stopped at preflight before executing a kernel because
its candidate fingerprint check used raw rather than normalized newlines. Its failure record is retained;
the corrected launcher matches the original acceptance script's text normalization.

Prepared protocol; this document does not establish that acceptance or comparison runs have completed. Inspect the selected output directory for execution evidence.

`run_comparison.py` loads isolated repository snapshots (not active development checkouts):

| Component | Commit | Purpose |
| --- | --- | --- |
| P1 | `434e5f9` | Referee and original acceptance baseline |
| P2 | `919c6be` | Random search arm |
| P3 | `2ce9416` | Referee-feedback and model-alone agent arms |

## Launch protocol

On the Trainium seat, inspect core availability immediately before launch and select core 3 only if free. Do not overlap this run with self-tests. Run with root, the Neuron SDK, and the requested model already served by vLLM. Pass repository-root paths for each snapshot and a fresh output directory:

```sh
python run_comparison.py \
  --p1 /path/to/p1-434e5f9 \
  --p2 /path/to/p2-919c6be \
  --p3 /path/to/p3-2ce9416 \
  --out /path/to/fresh-comparison-output \
  --core 3 --budget 8 --repeat 3 \
  --base http://localhost:8000/v1 --model Qwen/Qwen3-8B
```

`--p1`, `--p2`, `--p3`, `--out`, and `--core` are required. `--budget` defaults to 8 evaluations per arm per repeat; `--repeat` defaults to 3. The endpoint and model above are the defaults. The runner fixes `CHIPBOOST_SEAT=100`, pins `CHIPBOOST_CORE`, and holds a per-core launcher lock. Runtime allocation determines whether the selected core is available; there is no fallback to another core. The lock prevents another copy of this launcher from using the same core, but does not reserve it against unrelated programs.

## Acceptance gate

One persistent `RefereeWorker` owns the selected core across acceptance and all comparison arms. The runner reads P1 `projects/02-kernel-agent/reference_level4.py` as text and writes `acceptance_candidate.py`, matching the original acceptance script's newline normalization. The baseline file is unchanged. It checks the referee-style source SHA-1 prefix `638cfdf0f2fa` and the P1 referee SHA-256 `5f366558ae933c88262bbde806afbcd4a568fdbfab4fc3395951f1b3799d71be`.

Comparison starts only after the acceptance record validates against the schema and reports `verdict=no_gain`, `sim_ok=true`, and `chip_ok=true`. An unavailable core, missing verdict, or failed assertion stops the launch. This retries the interrupted acceptance case; it is not a rerun of the full historical acceptance suite.

## Comparison design and records

The planned comparison is 3 arms × 3 repeats × 8 evaluations, or 72 comparison records plus the acceptance record. Arms are `referee`, `model_alone`, and `random_search`; their order rotates between repeats. All run sequentially through the same worker. Each arm must produce exactly its evaluation budget with schema-valid records before the next arm starts.

P3 normally grades the start kernel outside its budget. The launcher reuses the passing acceptance record
for that startup measurement, after verifying identical source bytes, so neither model arm gets an extra
referee evaluation. Every generated candidate is graded afresh. Routine worker recycling is disabled for
the planned batch by setting a larger `max_checks`; watchdog recovery may still replace a failed worker.
P2's extended checker imports are isolated from the pinned P1 checker used by the worker.

The P2 random-search arm samples caps over an expert template. It shares the timing baseline with the agent arms, but has a different candidate prior. Results therefore compare these complete search setups and do not isolate feedback as the only variable across all three arms.

| Output | Meaning |
| --- | --- |
| `state.json` | Phase, pinned core, hashes, acceptance state, completed arms, and terminal failure details |
| `acceptance_candidate.py` | Exact retry candidate bytes |
| `acceptance.json` | Acceptance record; inspect the gate state as well as this file |
| `infrastructure.jsonl` | Infrastructure events kept separate from kernel verdicts |
| `<arm>-r<repeat>.jsonl` | Comparison records, with zero-based repeat index |
| `<arm>-r<repeat>.log` | Captured arm stdout/stderr |

The prior interruption at `2026-10-10T19:01:09Z` is logged separately: logical core 2 was busy and `check_isolated` returned `None`; no kernel verdict was recorded. It is marked `counted_as_kernel_attempt=false`. New worker failures returning no record are also written to the infrastructure log. Check `state.json` for other exceptions and incomplete runs; the existence of output files alone does not establish success. Use a new output directory for another invocation because the runner refuses to reuse an existing `state.json`.

Watchdog events are also logged separately but retain the referee's `wrong` verdict and count as an
attempt; a candidate timeout alone does not establish infrastructure failure.
