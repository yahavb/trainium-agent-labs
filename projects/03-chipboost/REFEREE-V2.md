# DMA instruction experiment

Execution starts only after `/tmp/p1-comparison-20261010-2/state.json` reports all nine original runs complete.
The optional `--wait-for-predecessor` flag queues a process now, with a durable waiting state and PID,
polls every 30 seconds, and aborts if the original fails. It acquires the core lock only after completion,
waits for the original worker to release it, then repeats the full preflight.
The new treatment is the known DMA size-mismatch mapping in `instruction_given`, applied to a fresh
snapshot of P1 `434e5f9`. **Do not use the current merged referee:** its outer-loop reuse diagnosis
would change the initial prompt and confound this experiment. Leave the completed original outputs
and its source snapshots untouched.

Materialize only `_DMA_4X_ERROR`, `_DMA_4X_INSTR`, and the revised `_child_failure` function from the
DMA mapping change into the original P1 snapshot. Review the resulting source diff to establish that
this is the entire treatment, record its SHA-256, and retain that diff with the experiment. P3 stays
at `/tmp/p1-comparison-p3-2ce9416`; it already passes `instruction_given` to the next repair prompt,
so no P3 prompt change is necessary. Keep the entire P3 snapshot, including its Project 02 imports.

The dedicated launcher verifies predecessor metadata and all 72 original records, uses the same
per-core lock as the original launcher, and refuses a nonempty output directory. Inspect hardware
availability immediately before launch: the lock does not exclude unrelated processes. Use core 3
only after the old worker releases it; do not overlap self-tests there. Run one persistent worker.

```sh
python run_referee_v2.py \
  --p1 /tmp/p1-dma-v2 \
  --p3 /tmp/p1-comparison-p3-2ce9416 \
  --predecessor /tmp/p1-comparison-20261010-2 \
  --expected-referee-sha256 be620d5ef40f53780f0cde4ac3f1b05fddb4b05f3d5ee79c3e97ddf705116ebb \
  --out /tmp/p1-referee-dma-v2-20261010-1 --core 3 --wait-for-predecessor
```

Queued on seat-100 as PID 810896; verified status `waiting_for_predecessor` on Oct 10, 2026.
Source manifest, reviewed delta, and real-candidate replay are in `experiments/dma-feedback-v2/`.

The launch performs a new chip A/A acceptance check with the original normalized candidate and raw
baseline fingerprints, then runs 3 independent referee repeats of 8 evaluations each. It reuses that
acceptance record only for each run's unbudgeted startup grade. All 24 generated candidates are
graded afresh; infrastructure failures remain separate from kernel attempts.
The launcher also checks the original referee's hash and compares parsed source trees, permitting
changes only to the two DMA constants and `_child_failure`; an outer-loop diagnosis change is rejected.

Keep Qwen/Qwen3-8B at `http://localhost:8000/v1`, temperature 0.6, top_p 0.95, thinking off,
2500 maximum answer tokens, context 8192, one sample per round, no early give-up, and the identical
reference_level4.py start. The original P3 sends no model seed; these are independent stochastic
repeats, **not paired-seed trials**. The private referee input seeds also remain unpredictable.
Do not introduce generation seeds only in v2. Preserve model-server configuration across batches.

Report the original seat-100 referee's 24 evaluations against v2's 24, separately from model-alone
and seat-101 experiments. Primary outcomes: runs finding a held-out-validated `faster` kernel and
best valid speedup per run. Also report correctness rate, DMA-mismatch frequency, and the outcome
of the attempt immediately following a mapped instruction. A reduction in this error alone is a
repair result, not proof of speed improvement. The sequential, unseeded experiment supports a
message-treatment comparison with those limitations; it is not a deterministic counterfactual.

The original three-arm reporting script intentionally requires its pinned referee hash and nine
runs. Do not relabel or concatenate these v2 logs to make them pass that original protocol. Keep
v2's state, source diff, hashes, acceptance and infrastructure records beside its three log files.
