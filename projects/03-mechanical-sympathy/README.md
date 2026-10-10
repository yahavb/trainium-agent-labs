# Project 03: Mechanical Sympathy

This project checks Samudra inference against a CPU reference and then improves
the same workload on AWS Trainium.

## Experiment contract

The CPU implementation defines correctness. The first correct Trainium result
defines Trainium v0. Later Trainium results compare with Trainium v0.

```text
CPU reference output
        |
        v
correctness checker <----- Trainium candidate
        |                         |
        | fail                    | pass
        v                         v
fix correctness           measure performance
                                  |
                                  +-- primary: rollout throughput
                                  +-- fallback: single-step latency
```

The checker does not expose performance results for an incorrect candidate.

## Repository roles

The hackathon fork is the primary team repository:

```text
https://github.com/pranoyghosh/trainium-agent-labs
```

It contains runners, the checker, attempt records, fixtures, and reports. The
Samudra fork contains only reusable Samudra changes:

```text
https://github.com/pranoyghosh/Samudra
```

`samudra-source.json` pins the exact Samudra commit. Run this command after the
fork exists:

```bash
./bootstrap_samudra.sh
```

See `GIT_WORKFLOW.md` for branch and remote commands.

## CPU reference

The released Samudra 2 one-degree EMA checkpoint and the one-degree OM4 source
are the reference inputs. First create one repeatable real-data fixture:

```bash
cd .scratch/Samudra
source .venv/bin/activate
python ../../runners/data_smoke.py \
  --samudra-root "$PWD" \
  --checkpoint ../../checkpoints/Samudra2/onedeg/ema_ckpt.pt \
  --data-root ../../data/om4-smoke \
  --reference-output ../../fixtures/cpu_reference.npz
```

Run this command twice. Compare the two CPU files. Then add these values to
`fixtures/manifest.json`:

- Checkpoint SHA-256.
- Reference file SHA-256.
- Numeric `atol` and `rtol`.
- `status: "frozen"`.

Do not freeze tolerances from a Trainium result. Use CPU repeatability and the
documented Trainium precision.

The full CPU rollout uses:

```bash
cd .scratch/Samudra
source .venv/bin/activate
python ../../runners/cpu_rollout.py \
  --samudra-root "$PWD" \
  --checkpoint ../../checkpoints/Samudra2/onedeg/ema_ckpt.pt \
  --predictions ../../rollouts/cpu_8_year/predictions.zarr \
  --report ../../results/cpu-baseline.md
```

The runner writes a small JSON manifest beside the Zarr store. Keep the Zarr
store outside Git.

## Correctness checker

The CPU and candidate files are compressed NumPy archives. Both files contain:

- `prediction`
- `time`
- `variables`

The CPU fixture also contains the model inputs and grid context. Run the
checker with:

```bash
python checker.py \
  --manifest fixtures/manifest.json \
  --candidate runs/trainium/candidate.npz \
  --performance-json runs/trainium/metrics.json \
  --json-out runs/trainium/check.json
```

Exit status `0` means correct. Exit status `1` means a correctness failure.
Exit status `2` means the reference or checker configuration is not ready.

## Trainium runner contract

`runners/trainium_runner.py` loads an adapter file. The adapter implements:

```python
def prepare(inputs, context):
    return prepared_candidate
```

The prepared candidate implements:

```python
def run():
    return prediction

def synchronize():
    ...
```

Compilation must happen in `prepare`. `synchronize` must wait for all device
work. This separation keeps compilation and warm-up out of steady-state timing.

Use `agent.py` to run and record one graded attempt:

```bash
python agent.py \
  --experiment trainium-v0 \
  --adapter trainium/candidate_adapter.py \
  --fixture fixtures/cpu_reference.npz \
  --precision bfloat16 \
  --workload single-step
```

The controller appends the result to `results/attempts.csv`. It adds timing
only when correctness passes.

## Performance rule

The primary metric is completed forecast steps per steady-state second. Keep
input data, checkpoint, shape, precision, and rollout length fixed.

Use single-step median and p95 latency only when:

1. One-step Trainium inference passes correctness.
2. Two full-rollout attempts are recorded.
3. Both fail because of an unsupported operation, repeated compilation, graph
   growth, or device memory.

Do not report CPU speed as the Trainium baseline.

## Files that stay outside Git

- `.scratch/`
- Checkpoints
- Datasets
- Full Zarr predictions
- Rollouts
- Neuron compiler caches
- Environment files and credentials
- `pod-remote/`
