# Trainium port of Samudra

This folder runs Samudra2 inference on one Trainium2 NeuronCore through torch-xla. It contains
the adapter for `runners/trainium_runner.py`, a standalone copy of the model, and the benchmark,
probe and agent tools we used on seat 212.

## Files

| File | Purpose |
| --- | --- |
| `candidate_adapter.py` | Adapter for `runners/trainium_runner.py`. Runs the CPU fixture's inputs on Trainium. `--precision float32` or `bfloat16` (compiler auto-cast) |
| `samudra_core.py` | Standalone inference copy of the `samudra_om4_v2` UNet. Same parameter names as Samudra, so checkpoints load strictly. Bit-identical to the original modules (`verify_vs_original.py`) |
| `verify_vs_original.py` | Checks `samudra_core.py` against a Samudra checkout: `python verify_vs_original.py <samudra-root>` |
| `bench.py` | Benchmark plus check against our own CPU reference (random inputs or real weights via `--hf onedeg`). Appends to `results.jsonl` |
| `status.py` | Shows what is running and whether each `--tag` run finished, is running, or crashed |
| `probe_ops.py` | Compiles single ops or model slices on the chip in seconds and compares them with the CPU |
| `sweep.py` | One forward pass per grid (2, 1, 1/2 and 1/4 degree) on the CPU and on Trainium |
| `agent_loop.py` | Kernel agent: Qwen3-8B (vLLM) writes a fused InstanceNorm + CappedGELU NKI kernel. Includes a staged checker, `--selftest`, `--feedback v1/v2` and `--repeat` |
| `analyze_attempts.py` | Turns agent attempt logs into summary, failure-breakdown, reward-by-round and wall tables |

## One-time setup on a seat

```bash
# torch-xla is installed without its Neuron PJRT plugin. Install it into a separate
# folder so the vLLM environment is not changed.
pip install --no-deps --target /workspace/xla_plugin libneuronxla \
  --extra-index-url https://pip.repos.neuron.amazonaws.com

# Find a free logical NeuronCore. vLLM usually holds cores 0 and 1.
for c in 0 1 2 3; do
  PYTHONPATH=/workspace/xla_plugin PJRT_DEVICE=NEURON NEURON_RT_VISIBLE_CORES=$c python -c \
    "import torch, torch_xla, torch_xla.runtime as xr; d=torch_xla.device(); print('core $c OK', xr.device_type(), (torch.ones(2,device=d)*2).cpu())" \
    2>&1 | grep -E "OK|not available" | head -1
done
```

## Run the adapter on the CPU fixture

Run from `projects/03-mechanical-sympathy`:

```bash
NEURON_RT_VISIBLE_CORES=2 python runners/trainium_runner.py \
  --adapter trainium/candidate_adapter.py --fixture fixtures/cpu_reference.npz \
  --candidate-output runs/trainium/candidate.npz --metrics-json runs/trainium/metrics.json \
  --precision float32
```

The first call compiles for about 5 minutes; later runs use the Neuron compile cache. Set
`SAMUDRA_CKPT` if the checkpoint is not at the manifest's `checkpoint.path_on_seat`.

## Things to know

- **Circular padding.** torch-xla lowers `F.pad(mode="circular")` with a boolean mask constant
  the size of the tensor. neuronx-cc 2.27 cannot parse it once the tensor is large enough
  (`NCC_EMOD021`, "Failed to parse StableHLO"). `samudra_core.wrap_lon` does the same longitude
  wrap with slices and `torch.cat`. The result is bit-identical on the CPU. The official
  Samudra code (`blocks.py`, `samudra.py`, `decoder.py`) still uses `F.pad(mode="circular")`,
  so running it on Trainium unchanged hits this error.
- **bf16 on the 1 degree grid.** Converting the whole model to bf16 crashes the compiler with
  `NCC_IBIR229` (state buffer allocation), also at `--optlevel=1`. Keeping the model in fp32
  with `NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=bf16"` compiles and is 4.4x
  faster than fp32. The adapter's `bfloat16` precision uses this route.
- **Device memory.** One logical core has 24 GB. The 1/2 degree grid in fp32 needs 28.5 GB
  (`NCC_EXSP001`).
- **PJRT_DEVICE.** Set `PJRT_DEVICE=NEURON`. Without it, torch-xla can fall back to the CPU
  without an error.
- **CPU threads.** torch sees the host's 192 cores, but the pod is limited to 11 CPUs. Set
  `torch.set_num_threads(11)`. With 192 threads, our 2 degree CPU forward pass took 54 s;
  with 11 it took 0.36 s.
