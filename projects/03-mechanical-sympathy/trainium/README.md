# Trainium port of Samudra

This folder runs Samudra2 inference on one Trainium2 NeuronCore through torch-xla. It contains
the adapter for `runners/trainium_runner.py`, a standalone copy of the model, and the benchmark,
probe and agent tools we used on seat 212.

## Files

| File | Purpose |
| --- | --- |
| `candidate_adapter.py` | Adapter for `runners/trainium_runner.py`. Runs the CPU fixture's inputs on Trainium. Use `--precision fp32` or `bf16-autocast` |
| `samudra_core.py` | Standalone inference copy of the `samudra_om4_v2` UNet. Same parameter names as Samudra, so checkpoints load strictly. Bit-identical to the original modules (`verify_vs_original.py`) |
| `verify_vs_original.py` | Checks `samudra_core.py` against a Samudra checkout: `python verify_vs_original.py <samudra-root>` |
| `samudra_tiled.py` | Optional latitude-band tiling of the large UNet blocks (see "Segmenting and tiling" below). Used by `bench.py --tile-rows` |
| `test_tiled.py` | CPU check that the tiled model matches the untiled one: `python test_tiled.py --ckpt <ema_ckpt.pt>` |
| `bench.py` | Benchmark plus check against our own CPU reference (random inputs or real weights via `--hf onedeg`). Appends to `results.jsonl` |
| `status.py` | Shows what is running and whether each `--tag` run finished, is running, or crashed |
| `probe_ops.py` | Compiles single ops or model slices on the chip in seconds and compares them with the CPU |
| `sweep.py` | One forward pass per grid (2, 1, 1/2 and 1/4 degree) on the CPU and on Trainium |
| `agent_loop.py` | Kernel agent: Qwen3-8B (vLLM) writes a fused InstanceNorm + CappedGELU NKI kernel. Includes a staged checker, `--selftest`, `--feedback v1/v2`, `--prompt v1/v2` and `--repeat` |
| `card_check_kernel.py` | A correct kernel built only from the API card that `--prompt v2` adds. It proves the card's facts work in NKI 0.6 and that the task can be solved. The agent never reads it, and it is never shown to the model |
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
  --precision fp32
```

Use `bf16-autocast` for the compiler auto-cast run. Run the checker in
`--diagnostic-only` mode before the team freezes precision tolerances. The
diagnostic report has no correctness pass and no performance result.

The first call compiles for about 5 minutes; later runs use the Neuron compile cache. Set
`SAMUDRA_CKPT` if the checkpoint is not at the manifest's `checkpoint.path_on_seat`.

## Agent experiments

Each version changes one thing. All use 8 rounds x 4 samples x 3 runs with Qwen3-8B.

| Version | Command flags | What changes |
| --- | --- | --- |
| v1 | `--feedback v1 --prompt v1` | Baseline: the organizers' error enrichment, the task, and an example kernel |
| v2 | `--feedback v2 --prompt v1` | Feedback quotes the failing line and gives its exact rewrite (`keepdims`, `(rows, 1)`, the real function for invented names) |
| v3 | `--feedback v2 --prompt v2` | Adds a verified NKI API card to every prompt |

```bash
python agent_loop.py --selftest      # prove the checker first: SELFTEST OK
python agent_loop.py --rounds 8 --samples 4 --repeat 3 --feedback v2 --prompt v2 --log attempts_v3.jsonl
python analyze_attempts.py attempts_v1.jsonl attempts_v2.jsonl attempts_v3.jsonl
python -c "import agent_loop as al; print(al.grade(open('card_check_kernel.py').read(), 2e-2))"   # 1.0
```

Keep each candidate result in its JSONL attempt log. Add one summary row per run to
`../results/agent_attempts.csv`. This file tracks kernel-agent solve results. It stays
separate from `../results/attempts.csv`, which tracks Samudra forecast inference results.

The agent needs the seat's vLLM server (Qwen3-8B, `./serve.sh`) and the organizers'
`/workspace/projects/02-kernel-agent`. It does not need a free NeuronCore, because kernels are
graded in the NKI CPU simulator.

## Segmenting and tiling (seat-211)

At 1 degree the two full-resolution ConvNeXt blocks have 84 MB and 145 MB intermediate tensors,
far more than on-chip memory, and the device profile shows about 55 GB of spill per forward pass
(`runners/profiling.py`, `PROFILING.md`). Two ways to give the compiler smaller problems:

- **Segmenting** (`--segments 1`): a `torch_xla.sync()` after every UNet layer, so each layer is
  compiled as its own graph. The math is unchanged.
- **Tiling** (`--tile-rows R`): `samudra_tiled.py` runs each large block on bands of `R` latitude
  rows. Each band carries the halo its two 3x3 convolutions need; rows beyond a pole are zeroed
  before each 3x3 convolution, as the original pads its intermediate tensor. `--tile-band-cut 1`
  adds a `torch_xla.sync()` after each band (only with `--segments 1`). Tiling refuses
  InstanceNorm blocks, which are not row-local. `test_tiled.py` shows the tiled model is
  bit-identical to the untiled one on the CPU for 15, 20, 45 and 90 row bands (1e-6 relative
  for 10 rows).

Results with `bench.py --grid 1deg --ckpt <onedeg ema_ckpt.pt>`, fp32, batch 1, random inputs,
core 2, median of 20 passes after 3 warm-up passes. Every run passed the check against the
untiled CPU reference (relative RMS error 1.2e-6, land exactly zero). One run per row.

| Configuration | Median ms | Speedup vs plain |
| --- | ---: | ---: |
| Plain | 230.3 | 1.00x |
| `--segments 1` | **138.2** | **1.67x** |
| `--tile-rows 20` | 238.5 | 0.97x |
| `--tile-rows 45` | 156.2 | 1.47x |
| `--segments 1 --tile-rows 20` | 151.2 | 1.52x |
| `--segments 1 --tile-rows 20 --tile-band-cut 1` | 140.8 | 1.64x |
| `--segments 1 --tile-rows 45` | 146.8 | 1.57x |
| `--segments 1 --tile-rows 45 --tile-band-cut 1` | 139.8 | 1.65x |

Segmenting alone is the fastest. Tiling helps without segmenting when bands are large enough
(45 rows), but adds nothing on top of segmenting. Without a cut between bands the compiler may
interleave them; the cuts recover most of the loss. Why segmenting helps has not been profiled
yet. The segments runs compiled in about 100-140 s instead of about 300 s, because the
per-layer graphs of untiled layers were already in the compile cache.

```bash
PYTHONPATH=/workspace/xla_plugin python bench.py --device cpu --grid 1deg --ckpt $CK   # reference
PYTHONPATH=/workspace/xla_plugin python bench.py --device neuron --grid 1deg --ckpt $CK --cores 2 \
  --segments 1 --tile-rows 45 --tile-band-cut 1
```

## Things to know

- **Input cut.** With `--segments 1` (or `--input-cut 1`), `SamudraNet.forward` puts a
  `torch_xla.sync()` right after joining the prognostic and boundary inputs. With the join in the
  same graph, neuronx-cc runs the first block in small, ragged matmuls (47.9 ms); with the cut it
  takes 15.8 ms. Segmented 1 degree pass: 138.2 to 107.3 ms. See
  `../results/input-cut-2026-10-10/README.md` and `probe_block0.py`.
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
