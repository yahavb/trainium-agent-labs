# Profiling an attempt

Profiling is opt-in. It answers *where* the time goes, so you can pick what to
optimize. It never replaces the timed run: **do not quote timings from a
profiled run.**

```text
trainium_runner.py --torch-profile DIR   host side: Python, graph recording, CPU ops
        |
        | (runner exits and frees the NeuronCore)
        v
runners/profiling.py neuron ...          device side: engines, memory traffic, spill
```

| Question | Tool | Output |
| --- | --- | --- |
| Which PyTorch operators cost the most on the host? | `--torch-profile` | `torch_ops.txt`, `torch_trace.json`, `torch_stacks.txt` |
| What does the NeuronCore spend its time on? | `profiling.py neuron` | `summary.md`, `metrics.json`, `summary.json` |
| How long does each stage of my own loop take? | `StageTimer` in `runners/profiling.py` | `timings.json` |

On Trainium, `torch.profiler` sees only the host: recording the graph and
waiting for the device. Use the Neuron profile for device time.

## One-time setup on your pod

Run these from your pod shell (`kubectl exec -it <your-seat> -- bash`).

1. Use the Neuron Python environment. It has torch, torch-xla and the Neuron
   compiler:

   ```bash
   /opt/conda/bin/python -c "import torch, torch_xla; print(torch.__version__)"
   ```

2. Install the torch-xla Neuron plugin into a side folder, if
   `/workspace/xla_plugin` does not exist yet. This leaves the global
   environment and vLLM untouched:

   ```bash
   pip install --no-deps --target /workspace/xla_plugin libneuronxla \
     --extra-index-url https://pip.repos.neuron.amazonaws.com
   ```

   Your adapter must put `/workspace/xla_plugin` on `sys.path` (or run with
   `PYTHONPATH=/workspace/xla_plugin`).

3. Find a free NeuronCore. vLLM usually holds cores 0 and 1. Use a core that
   prints `OK NEURON`:

   ```bash
   for c in 0 1 2 3; do echo "== core $c"; \
     PYTHONPATH=/workspace/xla_plugin PJRT_DEVICE=NEURON NEURON_RT_VISIBLE_CORES=$c \
     /opt/conda/bin/python -c "import torch, torch_xla, torch_xla.runtime as xr; \
     d = torch_xla.device(); print('OK', xr.device_type(), (torch.ones(2, device=d) * 2).cpu())" \
     2>&1 | grep -E "^OK|not available" | head -1; done
   ```

4. Check the profiler is installed. `neuron-profile` was removed; the tool is
   `neuron-explorer`:

   ```bash
   neuron-explorer --version
   ```

5. Set CPU threads to your pod's quota. The host shows 192 cores, but a seat
   gets about 11. Too many threads makes CPU runs many times slower:

   ```bash
   cat /sys/fs/cgroup/cpu.max         # "1100000 100000" means 11 CPUs
   export OMP_NUM_THREADS=11
   ```

## Profile an attempt

From `projects/03-mechanical-sympathy`, replace `CORE` with your free core.

**Step 1: run the attempt.** Add `--torch-profile` for the host-side report.
The profiled runs happen after the timed runs, so the latency in
`metrics.json` is not affected:

```bash
PJRT_DEVICE=NEURON NEURON_RT_VISIBLE_CORES=CORE \
  /opt/conda/bin/python runners/trainium_runner.py \
  --adapter path/to/adapter.py \
  --fixture fixtures/cpu_reference.npz \
  --candidate-output runs/my-attempt/candidate.npz \
  --metrics-json runs/my-attempt/metrics.json \
  --precision float32 \
  --torch-profile runs/my-attempt/torch-profile
```

The runner records when `prepare()` started compiling
(`compile_started_unix` in `metrics.json`). Step 2 uses it to find the
program your adapter compiled.

**Step 2: profile the device.** Run this after the runner exits. The profiler
needs the NeuronCore to itself, and the runner holds it until it exits:

```bash
/opt/conda/bin/python runners/profiling.py neuron \
  --metrics-json runs/my-attempt/metrics.json \
  --out-dir runs/my-attempt/neuron-profile \
  --core CORE
```

This picks the largest program compiled during the run from the compile
cache (`/var/tmp/neuron-compile-cache`), runs it once under
`neuron-explorer capture`, and writes:

| File | What it is |
| --- | --- |
| `summary.md` | Readable table: engine time, utilization, memory-bound or compute-bound, spill |
| `metrics.json` | The same key numbers, for scripts and the attempt log |
| `summary.json` | Every metric `neuron-explorer` reports |
| `profile.ntff` | Raw profile. Can be about 1 GB. Ignored by Git |
| `model.neff` | Copy of the profiled program. Ignored by Git |

If the adapter loaded its program from the compile cache instead of compiling
it, step 2 finds nothing new. Pass the program directly with
`--neff /var/tmp/neuron-compile-cache/<compiler>/MODULE_<hash>/model.neff`.
List candidates, newest first, with
`ls -lt /var/tmp/neuron-compile-cache/*/MODULE_*/model.neff`.

To re-print a summary later without the device:

```bash
/opt/conda/bin/python runners/profiling.py summarize runs/my-attempt/neuron-profile/summary.json
```

## Time the stages of your own loop

For a rollout adapter, `StageTimer` breaks one step into its parts. Use
`sync=True` around device work. torch-xla is lazy, so without the sync the
timer measures graph recording instead of compute:

```python
from profiling import StageTimer   # runners/ is on sys.path when the runner runs

timer = StageTimer()
for step in range(steps):
    with timer.time("read_boundary"):
        boundary = load_boundary(step)
    with timer.time("model", sync=True):
        state = model(state, boundary)
    with timer.time("write"):
        save(state)
timer.write_json(Path("runs/my-attempt/timings.json"))
```

`timings.json` reports each stage's count, total, median, first call and
median without the first call. The first call carries warm-up and, on
Trainium, compilation.

## Read the device summary

Example: Samudra 2, 1 degree, fp32, one forward pass, seat-211
(`tests/neuron_summary_samudra_1deg_fp32.json`):

| Engine | Active ms | Share of pass |
| --- | ---: | ---: |
| DMA (memory transfers) | 148.3 | 65% |
| Tensor engine (matmul / conv) | 141.6 | 62% |
| Sync engine | 44.6 | 20% |
| Scalar engine (activations) | 39.8 | 17% |
| Vector engine (element-wise, norms) | 22.2 | 10% |
| GpSimd engine | 2.3 | 1% |

| Indicator | Value |
| --- | ---: |
| Compute utilization (MFU) | 20% |
| Arithmetic intensity / balance point | 31.8 / 109.8 |
| Bound by | memory |
| Spill save + reload per pass | 23.38 GB + 31.61 GB |
| Inputs + weights | 0.58 GB |

How to read it:

- **Engines overlap.** Each row is the time that engine was busy. The rows
  do not add up to the pass time.
- **Bound by.** If arithmetic intensity is below the balance point, the
  program cannot keep the math units fed: it is memory-bound. Moving less data
  helps more than faster math.
- **Spill** is intermediate data that does not fit in on-chip memory and goes
  to HBM and back. Here it is about 95% of HBM traffic. Changes that cut spill
  are the main lever: bf16 (with a justified tolerance), tiling or segmenting
  the graph, compiler flags, and fusing operations.

Compare `metrics.json` between attempts to see whether a change cut spill or
raised utilization. Report speed from the unprofiled timed run and the attempt
log, not from these files.

## Large files

`runs/`, `*.neff` and `*.ntff` are ignored by Git. Keep raw profiles on your
pod. Commit only small summaries, such as `summary.md`, when a result needs
them.
