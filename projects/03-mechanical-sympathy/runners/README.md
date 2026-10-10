# Runner options

`cpu_rollout.py` retains its prediction-writing interface and enables the fast
inference path by default. Use `--no-optimized` to restore the previous thread,
checkpointing, and on-demand forcing behavior. `--threads N` and explicit forcing
preload flags override their respective defaults.

The default path uses eight PyTorch threads unless `--threads` is supplied,
disables activation checkpointing and parameter gradients for inference, preloads
forcing before the rollout, and directly reuses the prediction as the next history
when input/output step counts match. Prediction fields are still written to Zarr.
Preloading takes additional resident memory and is included in total wall time;
`boundary_preload_seconds` reports it separately. This does not establish forecast
accuracy or guarantee a speedup on a different CPU.

Independent options:

- `--data-root DIR`: read local OM4, means, and standard-deviation stores.
- `--threads N`: explicitly set PyTorch CPU threads.
- `--preload-boundaries` / `--no-preload-boundaries`: override the optimized-mode default.

Example (use a fresh output directory):

```bash
python runners/cpu_rollout.py \
  --samudra-root /amogh/samudra-one-degree \
  --checkpoint /amogh/samudra-one-degree/checkpoints/onedeg/ema_ckpt.pt \
  --data-root /amogh/samudra-one-degree/data/om4_onedeg \
  --predictions /tmp/cpu-rollout/predictions.zarr \
  --report /tmp/cpu-rollout/report.md
```

## Forward timing benchmark

`benchmark_forward.py` ports the warmed CPU/Trainium benchmark from the Samudra
workspace. It is a separate entry point because it deliberately omits prediction
writing and uses a prepared input cache; it cannot replace the prediction-producing
CPU runner or the fixture/checker contract of `trainium_runner.py`.

Run from the Samudra root, using absolute cache and output paths. Preparation and
compilation are outside forward timing. Reports include raw timing samples, input
and checkpoint hashes, CPU thread settings, and Neuron configuration. CPU fallback
or compilation during a timed call fails the run. LNC and precision changes can
trigger long compilations; reuse the existing BF16 LNC=2 configuration first.

```bash
cd /amogh/samudra-one-degree
python /amogh/trainium-agent-labs/projects/03-mechanical-sympathy/runners/benchmark_forward.py \
  --samudra-root /amogh/samudra-one-degree samudra_om4_v2/eval.yaml \
  --data /amogh/samudra-one-degree/data/om4_onedeg/data.local.yaml \
  --checkpoint /amogh/samudra-one-degree/checkpoints/onedeg/ema_ckpt.pt \
  --case /amogh/samudra-one-degree/outputs/forward_benchmark/eval_case.pt \
  --backend cpu --threads 8 --max-calls 2 --single-repeats 1 \
  --output /tmp/cpu-forward-benchmark.json
```

The adapter-based `trainium_runner.py` and existing profiling/checker tools are
unchanged. All model weights, prepared inputs, and prediction stores remain outside Git.
