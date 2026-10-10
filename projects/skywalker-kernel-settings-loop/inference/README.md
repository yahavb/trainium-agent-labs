# Inference timing: the kernel inside a traced model

Measures how long a small model takes to run on one NeuronCore when its matmuls are (a) AWS's
compiler default or (b) the team's blocked NKI kernel at a given block setting. Results are written
up in `docs/note.md` section 5.6.

## Files

- `model_infer.py`: traces a model with `torch_neuronx`, checks it against CPU, and times 50 calls.
- `layer_infer.py`: the same for a single layer with no pooling.
- `profile_neff.py`: on-chip time of a compiled model from a `neuron-explorer` capture.
- `model_inference_seat170.jsonl`: raw results, one line per model and variant.

`matmul_blocked.py` from `kernels/` must sit beside the scripts.

## Environment

Measured on seat 170, LNC=1, float32. The pod's own Python has no `torch_neuronx`, so this runs in
a separate environment:

```bash
python -m venv env && . env/bin/activate
pip install torch-neuronx --extra-index-url https://pip.repos.neuron.amazonaws.com
pip install islpy==2026.1
export PATH=$PWD/env/bin:$PATH
python model_infer.py stack:2048:36 compiler
python model_infer.py stack:2048:36 nki:4,2,4
python profile_neff.py work/stack_2048_36__nki_4-2-4/graph.neff 3
```

Three things that had to be fixed before anything ran:

1. **`islpy` version.** A fresh install gets `islpy` 2026.2.2 and every compile fails with `[NCC_ISMP902] Simplifier error: is_subset(): incompatible function arguments`. The pod's working compiler uses 2026.1; pinning to that fixes it.
2. **PATH.** `torch_neuronx` calls a helper named `libneuronpjrt-path` that lives in the environment's `bin`, so that directory must be on `PATH`.
3. **Core setting.** `torch_neuronx.trace` must be given `compiler_args=["--lnc=1"]`. Otherwise the model is compiled for two cores per logical core and the runtime refuses to load it at LNC=1.

## Single layer, no pooling (not in the JSONL)

Adapter shape K=256, M=4096, N=12288, on-chip time, three captures each:

| Variant | On chip (µs) | Tensor engine busy | Bytes moved | Transfers |
|---|---|---|---|---|
| Compiler default | 1,355.1 (1,355.1 / 1,355.2 / 1,355.0) | 97.8% | 240,123,904 | 458 |
| Our kernel at 4/1/2, inside the traced model | 1,460.3 (1,462.2 / 1,460.3 / 1,457.9) | 92.6% | 314,572,800 | 1,200 |

Wall-clock time for this layer alone is about 52 ms for both, because almost all of it is copying
the 201 MB output back to the host. The pooled models avoid that, which is why they are the ones
used for inference time.

## Known gap

The hardware profile of the 36-layer stack with the unblocked kernel (1/1/1) failed to capture
(`Failed to schedule execution request`, status 1204), so that row has a wall-clock time only.
