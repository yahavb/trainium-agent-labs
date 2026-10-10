"""layer_infer.py -- one model layer, run as real inference through torch_neuronx, three ways.

The layer is y = x W with a fixed weight W [K, N] and an input of M tokens. The input is passed
already transposed (lhsT, [K, M]) so every variant sees the same tensors:

  compiler        lhsT.t() @ W, everything chosen by neuronx-cc (no hand-written kernel)
  nki:M,N,K       the team's blocked NKI kernel inside the traced model, at those block settings

For each variant: trace, check against CPU, time repeated inference calls (wall clock, includes
moving the input and output between host and chip), then capture a hardware profile of the
compiled model (time on the chip only).

    python layer_infer.py K256_M4096_N12288 compiler
    python layer_infer.py K256_M4096_N12288 nki:4,1,2
"""
import glob, json, os, shutil, statistics, subprocess, sys, time

os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"      # one physical core, as the team's other measurements
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import numpy as np
import torch
import torch_neuronx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

tag, variant = sys.argv[1], sys.argv[2]
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 50
d = dict((p[0], int(p[1:])) for p in tag.split("_"))
K, M, N = d["K"], d["M"], d["N"]

r = np.random.default_rng(0)
W0 = torch.from_numpy(r.standard_normal((K, N)).astype(np.float32))
lhsT = torch.from_numpy(r.standard_normal((K, M)).astype(np.float32))


class CompilerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(W0.clone(), requires_grad=False)

    def forward(self, lhsT):
        return lhsT.t() @ self.W


class NkiLayer(torch.nn.Module):
    def __init__(self, knobs):
        super().__init__()
        self.W = torch.nn.Parameter(W0.clone(), requires_grad=False)
        self.knobs = knobs
        from matmul_blocked import nki_matmul_
        self.kernel = nki_matmul_

    def forward(self, lhsT):
        m, n, k = self.knobs
        return self.kernel(lhsT, self.W, TILES_IN_BLOCK_M=m, TILES_IN_BLOCK_N=n, TILES_IN_BLOCK_K=k)


if variant == "compiler":
    model = CompilerLayer()
else:
    model = NkiLayer(tuple(int(v) for v in variant.split(":")[1].split(",")))
model.eval()

out = dict(shape=tag, variant=variant, calls=calls)
work = os.path.join(HERE, "work", f"{tag}_{variant.replace(':', '_').replace(',', '-')}")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
try:
    t = time.time()
    traced = torch_neuronx.trace(model, (lhsT,), compiler_workdir=work, compiler_args=["--lnc=1"])
    out["compile_s"] = round(time.time() - t, 1)
    want = (lhsT.t() @ W0).numpy()
    got = traced(lhsT).numpy()
    out["correct"] = bool(got.shape == want.shape and np.allclose(got, want, rtol=2e-2, atol=2e-2))
    out["max_abs_err"] = float(np.abs(got - want).max())
    for _ in range(5):
        traced(lhsT)
    ts = []
    for _ in range(calls):
        t = time.perf_counter()
        traced(lhsT)
        ts.append((time.perf_counter() - t) * 1e3)
    out.update(wall_ms_median=round(statistics.median(ts), 3), wall_ms_min=round(min(ts), 3),
               wall_ms_p90=round(sorted(ts)[int(0.9 * len(ts))], 3))
    neffs = glob.glob(os.path.join(work, "**", "*.neff"), recursive=True)
    out["neff"] = neffs
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e)[:700]}"
    print(json.dumps(out))
    sys.exit(1)

# Free the chip before profiling: neuron-explorer loads the NEFF itself.
del traced
import gc
gc.collect()
print(json.dumps(out), flush=True)
