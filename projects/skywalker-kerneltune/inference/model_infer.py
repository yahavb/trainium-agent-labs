"""model_infer.py -- inference time of a small model whose matmuls are either the compiler's or ours.

Models, each traced with torch_neuronx and run on one NeuronCore (LNC=1, float32):

  stack:S:L    L matmul layers at size S x S, each h <- h.T @ W_i, then a mean over rows.
  pool:TAG     one layer at shape TAG (e.g. K256_M4096_N12288), then a mean over rows.
  mlp:T:B      B feed-forward blocks at Qwen3-8B's sizes (hidden 4096, intermediate 12288) on
               T tokens: x <- x + (silu(x Wg) * (x Wu)) Wd, then a mean over tokens.

Variants:

  compiler          plain PyTorch, neuronx-cc default optimisation (-O2)
  compiler-O1       plain PyTorch, lowest optimisation level
  compiler-O3       plain PyTorch, highest optimisation level
  compiler-bf16     plain PyTorch, with the compiler told to run matmuls in bfloat16
  torch-bf16        plain PyTorch with its inputs and weights cast to bfloat16, default compile
  nki:M,N,K         the team's blocked NKI kernel in every matmul at those block settings
  nkibf16:M,N,K     the same kernel fed bfloat16 tensors (inputs and weights cast once)
  nki:a,b,c/d,e,f   (mlp only) settings for the gate/up matmuls, then for the down matmul

Weights and inputs are identical across variants.
"""
import glob, json, os, shutil, statistics, sys, time

os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"

import numpy as np
import torch
import torch_neuronx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

spec, variant = sys.argv[1], sys.argv[2]
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 50
kind = spec.split(":")[0]
r = np.random.default_rng(0)


def rnd(*shape, scale=1.0):
    return torch.from_numpy((r.standard_normal(shape) * scale).astype(np.float32))


use_nki = variant.startswith("nki")
BF16 = variant.startswith("nkibf16") or variant == "torch-bf16"   # tensors cast to bfloat16 in the model
cc_args = ["--lnc=1"]
if variant.startswith("compiler-O"):
    cc_args.append(f"--optlevel={variant[-1]}")
if variant == "compiler-bf16":                # the compiler's own switch for 16-bit matmuls
    cc_args += ["--auto-cast=matmult", "--auto-cast-type=bf16"]
knob_sets = [tuple(int(v) for v in part.split(",")) for part in variant.split(":")[1].split("/")] if use_nki else []
if use_nki:
    from matmul_blocked import nki_matmul_


DT = torch.bfloat16 if BF16 else torch.float32


def mm(lhsT, W, knobs):
    """lhsT.T @ W, by the kernel or by the compiler."""
    if use_nki:
        m, n, k = knobs
        return nki_matmul_(lhsT, W, TILES_IN_BLOCK_M=m, TILES_IN_BLOCK_N=n, TILES_IN_BLOCK_K=k)
    return lhsT.t() @ W


if kind in ("stack", "pool"):
    if kind == "stack":
        S, L = int(spec.split(":")[1]), int(spec.split(":")[2])
        K, M, N = S, S, S
    else:
        d = dict((p[0], int(p[1:])) for p in spec.split(":")[1].split("_"))
        K, M, N, L = d["K"], d["M"], d["N"], 1
    Ws = [rnd(K, N, scale=1 / np.sqrt(K)) for _ in range(L)]
    x0 = rnd(K, M)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Ws = torch.nn.ParameterList([torch.nn.Parameter(w.clone().to(DT), requires_grad=False) for w in Ws])

        def forward(self, h):
            h = h.to(DT)
            for W in self.Ws:
                h = mm(h, W, knob_sets[0] if use_nki else None)
            return h.float().mean(dim=0)

    def reference(h):
        for W in Ws:
            h = h.t() @ W
        return h.mean(dim=0)

else:  # mlp
    T, B = int(spec.split(":")[1]), int(spec.split(":")[2])
    H, I = 4096, 12288
    L = 3 * B
    blocks = [(rnd(H, I, scale=1 / np.sqrt(H)), rnd(H, I, scale=1 / np.sqrt(H)), rnd(I, H, scale=1 / np.sqrt(I)))
              for _ in range(B)]
    x0 = rnd(T, H)
    k_up = knob_sets[0] if use_nki else None
    k_down = knob_sets[-1] if use_nki else None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Wg = torch.nn.ParameterList([torch.nn.Parameter(b[0].clone().to(DT), requires_grad=False) for b in blocks])
            self.Wu = torch.nn.ParameterList([torch.nn.Parameter(b[1].clone().to(DT), requires_grad=False) for b in blocks])
            self.Wd = torch.nn.ParameterList([torch.nn.Parameter(b[2].clone().to(DT), requires_grad=False) for b in blocks])

        def forward(self, x):
            x = x.to(DT)
            for Wg, Wu, Wd in zip(self.Wg, self.Wu, self.Wd):
                xT = x.t()
                g = torch.nn.functional.silu(mm(xT, Wg, k_up)) * mm(xT, Wu, k_up)
                x = x + mm(g.t(), Wd, k_down)
            return x.float().mean(dim=0)

    def reference(x):
        for Wg, Wu, Wd in blocks:
            g = torch.nn.functional.silu(x @ Wg) * (x @ Wu)
            x = x + g @ Wd
        return x.mean(dim=0)


out = dict(model=spec, variant=variant, matmuls=L, calls=calls)
work = os.path.join(HERE, "work", f"{spec.replace(':', '_')}__{variant.replace(':', '_').replace(',', '-').replace('/', '_')}")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
try:
    t = time.time()
    traced = torch_neuronx.trace(Model().eval(), (x0,), compiler_workdir=work, compiler_args=cc_args)
    out["compile_s"] = round(time.time() - t, 1)
    want = reference(x0).numpy()
    got = traced(x0).numpy()
    # Relative to the size of the answer: the pooled outputs are small numbers.
    out["rel_err"] = float(np.abs(got - want).max() / np.abs(want).max())
    out["correct"] = bool(got.shape == want.shape and out["rel_err"] < 2e-2)
    for _ in range(5):
        traced(x0)
    ts = []
    for _ in range(calls):
        t = time.perf_counter()
        traced(x0)
        ts.append((time.perf_counter() - t) * 1e3)
    out.update(wall_ms_median=round(statistics.median(ts), 3), wall_ms_min=round(min(ts), 3),
               wall_ms_max=round(max(ts), 3))
    out["neff"] = glob.glob(os.path.join(work, "**", "*.neff"), recursive=True)[0]
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e)[-600:]}"
print(json.dumps(out), flush=True)
