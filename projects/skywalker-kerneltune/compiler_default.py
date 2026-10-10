"""compiler_default.py -- the baseline a judge asks for: the same matmul with NO hand-written kernel.

Traces a plain `lhsT.T @ rhs` with torch_neuronx so neuronx-cc picks everything itself, runs it on
the chip, and leaves the NEFF in ./cd_<shape>/ for the same neuron-explorer capture hwcheck uses.
Run inside the separate torch 2.9 environment (the pod's own Python has no torch_neuronx):

    /workspace/dfenv/bin/python compiler_default.py K256_M4096_N12288
"""
import glob, os, shutil, sys, time
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"     # one physical core, as every other measurement here

import numpy as np
import torch
import torch_neuronx

tag = sys.argv[1]
d = dict((p[0], int(p[1:])) for p in tag.split("_")) if not tag.isdigit() else dict(K=int(tag), M=int(tag), N=int(tag))
K, M, N = d["K"], d["M"], d["N"]


class MatMul(torch.nn.Module):
    def forward(self, lhsT, rhs):
        return lhsT.t() @ rhs


r = np.random.default_rng(0)
lhsT = torch.from_numpy(r.standard_normal((K, M)).astype(np.float32))
rhs = torch.from_numpy(r.standard_normal((K, N)).astype(np.float32))
work = os.path.abspath(f"cd_{tag}")
shutil.rmtree(work, ignore_errors=True)
t = time.time()
traced = torch_neuronx.trace(MatMul(), (lhsT, rhs), compiler_workdir=work, compiler_args=sys.argv[2:] or None)
print(f"compiled in {time.time() - t:.1f}s")
got = traced(lhsT, rhs).numpy()
want = (lhsT.t() @ rhs).numpy()
print("correct:", bool(np.allclose(got, want, rtol=2e-2, atol=2e-2)), "max abs err", float(np.abs(got - want).max()))
print("NEFF:", glob.glob(os.path.join(work, "**", "*.neff"), recursive=True))
