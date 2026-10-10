"""Smoke test: torch.compile(backend="neuron_libtorch") on a free NeuronCore, outside vLLM.
Run:  NEURON_RT_VISIBLE_CORES=2 python baseline/smoke_torch_compile.py
"""
import os, time, torch
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")
import libtorch_neuronx_lite  # registers the 'neuron' device and the backends
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29571")
    dist.init_process_group("gloo", rank=0, world_size=1)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(4096, 12288, bias=False, dtype=torch.bfloat16)
        self.w2 = torch.nn.Linear(12288, 4096, bias=False, dtype=torch.bfloat16)
    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)))

m = Tiny().to("neuron:0")
x = torch.randn(1, 4096, dtype=torch.bfloat16).to("neuron:0")
cm = torch.compile(m, backend="neuron_libtorch", fullgraph=True)
t0 = time.time(); y = cm(x); print("first call (compile+run) %.1fs" % (time.time() - t0), y.shape, y.dtype, y.device)
t0 = time.time()
for _ in range(20):
    y = cm(x); _ = y[0, 0].cpu()   # the executor is async: force a sync each call
print("warm: %.3f ms/call (includes a 1-element D2H sync)" % ((time.time() - t0) / 20 * 1000))
ref = m.cpu()(x.cpu())
print("max abs diff vs cpu:", (y.cpu().float() - ref.float()).abs().max().item())
