"""Smoke test: compile and run a trivial NKI kernel on NeuronCore 0 while vLLM holds NC2/NC3.
Run:  NEURON_RT_VISIBLE_CORES=0 python baseline/smoke_nc0.py
"""
import os, time, numpy as np
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
import nki, nki.isa as nisa, nki.language as nl

@nki.jit
def add_one(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    o = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o, data=t, op0=nl.add, operand0=1.0)
    nisa.dma_copy(dst=out, src=o)
    return out

x = np.arange(128 * 512, dtype=np.float32).reshape(128, 512)
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
t0 = time.time(); y = add_one[lnc](x); t1 = time.time()
print(f"first call (compile+run): {t1-t0:.1f}s, correct={np.allclose(y, x + 1)}")
t0 = time.time()
for _ in range(10): y = add_one[lnc](x)
print(f"warm call avg: {(time.time()-t0)/10*1000:.2f} ms (includes host round trip)")
