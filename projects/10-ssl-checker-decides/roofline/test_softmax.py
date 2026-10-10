import os, sys, numpy as np, nki
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import softmax_kernels as sk
def ref(x):
    m = x.max(axis=1, keepdims=True); e = np.exp(x - m); return e / e.sum(axis=1, keepdims=True)
x = (np.random.default_rng(0).standard_normal((256, 512)) * 4).astype(np.float32)
for name in ("softmax_fused", "softmax_reload", "softmax_extrapass"):
    try:
        got = nki.simulate(getattr(sk, name))(x)
        err = float(np.abs(np.asarray(got) - ref(x)).max())
        print(name, "max_abs_err", err, "OK" if err < 1e-5 else "BAD")
    except Exception as e:
        print(name, "ERROR", type(e).__name__, str(e)[:600])
