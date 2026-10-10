"""Second probe: how the compiler's model scales matmul cost with K, M, N, the fixed latency of a
tiny kernel, and per-instruction overhead on Vector/Scalar."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict
import peaks_probe as pp

rng = np.random.default_rng(0)
res = {}
for (K, M, N) in [(128, 128, 512), (128, 128, 256), (128, 128, 128), (64, 128, 512), (128, 64, 512)]:
    l = rng.standard_normal((K, M)).astype(np.float32)
    r = rng.standard_normal((K, N)).astype(np.float32)
    t = {n: predict(pp.mm_kernel_factory(n, "bfloat16"), {"lhsT": l, "rhs": r})["ns"] for n in (2, 34)}
    per = (t[34] - t[2]) / 32
    res[f"mm_K{K}_M{M}_N{N}_ns_per_matmul"] = per
    print(K, M, N, per, flush=True)
# tiny kernel: fixed cost of a load + store
x = rng.standard_normal((128, 1)).astype(np.float32)
res["tiny_dma_128x1_ns"] = predict(pp.dma_kernel_factory(), {"x": x})["ns"]
x = rng.standard_normal((128, 64)).astype(np.float32)
res["tiny_dma_128x64_ns"] = predict(pp.dma_kernel_factory(), {"x": x})["ns"]
# per-instruction overhead: small free size
for F in (64, 512, 2048):
    x = rng.standard_normal((128, F)).astype(np.float32)
    for kind in ("vector_ts", "scalar_exp"):
        t = {n: predict(pp.engine_kernel_factory(kind, n, "float32"), {"x": x})["ns"] for n in (2, 34)}
        res[f"{kind}_F{F}_ns_per_op"] = (t[34] - t[2]) / 32
        print(kind, F, res[f"{kind}_F{F}_ns_per_op"], flush=True)
print(json.dumps(res, indent=1))
json.dump(res, open("/tmp/roof_peaks_probe2.json", "w"), indent=1)
