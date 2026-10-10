"""Does the compiler's prediction change at lnc=2 (this seat's NEURON_LOGICAL_NC_CONFIG) for unsharded kernels?"""
import json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, "/workspace/projects/02-kernel-agent")
import numpy as np, nkibench
from predict import predict
import peaks_probe as pp
from calibrate import load_baked
k4 = load_baked("/workspace/projects/02-kernel-agent/reference_level4.py", "nki_matmul_tiled_", {}, "ref4")
args, _ = nkibench.make_inputs(dict(K=256, M=512, N=1024), 4)
x = np.random.default_rng(0).standard_normal((128, 32768)).astype(np.float32)
for lnc in (1, 2):
    out = {}
    for lab, k, inp in (("ref4_512x256x1024", k4, {"lhsT": args[0], "rhs": args[1]}),
                        ("dma_32MiB", pp.dma_kernel_factory(), {"x": x})):
        try:
            out[lab] = predict(k, inp, lnc=lnc)["ns"]
        except Exception as e:
            out[lab] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps(dict(lnc=lnc, **out)))
