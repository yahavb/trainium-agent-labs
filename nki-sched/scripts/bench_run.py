"""Pod-side: compile a kernel to a NEFF (kept under $ART), check numerics, run once.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 ART=/tmp/art python bench_run.py kernel.py entry K M N
"""

import importlib.util
import os
import sys

import ml_dtypes
import numpy as np

path, entry, K, M, N = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ["NKI_ARTIFACTS_DIR"] = os.environ.get("ART", "/tmp/art")
spec = importlib.util.spec_from_file_location("k", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
kern = getattr(mod, entry)
r = np.random.default_rng(0)
a = r.standard_normal((K, M)).astype(ml_dtypes.bfloat16)
b = r.standard_normal((K, N)).astype(ml_dtypes.bfloat16)
lnc = int(os.environ.get("LNC", "1"))
got = np.asarray((kern[lnc] if lnc > 1 else kern)(a, b)).astype(np.float32)
want = a.astype(np.float32).T @ b.astype(np.float32)
err = float(np.max(np.abs(got - want) / (np.abs(want) + 1.0)))
print("max rel err", err, "OK" if err < 2e-2 else "MISMATCH")
