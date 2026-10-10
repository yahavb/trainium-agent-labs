"""Compile a kernel with the Neuron compiler and RUN IT ON THE NEURONCORE (nki.jit with numpy
arguments = standalone execution, no framework), then compare with numpy and report wall time.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python device_run.py kernel.py entry_name [reps]

Wall time includes launch overhead; it is a smoke-level comparison, not a profile.
"""

import importlib.util
import os
import sys
import time

import ml_dtypes
import numpy as np

SHAPES = [(128, 128, 512), (256, 256, 1024), (512, 128, 512), (256, 512, 1024)]  # K, M, N


def main(path, entry, reps=5):
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    spec = importlib.util.spec_from_file_location("kernel_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, entry)
    bad = 0
    for dt_name, dt in (("float32", np.float32), ("bfloat16", ml_dtypes.bfloat16)):
        for K, M, N in SHAPES:
            r = np.random.default_rng(0)
            a = r.standard_normal((K, M)).astype(dt)
            b = r.standard_normal((K, N)).astype(dt)
            want = a.astype(np.float32).T @ b.astype(np.float32)
            t0 = time.perf_counter()
            try:
                got = np.asarray(kernel(a, b)).astype(np.float32)  # first call compiles
            except Exception as e:  # noqa: BLE001
                if "NRT" in type(e).__name__ or "initialize NRT" in str(e):
                    print("NeuronCores are not available (held by another process on this node?): " + str(e).splitlines()[0])
                    sys.exit(3)
                raise
            first = time.perf_counter() - t0
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                kernel(a, b)
                ts.append(time.perf_counter() - t0)
            tol = 1e-3 if dt_name == "float32" else 2e-2
            err = float(np.max(np.abs(got - want) / (np.abs(want) + 1.0)))
            ok = got.shape == want.shape and err < tol
            bad += not ok
            print(f"{dt_name:9s} K={K} M={M} N={N}: err {err:.2e} {'ok' if ok else 'MISMATCH'}  "
                  f"first {first:.2f}s  steady median {np.median(ts) * 1e3:.2f} ms", flush=True)
    print("ALL OK" if not bad else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
