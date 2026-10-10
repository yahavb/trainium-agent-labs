"""Run an emitted matmul kernel through nki.simulate with bf16 (and float32) inputs.
nkibench only passes float32; this checks that the dtype-generic emission (dtypes follow the inputs,
fp32 PSUM accumulation, cast on copy-out) is right for bf16 too.

    python sim_dtypes.py kernel.py entry_name          # on a machine with the Neuron SDK
"""

import importlib.util
import sys

import ml_dtypes
import numpy as np
import nki

SHAPES = [(128, 128, 512), (256, 256, 1024), (512, 128, 512), (256, 512, 1024)]  # K, M, N


def main(path, entry):
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
            got = np.asarray(nki.simulate(kernel)(a, b)).astype(np.float32)
            tol = 1e-3 if dt_name == "float32" else 2e-2
            err = np.max(np.abs(got - want) / (np.abs(want) + 1.0))
            ok = got.shape == want.shape and err < tol
            bad += not ok
            print(f"{dt_name:9s} K={K} M={M} N={N}: max rel-ish err {err:.2e}  {'ok' if ok else 'MISMATCH'}")
    print("ALL OK" if not bad else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
