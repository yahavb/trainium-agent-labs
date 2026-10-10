"""
Go/no-go for the latency simulator: time ONE matmul kernel on the device and check its result.

    NEURON_RT_VISIBLE_CORES=2 python time_one.py              # 1024 x 1024 x 1024, bf16
    NEURON_RT_VISIBLE_CORES=2 python time_one.py 2048 1024 512
    DTYPE=fp32 NEURON_RT_VISIBLE_CORES=2 python time_one.py

Cores 0-1 are held by the vLLM server (tensor_parallel_size 2); use 2 or 3.
neuron-bench times with zero-filled inputs, so the latency is valid on its own; the baremetal run
with real inputs is only there to prove the kernel computes the right answer.
"""

import os
import sys
import time

import numpy as np
import neuronxcc.nki as nki

from matmul_tiled_old import nki_matmul_tiled_ as kernel

K, M, N = (int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 else (1024, 1024, 1024)

if os.environ.get("DTYPE", "bf16") == "fp32":
    dtype = np.float32
else:
    from ml_dtypes import bfloat16
    dtype = bfloat16
print(f"shape K={K} M={M} N={N}, dtype={np.dtype(dtype).name}")

rng = np.random.default_rng(0)
lhsT = rng.standard_normal((K, M)).astype(dtype)
rhs = rng.standard_normal((K, N)).astype(dtype)
expect = lhsT.astype(np.float32).T @ rhs.astype(np.float32)

print("\n--- correctness: nki.baremetal with real inputs ---")
t0 = time.perf_counter()
out = np.asarray(nki.baremetal(kernel)(lhsT, rhs)).astype(np.float32)
print(f"  compile + load + run: {time.perf_counter() - t0:.1f} s")
print(f"  out[0,:4]    = {out[0, :4]}")
print(f"  expect[0,:4] = {expect[0, :4]}")
err = np.abs(out - expect).max() / (np.abs(expect).max() + 1e-9)
print(f"  max relative error {err:.2e} ({'OK' if err < 2e-2 else 'WRONG'})")

print("\n--- latency: nki.benchmark (neuron-bench, on device) ---")
bench = nki.benchmark(warmup=5, iters=20)(kernel)
bench(lhsT, rhs)
lat = bench.benchmark_result.full_results["latency"]
print(f"  latency us: p0 {lat['0']}  p50 {lat['50']}  p90 {lat['90']}  p99 {lat['99']}")
print("\nGO" if err < 2e-2 else "\nTiming works, result still WRONG")
