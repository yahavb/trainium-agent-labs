#!/usr/bin/env python3
"""
device_bench.py -- layer 2 of the checker: REAL latency on the Trainium device. [device]

Everything else in this repo is [sim] (nki.simulate on the CPU). This compiles each kernel to a NEFF,
loads it on ONE logical NeuronCore that the model server is not using, checks its output on the device
against the NumPy reference, then times it with the runtime's device-side trace.

    NEURON_RT_VISIBLE_CORES=0 python device_bench.py agent_solutions/c1_L4_r0.py:nki_matmul_tiled_ ...

API used (nki 0.6; nki.benchmark / nki.baremetal do not exist in this SDK, LOG 11:43):
  nki.compiler.parallel_compile([KernelSpec(...)], {"target": "trn2", ...}) -> NEFF
  nki.runtime.SpikeModel.load_from_neff(neff, core_id=0).benchmark(inputs, outputs, warmup_iter,
      benchmark_iter, mode="device") -> BenchmarkResult(mean_ms, min_ms, max_ms, std_dev_ms, ...)
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import re

import numpy as np


def adapt(src, entry):
    """Mechanically convert a returns-its-output kernel into the compile API's form, where every tensor
    is an annotated parameter (nki.compiler.kernel_builder: "All tensor parameters must have `: Tensor`
    annotation"). Only three edits; the computation is untouched:
      1. def ENTRY(lhsT, rhs ...)  ->  def ENTRY(lhsT: Tensor, rhs: Tensor, result: Tensor ...)
      2. drop the line that allocates `result` in shared_hbm (it is now the parameter)
      3. `return result` -> `return`
    """
    src = "from nki.compiler.kernel_builder.tensor import Tensor\n" + src
    # 4. the compile API traces a PLAIN function: @nki.jit wraps it in a Kernel object whose
    #    __annotations__ the builder cannot see (measured: the same "not annotated" error with it on)
    src = re.sub(r"^\s*@nki\.jit\s*$\n", "", src, flags=re.M)
    # 5. in the compile API a tensor's .dtype is a numpy dtype the builder rejects ("Unknown dtype:
    #    float32"); every input here IS float32, so name the NKI dtype directly -- same arithmetic
    src = re.sub(r"dtype\s*=\s*\w+\.dtype\b", "dtype=nl.float32", src)
    src, n1 = re.subn(rf"def {entry}\(\s*lhsT\s*,\s*rhs\s*", f"def {entry}(lhsT: Tensor, rhs: Tensor, result: Tensor", src, count=1)
    src, n2 = re.subn(r"^\s*result\s*=\s*nl\.ndarray\([^\n]*shared_hbm\)\s*$\n", "", src, count=1, flags=re.M)
    src, n3 = re.subn(r"return\s+result\b", "return", src)
    if not (n1 and n2 and n3):
        raise ValueError(f"adapter could not rewrite {entry}: signature={n1} alloc={n2} return={n3}")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="+", help="FILE.py:entry_name")
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lnc", type=int, default=int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1")))
    ap.add_argument("--out", default="runs/device_bench.json")
    a = ap.parse_args()

    from nki.compiler import KernelSpec, parallel_compile
    from nki.runtime import SpikeModel, SpikeTensor

    r = np.random.default_rng(0)
    lhsT = r.standard_normal((a.K, a.M)).astype(np.float32)
    rhs = r.standard_normal((a.K, a.N)).astype(np.float32)
    want = (lhsT.astype(np.float64).T @ rhs.astype(np.float64)).astype(np.float32)
    floor_bytes = 4 * (lhsT.size + rhs.size + want.size)
    flops = 2.0 * a.M * a.N * a.K

    work = tempfile.mkdtemp(prefix="devbench_")
    sys.path.insert(0, work)
    os.environ["PYTHONPATH"] = work + os.pathsep + os.environ.get("PYTHONPATH", "")
    specs, labels = [], []
    for i, ke in enumerate(a.kernels):
        path, entry = ke.split(":")
        mod = f"bench_k{i}"
        with open(os.path.join(work, mod + ".py"), "w") as f:
            f.write(adapt(open(path).read(), entry))
        specs.append(KernelSpec(kernel_module=mod, kernel_func_name=entry,
                                inputs={"lhsT": lhsT, "rhs": rhs},
                                outputs={"result": np.zeros_like(want)},
                                hyperparams={}, config_id=i, config_label=os.path.basename(path)))
        labels.append(os.path.basename(path))
    summary = parallel_compile(specs, {"target": "trn2", "lnc": a.lnc, "verbose": False},
                               os.path.join(work, "artifacts"), max_workers=4, verbose=False)

    rows = []
    for lbl, res in zip(labels, summary.results):
        row = dict(kernel=lbl, shape=dict(K=a.K, M=a.M, N=a.N), compiled=not res.error_message)
        if res.error_message:
            row["error"] = res.error_message[:500]
            rows.append(row)
            print(f"{lbl}: COMPILE FAILED: {res.error_message[:300]}")
            continue
        model = SpikeModel.load_from_neff(res.neff_path, core_id=0)
        ins = {"lhsT": SpikeTensor.from_numpy(lhsT, name="lhsT"),
               "rhs": SpikeTensor.from_numpy(rhs, name="rhs")}
        outs = {"result": SpikeTensor.from_numpy(np.zeros_like(want), name="result")}
        b = model.benchmark(ins, outs, warmup_iter=a.warmup, benchmark_iter=a.iters, mode="device")
        got = outs["result"].numpy()
        err = float(np.abs(got - want).max() / (np.sqrt((want ** 2).mean()) or 1))
        row.update(correct_on_device=err < 2e-2, max_err_over_rms=err, mean_ms=b.mean_ms,
                   min_ms=b.min_ms, std_ms=b.std_dev_ms, iters=b.iterations,
                   achieved_gflops=flops / (b.min_ms * 1e-3) / 1e9,
                   floor_bandwidth_gbps=floor_bytes / (b.min_ms * 1e-3) / 1e9)
        rows.append(row)
        print(f"{lbl:28s} correct={row['correct_on_device']} (err {err:.2e})  "
              f"mean {b.mean_ms:.4f} ms  min {b.min_ms:.4f} ms  sd {b.std_dev_ms:.4f}  "
              f"{row['achieved_gflops']:.1f} GFLOP/s  [device, n={b.iterations}]")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
