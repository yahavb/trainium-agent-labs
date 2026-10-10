"""Offline (no-hardware) NKI kernel checks: compile to trn2 NEFF + static latency prediction.

Usage (inside the nki-cc container):
    from nkitool import analyze
    report = analyze(my_kernel, x, w, target="trn2")

What it does:
  1. Lowers the @nki.jit kernel through the NKI MLIR pipeline with emit_predicted_latency=True.
     The pipeline's cost model returns total_time_ns + per-engine estimated_utilization.
  2. Optionally runs neuronx-cc to produce a real NEFF (catches backend errors the CPU
     simulator does not, e.g. SBUF/PSUM allocation, instruction encoding).
"""
import os
import tempfile
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

from nki.compiler.ncc_driver import CompileOptions  # noqa: E402
from nki.framework.compiled import compile_kernel_to_nir  # noqa: E402


def analyze(kernel, *args, target="trn2", lnc=1, neff=True, birsim=False, backend_opt=True, artifacts_dir=None, **kwargs):
    inputs = kernel._bind_args(args, kwargs)
    work = artifacts_dir or tempfile.mkdtemp(prefix="nkitool_")
    opts = CompileOptions(
        target=target,
        lnc=lnc,
        artifacts_dir=work,
        output_path=os.path.join(work, "kernel.neff"),
        emit_predicted_latency=True,
        enable_simulation=birsim,
    )
    if not backend_opt:
        opts = opts.disable_backend_optimizations()

    t0 = time.time()
    nir = compile_kernel_to_nir(kernel, inputs=inputs, compile_opts=opts, enable_cache=False)
    report = dict(
        predicted_ns=nir.total_time_ns,
        utilization=nir.estimated_utilization,
        lower_s=round(time.time() - t0, 2),
        work_dir=work,
    )
    if neff:
        from nki.compiler.driver import _compile_bir_to_neff
        import numpy as np
        t1 = time.time()
        tensors = {k: v for k, v in inputs.items() if isinstance(v, np.ndarray)}
        compiled = _compile_bir_to_neff(nir, opts, tensors)
        if birsim:
            report["birsim_outputs"] = compiled.birsim_outputs
        report["neff"] = compiled.neff_path
        report["neff_bytes"] = os.path.getsize(compiled.neff_path)
        report["neuronx_cc_s"] = round(time.time() - t1, 2)
    return report


def bytes_of(*arrays):
    return sum(a.nbytes for a in arrays)


def fmt(report, hbm_bytes=None, flops=None, hbm_bw=2.9e12 / 8, peak=79e12):
    """One-line summary; roofline per NeuronCore (1/8 of a trn2 chip's HBM BW)."""
    ns = report["predicted_ns"]
    s = f"predicted {ns/1e3:9.2f} us"
    if hbm_bytes:
        floor = hbm_bytes / hbm_bw * 1e9
        s += f" | HBM floor {floor/1e3:8.2f} us ({floor/ns*100:5.1f}% of BW roofline)"
    if flops:
        s += f" | {flops/ns/1e3:7.2f} TFLOPS ({flops/ns/1e3/(peak/1e12)*100:5.1f}% of peak)"
    util = report.get("utilization") or {}
    if util:
        s += " | util " + " ".join(f"{k}={v:.2f}" for k, v in sorted(util.items()))
    return s
