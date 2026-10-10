"""Shared helper: compile an NKI kernel for trn2 (no device) and return the compiler's prediction."""
import os, shutil, tempfile, time
from nki.compiler.ncc_driver import CompileOptions
from nki.framework.compiled import compile_kernel_to_nir

def predict(kernel, inputs, lnc=1):
    d = tempfile.mkdtemp(prefix="_roof_", dir="/tmp")
    try:
        opts = CompileOptions(target="trn2", lnc=lnc, emit_predicted_latency=True,
                              enable_statistics=True, artifacts_dir=d)
        t = time.time()
        nir = compile_kernel_to_nir(kernel, inputs=inputs, compile_opts=opts, enable_cache=False)
        return dict(ns=nir.total_time_ns, util=nir.estimated_utilization,
                    macs=getattr(nir, "mac_count", None), compile_s=round(time.time() - t, 2))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def kernel_from_source(src, entry, tag):
    """The NKI parser frontend does not resolve closure variables, so variants are written as
    source with constants baked in and imported from a file."""
    import importlib.util
    path = f"/tmp/_roofk_{tag}.py"
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"_roofk_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)
