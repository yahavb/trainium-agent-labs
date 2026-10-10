#!/usr/bin/env python3
"""
probe_nki.py -- find out how this pod's nki runs a kernel on the device.

bench_device.py was written for older nki releases, which ran a kernel on the device through
nki.baremetal and timed it through nki.benchmark. nki 0.6.0 has neither; its only entry point is
nki.simulate. This prints what 0.6.0 does have -- the jit decorator, the object it returns, the
package's modules, where its source mentions NEFFs, devices or frameworks -- and then tries the
one device path that does not depend on nki's own API: calling the kernel with torch-xla tensors.
It writes nothing outside a temp directory.

    python probe_nki.py > probe.log 2>&1; cat probe.log
"""

import os

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")     # vLLM holds NC 2-3
os.environ.setdefault("PJRT_DEVICE", "NEURON")            # without it torch-xla silently uses the CPU

import importlib
import inspect
import pkgutil
import re
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def section(title):
    print(f"\n========== {title}")


def sig(fn):
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(signature not introspectable)"


def first_doc_line(obj):
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0][:110] if doc.strip() else ""


def public_api(obj, label):
    for name in sorted(n for n in dir(obj) if not n.startswith("_")):
        try:
            member = getattr(obj, name)
        except Exception as e:
            print(f"  {label}.{name}: <{type(e).__name__} on access>")
            continue
        kind = ("module" if inspect.ismodule(member) else "class" if inspect.isclass(member)
                else "callable" if callable(member) else type(member).__name__)
        extra = sig(member) if kind in ("callable", "class") else ""
        doc = first_doc_line(member) if kind != "module" else ""
        print(f"  {label}.{name}  [{kind}] {extra}" + (f"\n      {doc}" if doc else ""))


def run(title, fn):
    section(title)
    try:
        fn()
    except Exception:
        print("  RAISED:\n" + "\n".join("    " + t for t in traceback.format_exc().splitlines()[-12:]))


def top_level():
    import nki
    print(f"  nki {getattr(nki, '__version__', '?')} at {os.path.dirname(nki.__file__)}")
    public_api(nki, "nki")


def jit_decorator():
    import nki
    print(f"  nki.jit{sig(nki.jit)}")
    doc = inspect.getdoc(nki.jit) or "(no docstring)"
    print("\n".join("    " + t for t in doc.splitlines()[:80]))


def jitted_kernel():
    import nkibench
    kernel = nkibench.load_kernel(os.path.join(HERE, "my_attention_v0.py"), "nki_attention_")
    print(f"  type {type(kernel).__module__}.{type(kernel).__name__}")
    print(f"  mro  {[c.__name__ for c in type(kernel).__mro__]}")
    public_api(kernel, "kernel")
    doc = inspect.getdoc(type(kernel)) or ""
    if doc:
        print("  class docstring:\n" + "\n".join("    " + t for t in doc.splitlines()[:40]))


def submodules():
    import nki
    names = [m.name for m in pkgutil.walk_packages(nki.__path__, "nki.", onerror=lambda _: None)]
    print(f"  {len(names)} modules")
    for n in names[:250]:
        print(f"  {n}")


PATTERNS = [
    ("entry points", r"^\s*def (baremetal|benchmark|profile|run|execute|launch|compile\w*|to_neff|"
                     r"simulate\w*|load\w*|save\w*)\b"),
    ("modes / targets", r"\bmode\s*[=:]|platform_target|target\s*="),
    ("NEFF / artifacts", r"neff|artifact|ntff|working_dir"),
    ("frameworks", r"torch_xla|torchxla|torch_neuronx|\bjax\b|pjrt"),
    ("runtime", r"NEURON_RT|nrt_|libnrt|neuron-profile|neuron_profile"),
]


def source_search():
    import nki
    files = []
    for root in nki.__path__:
        for d, _, fs in os.walk(root):
            files += [os.path.join(d, f) for f in fs if f.endswith((".py", ".pyi"))]
    print(f"  {len(files)} source files (compiled extensions are not searched)")
    base = os.path.dirname(nki.__path__[0])
    for label, pat in PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        hits = []
        for path in files:
            try:
                for i, line in enumerate(open(path, errors="replace"), 1):
                    if rx.search(line):
                        hits.append(f"{os.path.relpath(path, base)}:{i}: {line.strip()[:120]}")
            except OSError:
                pass
        print(f"\n  -- {label}: {len(hits)} lines" + (" (first 25)" if len(hits) > 25 else ""))
        for h in hits[:25]:
            print(f"    {h}")


def frameworks():
    for mod in ("neuronxcc", "neuronxcc.nki", "torch", "torch_xla", "torch_neuronx", "jax",
                "jax_neuronx", "libneuronxla"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<16} {getattr(m, '__version__', 'installed')}")
        except Exception as e:
            print(f"  {mod:<16} not importable ({type(e).__name__})")
    try:
        old = importlib.import_module("neuronxcc.nki")
        print("  the OLD API in neuronxcc.nki has: " + ", ".join(
            n for n in ("baremetal", "benchmark", "profile", "simulate_kernel", "jit") if hasattr(old, n))
              + "  (it predates dst= keywords, so it may not accept these kernels)")
    except Exception:
        pass


def torch_xla_smoke():
    """V0 is the kernel the simulator already passed, so a failure here is about the device path,
    not the kernel."""
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import nkibench
    print(f"  torch-xla device type: {xr.device_type()}")
    if str(xr.device_type()).upper() != "NEURON":
        print("  NOT a NeuronCore -- refusing to call anything below a device result.")
        return
    dev = xm.xla_device()
    kernel = nkibench.load_kernel(os.path.join(HERE, "my_attention_v0.py"), "nki_attention_")
    r = np.random.default_rng(0)
    q, k, v = (r.standard_normal((96, 32)).astype(np.float32) for _ in range(3))
    qx, kx, vx = (torch.from_numpy(a).to(dev) for a in (q, k, v))
    t0 = time.perf_counter()
    out = kernel(qx, kx, vx)
    xm.mark_step()
    xm.wait_device_ops()
    got = (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy().astype(np.float64)
    qd, kd, vd = (a.astype(np.float64) for a in (q, k, v))
    s = qd @ kd.T / np.sqrt(32)
    e = np.exp(s - s.max(1, keepdims=True))
    want = (e / e.sum(1, keepdims=True)) @ vd
    err = np.abs(got - want).max() / np.sqrt((want ** 2).mean())
    print(f"  V0 ran through torch-xla in {time.perf_counter() - t0:.1f} s (includes compile); "
          f"error {err:.2e} of output RMS -- {'CORRECT' if err <= 2e-2 else 'WRONG'} on the device")
    for var in ("NEURON_CC_FLAGS", "NEURON_COMPILE_CACHE_URL", "XLA_FLAGS"):
        print(f"  {var}={os.environ.get(var, '(unset)')}")
    for cache in ("/var/tmp/neuron-compile-cache", os.path.expanduser("~/.cache/neuron")):
        if os.path.isdir(cache):
            neffs = [os.path.join(d, f) for d, _, fs in os.walk(cache) for f in fs if f.endswith(".neff")]
            neffs.sort(key=os.path.getmtime)
            print(f"  {cache}: {len(neffs)} NEFFs, newest: {neffs[-3:] if neffs else '-'}")


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}   NEURON_RT_VISIBLE_CORES={os.environ['NEURON_RT_VISIBLE_CORES']}")
    run("1. nki, top level", top_level)
    run("2. nki.jit", jit_decorator)
    run("3. what @nki.jit returns", jitted_kernel)
    run("4. nki's modules", submodules)
    run("5. nki's source: entry points, modes, NEFFs, frameworks, runtime", source_search)
    run("6. frameworks installed", frameworks)
    run("7. V0 on the device through torch-xla", torch_xla_smoke)
