#!/usr/bin/env python3
"""
probe_nki.py -- one command that tells P2 what this pod's NKI can do, and whether our kernels work.
Run it in the seat pod and paste the output back.

    cd /workspace/chipboost/projects/03-chipboost
    python tools/probe_nki.py 2>&1 | tee probe.txt             # versions, config, API, then kernels
    python tools/probe_nki.py --sim-only                        # only run our kernels, simulator
    python tools/probe_nki.py --sim-only --which heldout        # ...on the held-out shapes
    python tools/probe_nki.py --sim-only --device               # ...and once each on the chip

It checks: versions, Qwen3-8B's config.json against shapes.py, the exact signatures of the NKI calls
our kernels need, which activation functions exist, and then runs every kernel in kernels/ in the
simulator in bf16, printing PASS/FAIL, the worst error (even on a pass, so two correct kernels can be
compared for precision), bytes moved against the floor, and transfers.

Nothing touches the chip unless you pass --device, which runs each kernel ONCE on NeuronCore 2
(NEURON_RT_VISIBLE_CORES=2), next to vLLM, which P1 measured holding cores 0-1 on the seat pods. That only checks the kernel compiles for and
computes right on the hardware; timing it properly is P1's referee's job.
"""

import argparse
import inspect
import os
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "02-kernel-agent"))

KERNELS = [("matmul", "kernels/matmul_start.py"), ("matmul", "kernels/matmul_expert.py"),
           ("matmul", "kernels/matmul_expert_aws.py"), ("copy", "kernels/copy_floor.py"),
           ("rmsnorm", "kernels/rmsnorm_start.py"), ("swiglu", "kernels/swiglu_start.py")]


def section(title):
    print(f"\n== {title}")


def first_doc_line(obj):
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0][:110] if doc.strip() else "(no docstring)"


def signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(signature not introspectable)"


def resolve(mods, dotted):
    head, _, rest = dotted.partition(".")
    obj = mods.get(head)
    for part in rest.split(".") if rest else []:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def worst_error(got, want):
    """max |got - want| over the reference's RMS: describe_mismatch's measure, printed even on a pass."""
    import numpy as np
    g = np.asarray(got, np.float64)
    w = np.asarray(want, np.float64)
    if g.shape != w.shape or not np.all(np.isfinite(g)):
        return float("nan")
    return float(np.abs(g - w).max() / (np.sqrt((w ** 2).mean()) or 1.0))


def selected(a):
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    for op, path in KERNELS:
        stem = os.path.splitext(os.path.basename(path))[0]
        if only and stem not in only:
            continue
        if not os.path.exists(os.path.join(ROOT, path)):
            print(f"  {path}: not written yet")
            continue
        yield op, path, stem


def run_kernels(a):
    import nkibench
    import shapes
    for which in [w.strip() for w in a.which.split(",") if w.strip()]:
        section(f"our kernels in the simulator, bf16, {which} shapes")
        for op, path, stem in selected(a):
            for case in shapes.cases(op, which):
                args = shapes.make_inputs(op, case)
                before = [x.copy() for x in args]
                want = shapes.reference(op, args)
                t0 = time.time()
                try:
                    kernel = nkibench.load_kernel(os.path.join(ROOT, path), shapes.entry(op))
                    got, counted = nkibench.simulate_and_count(kernel, args)
                except Exception as e:
                    print(f"  {stem:<18} {shapes.label(op, case):<52} RAISED {type(e).__name__}: "
                          f"{str(e)[:300]}")
                    continue
                dt = time.time() - t0
                m = (nkibench.check_inputs_untouched(before, args)
                     or nkibench.describe_mismatch(got, want, shapes.tolerance(op)))
                floor = shapes.work(op, case)[1]   # bf16 bytes; the float32 reference would inflate it
                print(f"  {stem:<18} {shapes.label(op, case):<52} {'PASS' if not m else 'FAIL'}  "
                      f"err {worst_error(got, want):.4f}  {dt:5.1f}s  "
                      f"{counted['bytes'] / max(floor, 1):.2f}x floor bytes, "
                      f"{counted['transfers']} transfers")
                if m:
                    print(textwrap.indent(m, "      ")[:700])
                for w in counted.get("warnings", []):
                    print(f"      warning: {w[:150]}")

    if not a.device:
        return
    section(f"our kernels ON THE CHIP, once each, first dev shape, "
            f"NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES')}")
    for op, path, stem in selected(a):
        case = shapes.cases(op, "dev")[0]
        args = shapes.make_inputs(op, case)
        want = shapes.reference(op, args)
        t0 = time.time()
        try:
            kernel = nkibench.load_kernel(os.path.join(ROOT, path), shapes.entry(op))
            got = kernel(*args)
        except Exception as e:
            print(f"  {stem:<18} {shapes.label(op, case):<52} RAISED {type(e).__name__}: "
                  f"{str(e)[:400]}")
            continue
        dt = time.time() - t0
        m = nkibench.describe_mismatch(got, want, shapes.tolerance(op))
        print(f"  {stem:<18} {shapes.label(op, case):<52} {'PASS' if not m else 'FAIL'}  "
              f"err {worst_error(got, want):.4f}  {dt:.1f}s incl. compile")
        if m:
            print(textwrap.indent(m, "      ")[:700])


def describe_environment():
    section("versions")
    print(f"python {sys.version.split()[0]}")
    import numpy as np
    print(f"numpy {np.__version__}")
    try:
        import ml_dtypes
        print(f"ml_dtypes {ml_dtypes.__version__}")
    except ImportError:
        print("ml_dtypes MISSING -- bf16 inputs cannot be built")
    import nki
    import nki.isa as nisa
    import nki.language as nl
    print(f"nki {getattr(nki, '__version__', '?')} at {os.path.dirname(nki.__file__)}")
    for k in sorted(os.environ):
        if k.startswith(("NKI_", "NEURON_PLATFORM", "NEURON_RT_VISIBLE", "NEURON_CC")):
            print(f"env {k}={os.environ[k]}")

    section("Qwen3-8B config.json vs shapes.py")
    import shapes
    shapes.verify_config()

    section("signatures of the calls our kernels need")
    mods = dict(nisa=nisa, nl=nl, nki=nki)
    for name in ["nisa.activation", "nisa.activation_reduce", "nisa.tensor_reduce",
                 "nisa.tensor_scalar", "nisa.tensor_tensor", "nisa.tensor_copy", "nisa.memset",
                 "nisa.dma_copy", "nisa.dma_transpose", "nisa.nc_matmul", "nisa.nc_transpose",
                 "nl.ndarray", "nl.zeros", "nl.affine_range", "nl.sequential_range",
                 "nl.static_range", "nl.ds", "nl.sum", "nl.mean", "nl.load", "nl.store",
                 "nl.broadcast_to", "nki.simulate", "nki.jit"]:
        obj = resolve(mods, name)
        if obj is None:
            print(f"  {name:<24} MISSING")
            continue
        print(f"  {name:<24} {signature(obj)}")
        print(f"  {'':<24} {first_doc_line(obj)}")

    section("activation and arithmetic identifiers in nki.language")
    ids = ["rsqrt", "sqrt", "reciprocal", "square", "exp", "silu", "sigmoid", "gelu", "copy",
           "identity", "multiply", "add", "subtract", "maximum", "divide", "power"]
    print("  present: " + ", ".join(i for i in ids if hasattr(nl, i)))
    print("  missing: " + (", ".join(i for i in ids if not hasattr(nl, i)) or "none"))
    # nl.tile_size values can only be read inside a running kernel (they ask the active backend).
    print("  tile_size names: " + ", ".join(k for k in dir(nl.tile_size) if not k.startswith("_")))

    section("public names")
    for label_, mod in (("nl", nl), ("nisa", nisa)):
        names = ", ".join(n for n in sorted(dir(mod)) if not n.startswith("_"))
        print(textwrap.fill(f"{label_}: {names}", 110, subsequent_indent="    "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-only", action="store_true",
                    help="skip versions and the API listing; only run our kernels")
    ap.add_argument("--which", default="dev", help="dev, timing, heldout, or a comma list")
    ap.add_argument("--only", default="", help="kernel file stems, e.g. matmul_expert,copy_floor")
    ap.add_argument("--device", action="store_true",
                    help="also run each kernel once on the chip, NeuronCore 2")
    a = ap.parse_args()
    if a.device:
        os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "2")   # vLLM holds 0-1; before nki starts a runtime
    try:
        import nki  # noqa: F401
    except ImportError as e:
        print(f"nki is not importable here ({e}): run this inside the seat pod.")
        return 2
    if not a.sim_only:
        describe_environment()
    run_kernels(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
