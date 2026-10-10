#!/usr/bin/env python3
"""
probe_nki.py -- one command that tells P2 what this pod's NKI can do. Run it in the seat pod and
paste the whole output back.

    cd /workspace/projects/03-chipboost && python tools/probe_nki.py 2>&1 | tee probe.txt

It checks, in order: versions, Qwen3-8B's config.json against shapes.py, the exact signatures of the
NKI calls our kernels need, which activation functions exist, whether tutorial sources ship inside the
installed package, and finally runs our matmul start kernel and copy kernel in the simulator in bf16.
Nothing here touches the chip, so it is safe while vLLM is running.
"""

import inspect
import os
import sys
import textwrap
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "02-kernel-agent"))


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


def main():
    section("versions")
    print(f"python {sys.version.split()[0]}")
    try:
        import numpy as np
        print(f"numpy {np.__version__}")
    except ImportError:
        print("numpy MISSING")
        return 1
    try:
        import ml_dtypes
        print(f"ml_dtypes {ml_dtypes.__version__}")
    except ImportError:
        print("ml_dtypes MISSING -- bf16 inputs cannot be built")
    try:
        import nki
        import nki.isa as nisa
        import nki.language as nl
        print(f"nki {getattr(nki, '__version__', '?')} at {os.path.dirname(nki.__file__)}")
    except ImportError as e:
        print(f"nki NOT importable: {e}. Run this inside the seat pod.")
        return 1
    for k in sorted(os.environ):
        if k.startswith(("NKI_", "NEURON_PLATFORM", "NEURON_RT_VISIBLE", "NEURON_CC")):
            print(f"env {k}={os.environ[k]}")

    section("Qwen3-8B config.json vs shapes.py")
    try:
        import shapes
        shapes.verify_config()
    except Exception as e:
        print(f"could not check: {type(e).__name__}: {e}")

    section("signatures of the calls our kernels need")
    mods = dict(nisa=nisa, nl=nl, nki=nki)
    wanted = ["nisa.activation", "nisa.activation_reduce", "nisa.tensor_reduce",
              "nisa.tensor_scalar", "nisa.tensor_tensor", "nisa.tensor_copy", "nisa.memset",
              "nisa.dma_copy", "nisa.dma_transpose", "nisa.nc_matmul", "nisa.nc_transpose",
              "nl.ndarray", "nl.zeros", "nl.affine_range", "nl.sequential_range",
              "nl.static_range", "nl.ds", "nl.sum", "nl.mean", "nl.load", "nl.store",
              "nl.broadcast_to", "nki.simulate", "nki.jit"]
    for name in wanted:
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
    print("  tile_size: " + ", ".join(f"{k}={getattr(nl.tile_size, k)}" for k in dir(nl.tile_size)
                                     if not k.startswith("_")))

    section("public names")
    for label_, mod in (("nl", nl), ("nisa", nisa)):
        names = ", ".join(n for n in sorted(dir(mod)) if not n.startswith("_"))
        print(textwrap.fill(f"{label_}: {names}", 110, subsequent_indent="    "))

    section("tutorial sources inside the installed package")
    hits = []
    for base, _, files in os.walk(os.path.dirname(nki.__file__)):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(base, f)
                try:
                    txt = open(p, errors="ignore").read()
                except OSError:
                    continue
                if "fully_optimized" in txt or "rmsnorm" in txt.lower():
                    hits.append(p)
    print("\n".join(f"  {h}" for h in hits[:10]) or "  none")

    section("bf16 smoke test in the simulator (our kernels)")
    try:
        import nkibench
        for op, path in (("matmul", "kernels/matmul_start.py"), ("copy", "kernels/copy_floor.py")):
            for case in shapes.cases(op, "dev"):
                kernel = nkibench.load_kernel(os.path.join(ROOT, path), shapes.entry(op))
                args = shapes.make_inputs(op, case)
                before = [a.copy() for a in args]
                want = shapes.reference(op, args)
                t0 = time.time()
                try:
                    got, counted = nkibench.simulate_and_count(kernel, args)
                except Exception as e:
                    print(f"  {path} {shapes.label(op, case)}: RAISED {type(e).__name__}: "
                          f"{str(e)[:300]}")
                    continue
                dt = time.time() - t0
                m = (nkibench.check_inputs_untouched(before, args)
                     or nkibench.describe_mismatch(got, want, shapes.tolerance(op)))
                verdict = "PASS" if not m else "FAIL: " + m.splitlines()[0][:150]
                print(f"  {path} {shapes.label(op, case)}: {verdict}  ({dt:.1f}s sim, "
                      f"{counted['bytes']:,} B in {counted['transfers']} transfers, "
                      f"out dtype {getattr(got, 'dtype', '?')})")
                for w in counted.get("warnings", []):
                    print(f"      warning: {w[:150]}")
    except Exception:
        traceback.print_exc(limit=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
