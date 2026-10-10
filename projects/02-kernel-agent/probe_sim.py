#!/usr/bin/env python3
"""
probe_sim.py -- ask the simulator what it actually is, before building on a guess about it.

The checker work in diagnose.py rests on three assumptions about `nki.simulate`, none of which
can be checked on a laptop: that a kernel's own lines appear in the traceback under the
candidate's file name, that replacing a function on `nki.language` / `nki.isa` is seen by a
running kernel, and that a tile knows its shape and its buffer. This prints the facts. Run it in
the seat pod; it needs no model and takes seconds.

    python probe_sim.py
"""

import importlib.util
import inspect
import traceback

import numpy as np

KERNEL = '''
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def probe_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray((64, 32), dtype=a.dtype, buffer=nl.sbuf)
    acc = nl.ndarray((64, 32), dtype=nl.float32, buffer=nl.psum)
    nisa.dma_copy(dst=tile, src=a[0:64, 0:32])
    nisa.dma_copy(dst=out[0:64, 0:32], src=tile)
    nisa.dma_copy(dst=tile, src=a[64:128, 0:32])
    nisa.dma_copy(dst=out[64:128, 0:32], src=tile)
    return out
'''

BROKEN = KERNEL.replace("tile = nl.ndarray((64, 32),", "tile = nl.ndarray((64, 16),")


def section(title):
    print(f"\n---- {title}")


def load(src, name):
    path = f"/tmp/_probe_{name}.py"
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"probe_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path, mod.probe_kernel


def describe(obj):
    names = [n for n in dir(obj) if not n.startswith("__")]
    out = [f"type={type(obj).__module__}.{type(obj).__name__}"]
    for attr in ("shape", "dtype", "buffer", "_buffer", "region", "memory", "mem", "location",
                 "name", "ndim", "nbytes"):
        if hasattr(obj, attr):
            try:
                out.append(f"{attr}={getattr(obj, attr)!r}"[:70])
            except Exception as e:
                out.append(f"{attr}=<{type(e).__name__}>")
    return "; ".join(out), names


def main():
    try:
        import nki
        import nki.isa as nisa
        import nki.language as nl
    except ImportError as e:
        raise SystemExit(f"no nki here ({e}). Run this in the seat pod.")

    section("the package")
    print(f"nki {getattr(nki, '__version__', '?')} at {nki.__file__}")
    try:
        from importlib import metadata
        d = metadata.distribution("nki")
        files = [str(f) for f in (d.files or [])]
        native = [f for f in files if f.endswith((".so", ".pyd", ".dylib"))]
        print(f"distribution {d.metadata['Name']} {d.version}; license {d.metadata.get('License')!r}; "
              f"home {d.metadata.get('Home-page')!r}; {len(files)} files, {len(native)} native")
        print(f"requires {d.requires}")
    except Exception as e:
        print(f"no distribution metadata: {type(e).__name__}: {e}")
    print(f"nki public names: {[n for n in dir(nki) if not n.startswith('_')]}")
    print(f"nki.isa has {len([n for n in dir(nisa) if not n.startswith('_')])} public names: "
          f"{[n for n in dir(nisa) if not n.startswith('_')]}")

    section("signatures")
    for label, fn in (("nl.ndarray", nl.ndarray), ("nisa.dma_copy", nisa.dma_copy),
                      ("nisa.nc_matmul", nisa.nc_matmul), ("nisa.tensor_copy", nisa.tensor_copy)):
        try:
            print(f"{label}{inspect.signature(fn)}")
        except Exception as e:
            print(f"{label}: no signature ({type(e).__name__}: {e})")

    section("is a replaced function seen by a running kernel, and what does it receive?")
    seen = dict(ndarray=[], dma_copy=[])
    real_ndarray, real_dma = nl.ndarray, nisa.dma_copy

    def spy_ndarray(*args, **kw):
        out = real_ndarray(*args, **kw)
        seen["ndarray"].append((args, kw, out))
        return out

    def spy_dma(*args, **kw):
        seen["dma_copy"].append((args, kw))
        return real_dma(*args, **kw)

    path, kernel = load(KERNEL, "ok")
    a = np.arange(128 * 32, dtype=np.float32).reshape(128, 32)
    nl.ndarray, nisa.dma_copy = spy_ndarray, spy_dma
    try:
        run = nki.simulate(kernel) if hasattr(nki, "simulate") else None
        out = run(a) if run else nki.simulate_kernel(kernel, a)
        print(f"kernel ran; returned {type(out).__module__}.{type(out).__name__}, "
              f"equal to its input: {np.array_equal(np.asarray(out), a)}")
    except Exception:
        print("the valid probe kernel raised:")
        traceback.print_exc()
    finally:
        nl.ndarray, nisa.dma_copy = real_ndarray, real_dma
    print(f"nl.ndarray calls seen: {len(seen['ndarray'])} of 3; nisa.dma_copy calls seen: "
          f"{len(seen['dma_copy'])} of 4")
    for args, kw, out in seen["ndarray"]:
        text, names = describe(out)
        print(f"  ndarray(args={args!r}, kw={ {k: repr(v)[:30] for k, v in kw.items()} }) -> {text}")
    if seen["ndarray"]:
        print(f"  attributes of an allocated tile: {describe(seen['ndarray'][0][2])[1]}")
    for args, kw in seen["dma_copy"][:2]:
        for k, v in list(enumerate(args)) + list(kw.items()):
            print(f"  dma_copy {k}: {describe(v)[0]}")
    if seen["dma_copy"]:
        src = seen["dma_copy"][0][1].get("src")
        if src is not None:
            print(f"  attributes of a slice of the input: {describe(src)[1]}")
    for name in ("sbuf", "psum", "shared_hbm"):
        r = getattr(nl, name, None)
        print(f"  nl.{name}: {type(r).__name__} repr={r!r} "
              f"attrs={[n for n in dir(r) if not n.startswith('_')][:8]}")

    section("where does a failure inside the kernel show up in the traceback?")
    path, kernel = load(BROKEN, "broken")
    try:
        (nki.simulate(kernel) if hasattr(nki, "simulate") else
         (lambda x: nki.simulate_kernel(kernel, x)))(a)
        print("the broken probe kernel did NOT raise")
    except Exception as e:
        print(f"raised {type(e).__name__}: {str(e)[:200]}")
        chain, cur = [], e
        while cur is not None and cur not in chain:
            chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        for depth, err in enumerate(chain):
            print(f"  exception {depth}: {type(err).__name__}")
            for fr in traceback.extract_tb(err.__traceback__):
                mark = "  <-- candidate file" if fr.filename == path else ""
                print(f"    {fr.filename}:{fr.lineno} in {fr.name}{mark}")
        import diagnose
        n = diagnose.locate(e, path, BROKEN)
        print(f"  diagnose.locate -> {n}: "
              f"{diagnose.statement_at(BROKEN, n) if n else 'NO LINE FOUND'}")
        want = next(i for i, l in enumerate(BROKEN.splitlines(), 1) if "dma_copy(dst=tile" in l)
        print(f"  the first failing copy is on line {want}")


if __name__ == "__main__":
    main()
