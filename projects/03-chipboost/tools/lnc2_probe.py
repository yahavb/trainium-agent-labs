#!/usr/bin/env python3
"""
lnc2_probe.py -- does the second physical NeuronCore pay? P2's "use 100% of the chip" experiment.

At LNC=2 each logical NeuronCore is two physical ones, and a plain launch uses one (NKI 0.6.0 docs). This
runs kernels/matmul_expert_lnc2.py, which splits the output between the two programs of a kernel[2]
launch, and measures it against the SAME kernel launched plainly (one program, the best single-core
block caps random search found), at the matmul timing shapes:

  1. simulator: kernel[2] in both documented forms, plus the plain launch, against the float32 reference;
  2. compile options: what the plain and the [2] kernel objects carry (LNC / grid fields), since P1's
     timing path compiles through kernel._to_subclass(StandaloneKernel) and has no LNC setting of its own;
  3. chip: compile LNC=1 and LNC=2 (kernel[2] through timing.compile_kernel, else the plain kernel with its
     compile options set to LNC=2), check correctness with the referee's own check, then time them
     interleaved on the device clock (timing.time_ab). Prints the speedup the second core buys.

Run it on a free logical core, nothing else on it:
    CHIPBOOST_CORE=3 python tools/lnc2_probe.py
    python tools/lnc2_probe.py --sim-only
"""

import argparse
import dataclasses
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "02-kernel-agent"))

import shapes  # noqa: E402
import nkibench  # noqa: E402

KERNEL = "kernels/matmul_expert_lnc2.py"
ENTRY = "nki_matmul_tiled_"
SPEC = shapes.OPS["matmul"]
WORDS = ("lnc", "logical", "grid", "spmd", "num_core", "cores")


def load():
    return nkibench.load_kernel(os.path.join(ROOT, KERNEL), ENTRY)


def simulate(k, shape):
    import nki
    inp = SPEC["make_inputs"](shape, 5)
    want = SPEC["ref"](inp)
    args = list(inp.values())
    tried = [("plain nki.simulate(k)(...)", lambda: nki.simulate(k)(*args)),
             ("nki.simulate(k[2])(...)", lambda: nki.simulate(k[2])(*args)),
             ("nki.simulate(k)[2](...)", lambda: nki.simulate(k)[2](*args))]
    for name, f in tried:
        t0 = time.time()
        try:
            got = f()
            m = nkibench.describe_mismatch(got, want, shapes.tolerance("matmul"))
            print(f"  {name:<28} {shape}: {'PASS' if not m else 'FAIL: ' + m.splitlines()[0][:110]} "
                  f"({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  {name:<28} {shape}: RAISED {type(e).__name__}: {str(e)[:220]}")


def opts_fields(k):
    from nki.framework.compiled import StandaloneKernel
    opts = k._to_subclass(StandaloneKernel)._compile_opts()
    return opts, {a: getattr(opts, a) for a in dir(opts)
                  if not a.startswith("_") and any(w in a.lower() for w in WORDS)}


def compile_opts_lnc(k, inputs, lnc):
    """P1's timing.compile_kernel, with the compile options' LNC field forced to `lnc`."""
    from nki.framework.compiled import compile_kernel_to_nir, StandaloneKernel
    from nki.compiler.driver import _compile_bir_to_neff
    sk = k._to_subclass(StandaloneKernel)
    opts = sk._compile_opts()
    field = next((a for a in ("lnc", "logical_nc_config", "logical_nc") if hasattr(opts, a)), None)
    if field is None:
        raise RuntimeError(f"no LNC field on {type(opts).__name__}: {sorted(a for a in dir(opts) if not a.startswith('_'))}")
    try:
        setattr(opts, field, lnc)
    except Exception:
        opts = dataclasses.replace(opts, **{field: lnc})
    nir = compile_kernel_to_nir(sk, inputs=inputs, compile_opts=opts,
                                frontend=sk._frontend_cls(enable_backend_opt=sk._enable_backend_opt),
                                enable_cache=False)
    return _compile_bir_to_neff(nir, opts, inputs)


def chip(rounds):
    import timing
    import speedcheck
    k = load()
    print(f"  kernel objects: plain {type(k).__name__}, [2] {type(k[2]).__name__}")
    for label, kk in (("plain", k), ("[2]", k[2])):
        try:
            print(f"  compile options, {label}: {opts_fields(kk)[1]}")
        except Exception as e:
            print(f"  compile options, {label}: RAISED {type(e).__name__}: {str(e)[:200]}")

    routes = [("kernel[2] via timing.compile_kernel", lambda inp: timing.compile_kernel(k[2], inp)),
              ("plain kernel, compile options set to LNC=2", lambda inp: compile_opts_lnc(k, inp, 2))]
    tot1 = tot2 = 0.0
    flops = 0
    route = None
    for shape in SPEC["time_shapes"]:
        inp = SPEC["make_inputs"](shape, 11)
        want = SPEC["ref"](inp)
        loaded = {}
        t0 = time.time()
        loaded[1] = timing.Loaded(timing.compile_kernel(k, inp), inp)
        print(f"\n  {shape}: LNC=1 compiled and loaded in {time.time() - t0:.1f}s")
        for name, comp in ([r for r in routes if r[0] == route] if route else routes):
            t0 = time.time()
            try:
                loaded[2] = timing.Loaded(comp(inp), inp)
                route = name
                print(f"  {shape}: LNC=2 via '{name}' compiled and loaded in {time.time() - t0:.1f}s")
                break
            except Exception as e:
                print(f"  {shape}: LNC=2 via '{name}' RAISED {type(e).__name__}: {str(e)[:300]}")
        if 2 not in loaded:
            print("  no route compiled an LNC=2 kernel; stopping.")
            return
        for n in (1, 2):
            got = loaded[n].run()[0]
            bad = speedcheck._mismatch(got, want, SPEC["tol"], f"LNC={n} {shape}")
            print(f"  {shape}: LNC={n} output {'correct' if not bad else 'WRONG: ' + bad.splitlines()[0][:150]}")
            if bad:
                return
        ab = timing.time_ab(loaded[1], loaded[2], rounds=rounds)
        t1, t2 = ab["a"]["median_us"], ab["b"]["median_us"]
        f = SPEC["flops"](shape)
        print(f"  {shape}: LNC=1 {t1:8.1f} us ({f / t1 / 1e6:5.1f} TFLOP/s)   LNC=2 {t2:8.1f} us "
              f"({f / t2 / 1e6:5.1f} TFLOP/s)   second core buys {t1 / t2:.3f}x")
        tot1, tot2, flops = tot1 + t1, tot2 + t2, flops + f
    print(f"\n  TOTAL over the timing shapes: LNC=1 {tot1:.1f} us, LNC=2 {tot2:.1f} us -> {tot1 / tot2:.3f}x "
          f"({flops / tot2 / 1e6:.1f} TFLOP/s on one logical core). Route: {route}.")
    print("  (the referee's start kernel took ~960 us for the same two shapes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    try:
        import nki  # noqa: F401
    except ImportError:
        print("nki is not installed here: run this in the seat pod.")
        return 2
    print("== simulator")
    k = load()
    for shape in ((512, 256, 1024), (256, 512, 512)):   # an N split and an M split
        simulate(k, shape)
    if a.sim_only:
        return 0
    print(f"\n== chip, NeuronCore {os.environ.get('CHIPBOOST_CORE', '(timing.py default)')}")
    try:
        chip(a.rounds)
    except Exception:
        traceback.print_exc(limit=4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
