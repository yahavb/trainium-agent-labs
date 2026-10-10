#!/usr/bin/env python3
"""
verify_sdk.py -- check my_attention.py against the Neuron SDK actually installed in this pod.

my_attention.py was written without the SDK, and the lines marked VERIFY use instructions or
keywords none of the shipped reference kernels call. Passing nkibench shows the numbers came out right; this checks
the things a pass does not:

  1. WHICH SDK      versions of nki / neuronx-cc / torch-neuronx, for the write-up
  2. THE API        every nisa.* / nl.* name the kernel uses exists, and every keyword it passes
                    is in that function's real signature -- read from the SDK, not from docs
  2b. THE V1 SCHEDULE  the three SDK features the V1 schedule leans on (activation_reduce,
                    tensor_reduce negate=, nl.copy), and what to write instead if one is missing
  3. THE NUMBERS    nki.simulate against a float64 reference on the level-8 shapes, ragged
                    sequence lengths, and hostile values (overflow, identical rows)
  4. HARDWARE HAZARDS  simulator warnings that say a pattern is wrong ON THE DEVICE even when
                    the CPU result matches -- nkibench only fails these inside the agent
  5. COMPILES       the real compiler, to a NEFF, at the runtime's LNC. The simulator accepts
                    things the device compiler rejects (nl.divide in tensor_scalar did exactly that)

    python verify_sdk.py                         # my_attention.py
    python verify_sdk.py my_attention_v0.py      # any other kernel file defining nki_attention_

Exit code 0 only if every check passes. Paste the whole output into the write-up.
"""

import ast
import importlib
import inspect
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KERNEL_FILE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "my_attention.py")
TOL = 2e-2          # nkibench's grading tolerance: worst error as a fraction of the output RMS

failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)


def section(title):
    print(f"\n{title}")


def versions():
    section("1. WHICH SDK")
    for mod in ("nki", "neuronxcc", "torch_neuronx", "torch", "numpy"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<14} {getattr(m, '__version__', 'installed, no __version__')}")
        except Exception:
            print(f"  {mod:<14} not installed")
    import nki
    check(hasattr(nki, "simulate") or hasattr(nki, "simulate_kernel"),
          "nki exposes a CPU simulator")


def api_calls():
    """Every call in the kernel of the form nisa.f(...) / nl.f(...), with its keyword names."""
    calls = []
    for node in ast.walk(ast.parse(open(KERNEL_FILE).read())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("nisa", "nl"):
            calls.append((node.lineno, node.func.value.id, node.func.attr,
                          [kw.arg for kw in node.keywords if kw.arg]))
    return calls


def api():
    section("2. THE API -- names and keywords checked against the installed SDK")
    import nki.isa as nisa
    import nki.language as nl
    mods = {"nisa": nisa, "nl": nl}
    seen = set()
    for line, mod, name, kws in api_calls():
        key = (mod, name, tuple(kws))
        if key in seen:
            continue
        seen.add(key)
        fn = getattr(mods[mod], name, None)
        if fn is None:
            check(False, f"line {line}: {mod}.{name} does not exist in this SDK")
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            print(f"  ??    line {line}: {mod}.{name} exists; signature not introspectable")
            continue
        takes_any = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
        unknown = [k for k in kws if k not in sig.parameters and not takes_any]
        check(not unknown, f"line {line}: {mod}.{name}({', '.join(k + '=' for k in kws)})"
              + (f"  -- unknown keyword(s) {unknown}; real signature {name}{sig}" if unknown else ""))
    for name in ("exp", "maximum", "add", "multiply", "divide", "copy"):
        check(hasattr(nl, name), f"nl.{name} exists (used as an op)")


def accepts(fn, *keywords):
    """True if fn's signature takes every keyword, False if not (or fn is missing), None if the
    signature cannot be read."""
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return all(k in sig.parameters for k in keywords)


def schedule_features():
    """Section 2 already fails the run if the kernel passes a name or keyword this SDK lacks. This
    says, for each feature the V1 schedule depends on, which spelling the SDK has and what to write
    if it has none -- so a FAIL above turns into a one-line edit rather than a search of the docs."""
    section("2b. THE V1 SCHEDULE -- the SDK features it leans on, and the fallback for each")
    import nki.isa as nisa
    import nki.language as nl

    features = [
        ("row sum accumulated by the exp instruction (my_attention.py uses the first spelling)",
         [("nisa.activation_reduce(..., bias=, reduce_op=, reduce_res=)",
           accepts(getattr(nisa, "activation_reduce", None), "bias", "reduce_op", "reduce_res")),
          ("nisa.activation(..., reduce_op=, reduce_res=)",
           accepts(getattr(nisa, "activation", None), "reduce_op", "reduce_res"))],
         "go back to nisa.activation for exp plus nisa.tensor_reduce(dst=row_sum, op=nl.add, "
         "data=p, axis=1), but issue that tensor_reduce AFTER the pT tensor_copy, so it runs while "
         "the Tensor engine does P @ V instead of delaying it"),
        ("-max straight out of the max reduction",
         [("nisa.tensor_reduce(..., negate=)", accepts(getattr(nisa, "tensor_reduce", None), "negate"))],
         "reduce into row_max, then nisa.tensor_scalar(dst=neg_max, data=row_max, "
         "op0=nl.multiply, operand0=-1.0)"),
        ("qT leaves PSUM through the Scalar engine, scaled on the way",
         [("nl.copy as an activation op", hasattr(nl, "copy"))],
         "nisa.tensor_scalar(dst=qT, data=qT_ps, op0=nl.multiply, operand0=scale) -- still correct, "
         "but back on the Vector engine, so the kT copy queues behind it again"),
    ]
    for what, spellings, fallback in features:
        have = [name for name, ok in spellings if ok]
        unknown = [name for name, ok in spellings if ok is None]
        tag = "ok  " if spellings[0][1] else ("??  " if spellings[0][1] is None else "MISS")
        print(f"  {tag}  {what}")
        for name, ok in spellings:
            print(f"          {'has    ' if ok else ('unknown' if ok is None else 'lacks  ')}  {name}")
        if not spellings[0][1]:
            print(f"        instead: {'switch to ' + have[0] if have else fallback}"
                  + (f"  (could not read the signature of: {', '.join(unknown)})" if unknown else ""))

    # Not used yet: the next candidate change loads Q and K already transposed. Report what exists.
    dt = getattr(nisa, "dma_transpose", None)
    try:
        sig = f"dma_transpose{inspect.signature(dt)}" if dt else "not in this SDK"
    except (TypeError, ValueError):
        sig = "dma_transpose exists; signature not introspectable"
    print(f"  info  for the next step (Q, K loaded pre-transposed): {sig}")


def reference64(q, k, v):
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    s = q @ k.T / np.sqrt(q.shape[1])
    s -= s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=-1, keepdims=True)) @ v


def numbers():
    section(f"3. THE NUMBERS -- nki.simulate vs float64, pass at error <= {TOL:g} of output RMS")
    import nkibench
    kernel = nkibench.load_kernel(KERNEL_FILE, "nki_attention_")
    cases = [  # (label, seq, dim, scale)
        ("level-8", 128, 64, 1.0), ("level-8", 64, 128, 1.0), ("level-8", 96, 32, 1.0),
        ("ragged seq", 1, 64, 1.0), ("ragged seq", 37, 64, 1.0), ("ragged seq", 127, 128, 1.0),
        ("large values", 128, 64, 30.0),        # scores ~900: a naive exp() overflows
        ("identical rows", 128, 64, 1.0),       # softmax must come out exactly uniform
    ]
    hazards = set()
    for label, seq, dim, scale in cases:
        r = np.random.default_rng(seq * 1000 + dim)
        q, k, v = (r.standard_normal((seq, dim)).astype(np.float32) * scale for _ in range(3))
        if label == "identical rows":
            k[:] = k[0]
        before = [a.copy() for a in (q, k, v)]
        what = f"{label:<15} seq={seq:<4} dim={dim:<4}"
        try:
            out, counted = nkibench.simulate_and_count(kernel, (q, k, v))
        except Exception as e:
            check(False, f"{what} RAISED {type(e).__name__}: {str(e)[:160]}")
            continue
        want = reference64(q, k, v)
        out = np.asarray(out, np.float64)
        err = (float(np.abs(out - want).max() / (np.sqrt((want ** 2).mean()) or 1.0))
               if np.all(np.isfinite(out)) else float("inf"))
        floor = nkibench.minimum_hbm_bytes((q, k, v), want.astype(np.float32))
        untouched = all(np.array_equal(b, a) for b, a in zip(before, (q, k, v)))
        check(err <= TOL and untouched,
              f"{what} error {err:.2e}  HBM {counted['bytes']:,} B ({counted['bytes'] / floor:.2f}x floor)"
              + ("" if untouched else "  -- MODIFIED ITS INPUT"))
        hazards.update(w for w in counted.get("warnings", []) if "hardware" in w.lower())

    section("4. HARDWARE HAZARDS reported by the simulator")
    check(not hazards, "no warning says the kernel would be wrong on the device"
          + ("" if not hazards else ": " + " | ".join(sorted(hazards))[:300]))


def device_compile():
    """The simulator is more permissive than the device compiler: V0 divided with
    tensor_scalar(op0=nl.divide), passed every check above, and then failed to compile on seat-65
    with "unsupported operator 'divide'". So compile for real -- the same kernel -> NIR -> NEFF path a
    device run takes -- at the runtime's LNC. Nothing is executed and no NeuronCore is needed."""
    import tempfile
    from dataclasses import replace

    import nkibench
    from nki.compiler.frontend import resolve_frontend_cls
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel, compile_kernel_to_nir

    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    section(f"5. COMPILES FOR THE DEVICE -- neuronx-cc to a NEFF at LNC {lnc}, not run")
    kernel = nkibench.load_kernel(KERNEL_FILE, "nki_attention_")
    fe = resolve_frontend_cls()
    ck = (kernel[lnc] if lnc != 1 else kernel)._to_subclass(CompileKernel, _frontend_cls=fe)
    for seq, dim in ((128, 64), (64, 128), (96, 32)):
        args = tuple(np.zeros((seq, dim), np.float32) for _ in range(3))
        with tempfile.TemporaryDirectory(prefix="verify_neff_") as work:
            try:
                opts = replace(ck._compile_opts(), artifacts_dir=work,
                               output_path=os.path.join(work, "kernel.neff"))
                nir = compile_kernel_to_nir(ck, inputs=ck._bind_args(args, {}), compile_opts=opts,
                                            frontend=fe(enable_backend_opt=False), enable_cache=False)
                compiled = compile_bir_to_neff(
                    opts, nir, input_arrays=[],
                    argument_names=[s.name for s in nir.descriptor.input_specs],
                    output_arg_names=[s.name for s in nir.descriptor.output_specs])
                check(os.path.exists(compiled.neff_path), f"seq={seq:<4} dim={dim:<4} compiles to a NEFF")
            except Exception as e:
                lines = [t.strip() for t in str(e).splitlines() if "error" in t.lower()] or [str(e)]
                check(False, f"seq={seq:<4} dim={dim:<4} does NOT compile: {lines[-1][:200]}")


def main():
    try:
        import nki  # noqa: F401
    except ImportError:
        sys.exit("The Neuron SDK is not installed here. Run this inside the seat pod.")
    print(f"kernel: {KERNEL_FILE}")
    versions()
    api()
    schedule_features()
    numbers()
    device_compile()
    print("\n" + ("VERIFIED: every check passed. The numbers are SIMULATOR results and the NEFF was "
                  "compiled but not run -- run bench_device.py for the device."
                  if not failures else f"NOT VERIFIED: {len(failures)} check(s) failed, listed above."))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
