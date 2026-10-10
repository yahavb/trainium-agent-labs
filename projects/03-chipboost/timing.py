#!/usr/bin/env python3
"""
timing.py -- compile an @nki.jit kernel once, run it on a free NeuronCore, and time it on the device.

NKI 0.6.0 (SDK 2.32) has no public benchmark API: calling a kernel from Python recompiles and times the
host, which on a 25 us matmul reads ~75 us. This uses the path the SDK itself uses internally:

    compile_kernel_to_nir -> _compile_bir_to_neff  (NEFF, ~2 s)
    SpikeModel.benchmark(mode="device")            (NeuronCore clock, nc_exec_running trace)

so compile time and host overhead are excluded. These are internal APIs and may change between SDKs;
`python timing.py --selftest` checks they still behave before anything else trusts them.

Cores. A NeuronCore cannot be shared between processes. On the seat pods vLLM (tensor parallel 2) holds
logical cores 0-1, so this defaults to core 2. Override with CHIPBOOST_CORE, or let it try 2 then 3.

    python timing.py --selftest            # timer sanity: correct output, scaling, noise
    python timing.py --selftest --shapes qwen
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

import numpy as np

_CORE = None
_DEFAULT_CORES = (2, 3)


def _pick_core():
    """Bind this process to one free logical NeuronCore. Must run before the runtime starts."""
    global _CORE
    if _CORE is not None:
        return _CORE
    wanted = os.environ.get("CHIPBOOST_CORE")
    candidates = [int(wanted)] if wanted else list(_DEFAULT_CORES)
    from nki.runtime import configure, reset, get_spike_singleton
    last = None
    for c in candidates:
        try:
            configure(visible_cores=[c])
            get_spike_singleton()          # nrt_init: fails with "cores busy" if taken
            _CORE = c
            return c
        except Exception as e:             # busy core: reset and try the next
            last = e
            try:
                reset()
            except Exception:
                pass
    raise RuntimeError(f"no free NeuronCore among {candidates} (vLLM holds 0-1 on seat pods): {last}")


def load_kernel(path, entry):
    """Import a kernel file and return its @nki.jit entry point."""
    spec = importlib.util.spec_from_file_location(f"cand_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, entry):
        raise AttributeError(f"no function named {entry} in {path}")
    return getattr(mod, entry)


def compile_kernel(kernel, inputs):
    """Compile an @nki.jit kernel for these exact input shapes/dtypes. Returns a CompiledKernel."""
    from nki.framework.compiled import compile_kernel_to_nir, StandaloneKernel
    from nki.compiler.driver import _compile_bir_to_neff
    sk = kernel._to_subclass(StandaloneKernel)
    opts = sk._compile_opts()
    nir = compile_kernel_to_nir(sk, inputs=inputs, compile_opts=opts,
                                frontend=sk._frontend_cls(enable_backend_opt=sk._enable_backend_opt),
                                enable_cache=False)
    return _compile_bir_to_neff(nir, opts, inputs)


class Loaded:
    """A compiled kernel resident on the device, with its inputs already copied over."""

    def __init__(self, ck, inputs):
        from nki.runtime import SpikeTensor
        _pick_core()
        self.ck = ck
        self.model = ck._ensure_loaded(0, 1)    # rank 0 = the one visible core
        ins = ck.prepare_inputs({k: v for k, v in inputs.items() if isinstance(v, np.ndarray)})
        self.inputs = {k: SpikeTensor.from_numpy(v, name=k, core_id=0) for k, v in ins.items()}
        self.outputs = {k: SpikeTensor.from_numpy(v, name=k, core_id=0)
                        for k, v in ck.prepare_outputs().items()}

    def run(self):
        """Execute once on the device and return the outputs as numpy arrays."""
        for t in self.outputs.values():              # poison, so a kernel that writes nothing fails
            fill = 0 if np.dtype(t.dtype).kind in "iub" else np.nan
            t.write_from_numpy(np.full(t.shape, fill, dtype=t.dtype))
        self.model(self.inputs, outputs=self.outputs)
        return [t.numpy() for t in self.outputs.values()]

    def time(self, warmup=3, iters=20):
        """Device-side execution time. Returns microsecond stats over `iters` runs."""
        b = self.model.benchmark(self.inputs, outputs=self.outputs,
                                 warmup_iter=warmup, benchmark_iter=iters, mode="device")
        return _stats([d * 1000.0 for d in b.durations_ms])


def _stats(us):
    s = sorted(us)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return dict(median_us=statistics.median(s), iqr_us=q(0.75) - q(0.25),
                min_us=s[0], max_us=s[-1], n=len(s), samples_us=us)


def time_ab(a, b, rounds=5, warmup=3, iters=20):
    """Interleave A and B (A B A B ...) so drift hits both equally. Returns per-arm pooled stats
    and the speedup a/b computed from the pooled medians."""
    sa, sb = [], []
    for _ in range(rounds):
        sa += a.time(warmup, iters)["samples_us"]
        sb += b.time(warmup, iters)["samples_us"]
    A, B = _stats(sa), _stats(sb)
    return dict(a=A, b=B, speedup=A["median_us"] / B["median_us"])


def noise_threshold(stats):
    """Smallest speedup that beats the measured noise of a baseline: max(5%, 2x relative IQR)."""
    return 1.0 + max(0.05, 2.0 * stats["iqr_us"] / stats["median_us"])


# ---------------------------------------------------------------- selftest

def _matmul_inputs(K, M, N, seed=0):
    import ml_dtypes
    r = np.random.default_rng(seed)
    return {"lhsT": r.standard_normal((K, M)).astype(ml_dtypes.bfloat16),
            "rhs": r.standard_normal((K, N)).astype(ml_dtypes.bfloat16)}


def selftest(shapes="small"):
    here = os.path.dirname(os.path.abspath(__file__))
    ref = os.path.join(here, "..", "02-kernel-agent", "reference_level4.py")
    kern = load_kernel(ref, "nki_matmul_tiled_")
    core = _pick_core()
    print(f"bound to NeuronCore {core}")
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    if shapes == "qwen":
        # Qwen3-8B per-core shapes under tensor parallel 2, 256 prompt tokens: q_proj 4096->2048, gate/up 4096->6144
        cases = [(4096, 256, 2048), (4096, 256, 6144)]
    else:
        cases = [(512, 256, 1024), (512, 256, 2048)]   # second does exactly 2x the work of the first

    timed = []
    for K, M, N in cases:
        inp = _matmul_inputs(K, M, N)
        t0 = time.time()
        ck = compile_kernel(kern, inp)
        ct = time.time() - t0
        L = Loaded(ck, inp)
        out = L.run()[0].astype(np.float32)
        want = inp["lhsT"].astype(np.float32).T @ inp["rhs"].astype(np.float32)
        err = float(np.nanmax(np.abs(out - want)) / np.abs(want).max())
        print(f"\nmatmul K={K} M={M} N={N}  (compile {ct:.1f}s)")
        check("output written", not np.isnan(out).any(), f"{np.isnan(out).sum()} NaN left of {out.size}")
        check("output correct", err < 2e-2, f"max rel err {err:.2e} vs fp32 reference")
        s = L.time()
        rel = s["iqr_us"] / s["median_us"]
        check("noise under 5%", rel < 0.05, f"median {s['median_us']:.1f} us, IQR {100*rel:.1f}%")
        print(f"        {2*K*M*N / (s['median_us']*1e-6) / 1e12:.1f} TFLOP/s")
        timed.append((K * M * N, s["median_us"], L))

    (w1, t1, L1), (w2, t2, L2) = timed[0], timed[-1]
    ratio = t2 / t1
    expect = w2 / w1
    check("time scales with work", 0.6 * expect < ratio < 1.4 * expect,
          f"{expect:.1f}x the work took {ratio:.2f}x the time")

    ab = time_ab(L1, L1, rounds=3)
    check("A/A interleave is ~1.0", abs(ab["speedup"] - 1) < 0.03,
          f"same kernel vs itself: {ab['speedup']:.3f}  (noise threshold {noise_threshold(ab['a']):.3f})")

    print("\nTIMER OK" if ok else "\nTIMER FAILED -- do not trust timings")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shapes", choices=("small", "qwen"), default="small")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.shapes) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
