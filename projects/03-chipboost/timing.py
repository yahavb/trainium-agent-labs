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


class InterfaceError(RuntimeError):
    pass


class Neff:
    """A compiled kernel file, loaded and driven by the referee WITHOUT importing the candidate's code.

    An untrusted process compiles the candidate to a NEFF; this checks the NEFF's interface against what
    the op requires, feeds it the referee's own inputs and reads the output back. Every run starts from an
    output buffer refilled with fresh random garbage, so a kernel cannot skip work when it sees its output
    already holds an answer (it scored 26x that way when outputs were poisoned once and then reused)."""

    def __init__(self, neff_path, inputs, out_shape, out_dtype, seed=0):
        from nki.runtime import SpikeModel, SpikeTensor
        _pick_core()
        self.model = SpikeModel.load_from_neff(neff_path, core_id=0)
        if self.model.alias_info:
            raise InterfaceError(f"the output is aliased to an input ({self.model.alias_info}): the kernel writes "
                                 f"its result into its input. Allocate a new output in nl.shared_hbm.")
        info_in, info_out = self.model.input_tensors_info, self.model.output_tensors_info
        if set(info_in) != set(inputs):
            raise InterfaceError(f"kernel inputs are {sorted(info_in)}, expected {sorted(inputs)}")
        for n, a in inputs.items():
            if tuple(info_in[n].shape) != a.shape or info_in[n].size != a.nbytes:
                raise InterfaceError(f"input {n} is {tuple(info_in[n].shape)} ({info_in[n].size} bytes), "
                                     f"expected {a.shape} ({a.nbytes} bytes)")
        if len(info_out) != 1:
            raise InterfaceError(f"kernel returns {len(info_out)} outputs, expected 1")
        self.oname, o = next(iter(info_out.items()))
        self.dtype = np.dtype(out_dtype)
        if tuple(o.shape) != tuple(out_shape) or o.size != int(np.prod(out_shape)) * self.dtype.itemsize:
            raise InterfaceError(f"output is {tuple(o.shape)} ({o.size} bytes), expected {tuple(out_shape)} "
                                 f"{self.dtype}")
        self.shape = tuple(out_shape)
        self.inputs = {n: SpikeTensor.from_numpy(a, name=n, core_id=0) for n, a in inputs.items()}
        self.out = SpikeTensor.from_numpy(np.zeros(self.shape, self.dtype), name=self.oname, core_id=0)
        self._rng = np.random.default_rng(seed)

    def set_inputs(self, inputs):
        for n, a in inputs.items():
            self.inputs[n].write_from_numpy(a)

    def read_inputs(self):
        return {n: t.numpy() for n, t in self.inputs.items()}

    def _poison(self):
        self.out.write_from_numpy((self._rng.standard_normal(self.shape) * 1e3).astype(self.dtype))

    def run(self):
        self._poison()
        self.model(self.inputs, outputs={self.oname: self.out})
        return self.out.numpy()

    def time_fresh(self, n, verify=None, feed=None):
        """n single device-timed runs, each from a freshly poisoned output. `feed(j)` -> (key, inputs) swaps in
        a different input set before run j (host write, not timed); `verify(out, key)` checks every output."""
        us = []
        for j in range(n):
            key = None
            if feed is not None:
                key, inp = feed(j)
                self.set_inputs(inp)
            self._poison()
            b = self.model.benchmark(self.inputs, outputs={self.oname: self.out},
                                     warmup_iter=0, benchmark_iter=1, mode="device")
            us += [d * 1000.0 for d in b.durations_ms]
            if verify is not None:
                verify(self.out.numpy(), key)
        return us


def time_ab_fresh(a, b, rounds=3, n=10, verify_a=None, verify_b=None, feed=None):
    """Interleave baseline a and candidate b (a b a b ...). Every run starts from fresh output garbage; with
    `feed`, both arms see the same sequence of different input sets, so a kernel cannot return a result it
    computed on an earlier call; every run is verified."""
    a.run()
    b.run()                                   # one untimed run each: warm instruction caches
    sa, sb = [], []
    for r in range(rounds):
        f = (lambda j, r=r: feed(r * n + j)) if feed is not None else None
        sa += a.time_fresh(n, verify_a, f)
        sb += b.time_fresh(n, verify_b, f)
    A, B = _stats(sa), _stats(sb)
    return dict(a=A, b=B, speedup=A["median_us"] / B["median_us"])


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


NOISE_FLOOR = 0.01


def noise_threshold(*stats):
    """The ONE definition of the referee's noise threshold: 1 + max(1%, 2x the relative IQR), where the IQR is
    the worst among `stats` -- pass both arms of every timed shape, so each shape counts its noisier arm and the
    shapes pool conservatively (the max). A speedup >= thr is a gain, <= 1/thr a loss, in between `no_gain`.

    Why 1%: on seat-100 the A/A spread (start kernel vs itself, 45 checks) was sd 0.011% and the worst per-shape
    deviation 0.063%, so 1% is ~16x the worst seen; the old 5% floor was ~170x the noise and called real 2-4%
    speedups `slower`. Noisy (small) shapes still raise the threshold through the IQR term."""
    rel = max(s["iqr_us"] / s["median_us"] for s in stats)
    return 1.0 + max(NOISE_FLOOR, 2.0 * rel)


# ---------------------------------------------------------------- selftest

def selftest(shapes="small"):
    # The referee's own loader (nkibench) and matmul inputs/reference, so the timer is proven on exactly what the
    # referee feeds it. Imported here, not at module level: speedcheck imports this module (lazily) too.
    import speedcheck
    spec = speedcheck.OPS["matmul"]
    ref = os.path.join(speedcheck.NKIBENCH_DIR, "reference_level4.py")
    kern = speedcheck.nkibench.load_kernel(ref, "nki_matmul_tiled_")
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
        inp = spec["make_inputs"]((K, M, N), 0)
        t0 = time.time()
        ck = compile_kernel(kern, inp)
        ct = time.time() - t0
        L = Loaded(ck, inp)
        out = L.run()[0].astype(np.float32)
        want = np.asarray(spec["ref"](inp), np.float32)
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
          f"same kernel vs itself: {ab['speedup']:.3f}  (noise threshold {noise_threshold(ab['a'], ab['b']):.3f})")

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
