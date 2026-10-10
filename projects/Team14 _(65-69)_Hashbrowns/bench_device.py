#!/usr/bin/env python3
"""
bench_device.py -- run attention-kernel variants on a NeuronCore, check the device's answers, time them.

    python bench_device.py                      # the default variant set at seq=96 dim=32
    python bench_device.py --calibrate          # ... plus the calibration kernels in calibration.py
    python bench_device.py --all-shapes         # the three level-8 shapes
    python bench_device.py a.py b.py+opt c.py:fn # any kernels: file[:entry][+opt]
    python bench_device.py --profile            # afterwards: neuron-profile the NEFFs saved above

Every number here comes from the device. The simulator is used for nothing.

A VARIANT is a kernel file, optionally an entry point other than nki_attention_ (file.py:fn), and
optionally "+opt". nki 0.6.0 compiles standalone kernels with its MLIR instruction scheduling and
linear-scan allocation turned OFF (CompileKernel._compile_opts -> disable_backend_optimizations), so
each engine runs its instructions in source order. "+opt" compiles the same kernel with
_enable_backend_opt=True, letting the scheduler reorder -- the closest thing this backend has to
torch.compile. It is marked experimental in the SDK.

For each variant and shape, in this order:
  COMPILE      kernel -> NIR -> NEFF, as StandaloneKernel does on a call (compile_kernel_to_nir, then
               compile_bir_to_neff). Compile time is reported apart from everything else, with the
               compiler's own latency estimate (NirResult.total_time_ns) -- an estimate, not a timing.
  CORRECT      the compiled NEFF runs on random inputs and on large ones (scores ~900, where a wrong
               max subtraction overflows); worst error against float64 as a fraction of the output
               RMS, pass at 2e-2. A variant that is wrong on the device is not timed.
  FIRST CALL   kernel(q, k, v) through the public API, timed whole: trace, compile, load, run. What a
               user pays once. (Not for +opt, which the public call cannot express.)
  DEVICE       CompiledKernel.benchmark: device-side tracing over warm-up + timed iterations, mean /
               min / max / std. Steady state, kernel only. Rounds are interleaved across variants
               (A B C A B C ...) so drift hits every variant alike.
  END TO END   CompiledKernel.run timed on the host, call by call: inputs copied to the device, the
               kernel, the output copied back. Median / p90 / p99 over every call.
A variant is called faster than the first one only if its slowest round beats the first one's
fastest round; otherwise the difference is within noise and it says so.

LNC AND CORES: the pod's runtime runs the device at NEURON_LOGICAL_NC_CONFIG=2, and while vLLM is
attached an LNC 1 request gets nothing ("Logical Neuron Core(s) not available ... cores busy" on
seat-65). So kernels compile for the runtime's LNC -- 2 here, --lnc overrides -- which runs the same
single-core program on both physical cores of one logical core: the same answer, and the same cost
for every variant. Unless NEURON_RT_VISIBLE_CORES is set, logical cores are tried one at a time in a
short subprocess and the first free one is used; vLLM holds the others.

Each variant compiles twice per shape, so a full run takes several minutes. Start it with nohup, as
the README says for long jobs, so a dropped kubectl exec does not kill it.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "bench_out")
MANIFEST = os.path.join(OUT_DIR, "neffs.json")
ENTRY = "nki_attention_"
DEFAULT_VARIANTS = [
    "my_attention_v0.py",            # A  baseline: the kernel the simulator trace came from
    "my_attention_v0.py+opt",        # B  the same, compiler scheduling on
    "my_attention.py",               # D  V1: row sum fused into exp, scale and negate folded
    "my_attention.py+opt",           #    V1, compiler scheduling on
    "my_attention_v3_dmaT.py",       # C  V1 with Q and K transposed by the DMA on the way in
    "my_attention_v4_kchunk.py",     # E  V1 with P^T and P @ V in 32-key chunks
    "my_attention_v4_kchunk.py+opt", #    the same, compiler scheduling on (can it pipeline the chunks?)
    "my_attention_v2_bf16.py",       #    V1 with the P @ V half in bf16
]
CALIBRATION = ["calibration.py:calib_dma", "calibration.py:calib_vec8",
               "calibration.py:calib_act8", "calibration.py:calib_alt8"]
TOL = 2e-2               # nkibench's grading tolerance: worst error as a fraction of the output RMS
SHAPES = [(96, 32)]
ALL_SHAPES = [(128, 64), (64, 128), (96, 32)]


# ---------------------------------------------------------------- inputs and references

def make_inputs(seq, dim, scale=1.0, seed=0):
    r = np.random.default_rng(seed)
    return tuple(np.ascontiguousarray((r.standard_normal((seq, dim)) * scale).astype(np.float32))
                 for _ in range(3))


def attention64(q, k, v):
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    s = q @ k.T / np.sqrt(q.shape[1])
    s -= s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=-1, keepdims=True)) @ v


def errors(got, want):
    """(worst error / output RMS, worst absolute error); inf if the shape or values are wrong."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    if got.shape != want.shape or not np.all(np.isfinite(got)):
        return float("inf"), float("inf")
    worst = float(np.abs(got - want).max())
    return worst / (float(np.sqrt((want ** 2).mean())) or 1.0), worst


def as_array(x):
    """The single output array, from whatever an nki call handed back."""
    if hasattr(x, "outputs"):
        x = x.outputs
    if isinstance(x, dict):
        x = next(iter(x.values()))
    if isinstance(x, (list, tuple)):
        x = x[0]
    return np.asarray(x)


def failure(e):
    tail = "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip().splitlines()[-6:]
    return f"RAISED {type(e).__name__}: {str(e)[:300]}\n" + "\n".join("        " + t for t in tail)


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


# ---------------------------------------------------------------- variants

class Variant:
    def __init__(self, spec, lnc):
        self.spec = spec
        self.opt = spec.endswith("+opt")
        body = spec[:-4] if self.opt else spec
        path, _, entry = body.partition(":")
        self.entry = entry or ENTRY
        self.path = os.path.abspath(path) if os.path.exists(path) else os.path.join(HERE, path)
        self.label = (os.path.splitext(os.path.basename(path))[0]
                      + (f":{self.entry}" if self.entry != ENTRY else "") + ("+opt" if self.opt else ""))
        # Each file under its own module name, so kernels that share an entry-point name are never
        # mistaken for one another by anything that caches on the module.
        mod_name = "variant_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
        spec_ = importlib.util.spec_from_file_location(mod_name, self.path)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        kernel = getattr(mod, self.entry)
        self.kernel = kernel[lnc] if lnc != 1 else kernel
        self.reference = getattr(mod, "REFERENCES", {}).get(self.entry, attention64)
        self.calibration = self.reference is not attention64     # not attention: never compared with it


def compile_variant(v, args, work_dir):
    """What StandaloneKernel does on a call, stopped before it runs: kernel -> NIR -> NEFF.
    Returns the CompiledKernel and a function that keys input arrays by the NEFF's input names."""
    from dataclasses import replace

    from nki.compiler.frontend import resolve_frontend_cls
    from nki.compiler.ncc_driver import compile_bir_to_neff
    from nki.framework.compiled import CompileKernel, compile_kernel_to_nir

    fe = resolve_frontend_cls()
    ck = v.kernel._to_subclass(CompileKernel, _frontend_cls=fe, _enable_backend_opt=v.opt)
    shutil.rmtree(work_dir, ignore_errors=True)       # neuronx-cc wants a clean directory
    os.makedirs(work_dir)
    opts = replace(ck._compile_opts(), artifacts_dir=work_dir,
                   output_path=os.path.join(work_dir, "kernel.neff"))
    nir = compile_kernel_to_nir(ck, inputs=ck._bind_args(args, {}), compile_opts=opts,
                                frontend=fe(enable_backend_opt=ck._enable_backend_opt),
                                enable_cache=False)
    compiled = compile_bir_to_neff(opts, nir, input_arrays=[],
                                   argument_names=[s.name for s in nir.descriptor.input_specs],
                                   output_arg_names=[s.name for s in nir.descriptor.output_specs])
    want = [s.name for s in compiled.input_specs]

    def named(arrays):
        d = compiled.prepare_inputs(ck._bind_args(arrays, {}))
        return d if set(d) == set(want) else dict(zip(want, arrays))   # by position, if names differ
    return compiled, named


def file_hash(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12] if os.path.exists(path) else None


# ---------------------------------------------------------------- the device run

RUNTIME_DOWN = "Failed to initialize NRT runtime"


def stop_if_runtime_down(e):
    """The runtime starts once per process; if it could not get its core, every later call fails the
    same way. Say so once and stop, rather than once per variant."""
    if RUNTIME_DOWN in str(e):
        sys.exit(f"\nThe Neuron runtime could not start on NeuronCore "
                 f"{os.environ.get('NEURON_RT_VISIBLE_CORES')} at LNC "
                 f"{os.environ.get('NEURON_LOGICAL_NC_CONFIG')}: {str(e)[:200]}\n"
                 f"Check neuron-ls; the lines above starting with ERROR NRT say why.")


def probe_core():
    """--probe-core: compile the smallest calibration kernel and run it once on the core in
    NEURON_RT_VISIBLE_CORES. Exit 0 only if the runtime gave us that core and the answer is right."""
    v = Variant("calibration.py:calib_dma", int(os.environ["NEURON_LOGICAL_NC_CONFIG"]))
    args = make_inputs(96, 32)
    compiled, named = compile_variant(
        v, args, os.path.join(OUT_DIR, f"probe_core_{os.environ['NEURON_RT_VISIBLE_CORES']}"))
    sys.exit(0 if errors(as_array(compiled.run(**named(args))), args[0])[0] <= TOL else 3)


def choose_core(lnc):
    if os.environ.get("NEURON_RT_VISIBLE_CORES"):
        return os.environ["NEURON_RT_VISIBLE_CORES"]
    print(f"Looking for a free NeuronCore at LNC {lnc}:")
    for core in range(4 if lnc == 2 else 8):                 # one trn2 device: 8 physical cores
        env = dict(os.environ, NEURON_RT_VISIBLE_CORES=str(core))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--probe-core"],
                           env=env, capture_output=True, text=True, timeout=900)
        text = p.stdout + p.stderr
        if p.returncode == 0:
            print(f"  core {core}: free -- using it")
            return str(core)
        busy = "not available" in text or "cores busy" in text
        print(f"  core {core}: " + ("busy" if busy else f"failed (exit {p.returncode}):\n"
              + "\n".join("      " + t for t in text.strip().splitlines()[-12:])))
    sys.exit(f"No free NeuronCore at LNC {lnc}. neuron-ls shows who holds them; vLLM (serve.sh) is "
             f"the usual owner.")


def device_run(a):
    lnc = a.lnc or int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
    a.lnc = lnc
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)               # this process only
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    os.environ["NEURON_RT_VISIBLE_CORES"] = choose_core(lnc)
    import nki
    print(f"nki {getattr(nki, '__version__', '?')}   NeuronCore {os.environ['NEURON_RT_VISIBLE_CORES']}"
          f"   LNC {a.lnc}   target {os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']}   "
          f"warm-up {a.warmup}, {a.iters} timed iterations x {a.rounds} rounds, "
          f"end-to-end {a.e2e} calls x {a.rounds} rounds")

    specs = (a.variants or DEFAULT_VARIANTS) + (CALIBRATION if a.calibrate else [])
    variants = []
    for s in specs:
        try:
            variants.append(Variant(s, a.lnc))
        except Exception as e:
            print(f"\n{s}: FAILED TO LOAD -- {failure(e)}")
    manifest = []

    for seq, dim in (ALL_SHAPES if a.all_shapes else SHAPES):
        print(f"\n================ seq={seq} dim={dim}")
        rand, large = make_inputs(seq, dim), make_inputs(seq, dim, 30.0)
        info, live = {}, []

        print("\nCOMPILE + CORRECT ON THE DEVICE -- error = worst |got - float64 ref| / output RMS")
        for v in variants:
            row = info[v.label] = {}
            work = os.path.join(OUT_DIR, f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', v.label)}_{seq}x{dim}")
            t0 = time.perf_counter()
            try:
                compiled, named = compile_variant(v, rand, work)
            except Exception as e:
                print(f"  {v.label:<34} compile {failure(e)}")
                continue
            row.update(compiled=compiled, named=named, compile_s=time.perf_counter() - t0,
                       est_us=(compiled.total_time_ns / 1e3) if compiled.total_time_ns else None)
            ok, msgs = True, []
            for label, args in (("random", rand), ("large", large)):
                try:
                    rel, worst = errors(as_array(compiled.run(**named(args))), v.reference(*args))
                    msgs.append(f"{label} {rel:.1e} (max abs {worst:.1e})" + ("" if rel <= TOL else " WRONG"))
                except Exception as e:
                    stop_if_runtime_down(e)
                    rel, _ = float("inf"), None
                    msgs.append(f"{label} {failure(e)}")
                ok &= rel <= TOL
            if not v.opt and not a.no_public:
                t0 = time.perf_counter()
                try:
                    rel, _ = errors(as_array(v.kernel(*rand)), v.reference(*rand))
                    row["first_s"] = time.perf_counter() - t0
                    msgs.append(f"public call {rel:.1e}" + ("" if rel <= TOL else " WRONG"))
                    ok &= rel <= TOL
                except Exception as e:
                    stop_if_runtime_down(e)
                    msgs.append(f"public call {failure(e)}")
                    ok = False
            print(f"  {v.label:<34} {'ok  ' if ok else 'FAIL'}  compiled in {row['compile_s']:.1f} s;  "
                  + ";  ".join(msgs))
            if ok:
                live.append(v)
        if not live:
            print("  nothing was correct on the device, so nothing is timed.")
            continue

        dev = {v.label: dict(mean=[], lo=[], hi=[], std=[]) for v in live}
        e2e = {v.label: [] for v in live}
        for _ in range(a.rounds):
            for v in list(live):
                row = info[v.label]
                try:
                    res = row["compiled"].benchmark(warmup=a.warmup, iterations=a.iters, **row["named"](rand))
                except Exception as e:
                    stop_if_runtime_down(e)
                    print(f"  {v.label}: benchmark failed -- {failure(e)}")
                    live.remove(v)
                    continue
                if errors(as_array(res), v.reference(*rand))[0] > TOL:
                    print(f"  {v.label}: the benchmarked NEFF returned a WRONG answer; dropped")
                    live.remove(v)
                    continue
                d = dev[v.label]
                d["mean"].append(res.latency * 1e6)
                d["lo"].append(res.latency_min * 1e6)
                d["hi"].append(res.latency_max * 1e6)
                d["std"].append((res.latency_std or 0.0) * 1e6)
            for v in live:
                row, named_rand = info[v.label], info[v.label]["named"](rand)
                for _ in range(a.e2e):
                    t0 = time.perf_counter()
                    row["compiled"].run(**named_rand)
                    e2e[v.label].append((time.perf_counter() - t0) * 1e6)
        if not live:
            continue

        base = live[0].label
        b_med = float(np.median(dev[base]["mean"]))
        print(f"\nTIMING -- microseconds unless marked. device = CompiledKernel.benchmark, median of the "
              f"per-round means; end-to-end = host time per run() call")
        print(f"  {'variant':<34} {'compile':>8} {'1st call':>9}   {'device':>8} {'min':>7} {'max':>7} "
              f"{'std':>6}   {'e2e p50':>8} {'p90':>8} {'p99':>8}   {'estimate':>8}   speedup")
        for v in live:
            d, row = dev[v.label], info[v.label]
            med = float(np.median(d["mean"]))
            first = f"{row['first_s']:8.1f}s" if "first_s" in row else f"{'-':>9}"
            est = f"{row['est_us']:8.2f}" if row["est_us"] else f"{'-':>8}"
            print(f"  {v.label:<34} {row['compile_s']:7.1f}s {first}   {med:8.2f} {min(d['lo']):7.2f} "
                  f"{max(d['hi']):7.2f} {np.mean(d['std']):6.2f}   {pct(e2e[v.label], 50):8.1f} "
                  f"{pct(e2e[v.label], 90):8.1f} {pct(e2e[v.label], 99):8.1f}   {est}   "
                  + ("    -" if v.calibration else f"{b_med / med:5.2f}x"))
            manifest.append(dict(variant=v.label, seq=seq, dim=dim, lnc=a.lnc,
                                 core=os.environ["NEURON_RT_VISIBLE_CORES"],
                                 neff=row["compiled"].neff_path, device_us=med,
                                 rounds_us=d["mean"], e2e_p50_us=pct(e2e[v.label], 50)))
        print(f"\n  per-round device means (us): " + "; ".join(
            f"{v.label} " + " ".join(f"{x:.2f}" for x in dev[v.label]["mean"]) for v in live))
        lo_b, hi_b = min(dev[base]["mean"]), max(dev[base]["mean"])
        for v in live[1:]:
            if v.calibration:
                continue
            lo, hi = min(dev[v.label]["mean"]), max(dev[v.label]["mean"])
            print(f"  {v.label}: " + ("FASTER than" if hi < lo_b else "SLOWER than" if lo > hi_b
                                     else "within noise of") + f" {base}")
        hashes = {v.label: file_hash(info[v.label]["compiled"].neff_path) for v in live}
        same = sorted(l for l in hashes if list(hashes.values()).count(hashes[l]) > 1
                      and not l.startswith("calibration:"))
        if same:
            print(f"  note: byte-identical NEFFs: {', '.join(same)} -- the change made no difference "
                  f"to the binary (or one variant was handed another's NEFF).")

        cal = {k: float(np.median(dev[f"calibration:{k}"]["mean"])) for k in
               ("calib_dma", "calib_vec8", "calib_act8", "calib_alt8") if f"calibration:{k}" in dev}
        if len(cal) == 4:
            vec = (cal["calib_vec8"] - cal["calib_dma"]) / 8
            act = (cal["calib_act8"] - cal["calib_dma"]) / 8
            hand = (cal["calib_alt8"] - (cal["calib_vec8"] + cal["calib_act8"]) / 2) / 8
            print(f"\nCALIBRATION -- [{seq}x{dim}] tiles, device time")
            print(f"  launch + one load + one store (calib_dma)   {cal['calib_dma']:8.2f} us")
            print(f"  one more dependent Vector instruction        {vec:8.3f} us")
            print(f"  one more dependent Scalar instruction        {act:8.3f} us")
            print(f"  extra per hand-off between the two engines   {hand:8.3f} us")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nNEFFs listed in {os.path.relpath(MANIFEST, HERE)}. For per-engine detail, next run:"
          f"\n    python bench_device.py --profile")


# ---------------------------------------------------------------- profile the saved NEFFs

PROFILE_KEYS = re.compile(r"total_time|latency|active_time_percent|_util|dma|hbm|instruction",
                          re.IGNORECASE)


def profile_run(a):
    """A separate invocation on purpose: the benchmark process holds the NeuronCore until it exits,
    and neuron-profile capture needs it."""
    exe = shutil.which("neuron-profile") or shutil.which("neuron-explorer")
    if not exe:
        sys.exit("neither neuron-profile nor neuron-explorer is on PATH")
    if not os.path.exists(MANIFEST):
        sys.exit(f"no {MANIFEST}: run python bench_device.py first")
    env = dict(os.environ)
    summaries = {}
    for e in json.load(open(MANIFEST)):
        if a.variants and e["variant"] not in a.variants:
            continue
        tag = re.sub(r"[^A-Za-z0-9_.+-]", "_", f"{e['variant']}_{e['seq']}x{e['dim']}")
        env["NEURON_LOGICAL_NC_CONFIG"] = str(e["lnc"])
        env["NEURON_RT_VISIBLE_CORES"] = str(e.get("core", "0"))
        ntff = os.path.join(OUT_DIR, f"{tag}.ntff")
        print(f"\n{tag}")
        cap = [exe, "capture", "-n", e["neff"], "-s", ntff]
        p = subprocess.run(cap, capture_output=True, text=True, env=env, timeout=900)
        if p.returncode or not os.path.exists(ntff):
            print(f"  $ {' '.join(cap)}\n  capture failed (exit {p.returncode}):")
            print("\n".join("    " + t for t in (p.stdout + p.stderr).strip().splitlines()[-25:]))
            continue
        for fmt, suffix in (("summary-json", "summary.json"), ("json", "full.json")):
            view = [exe, "view", "-n", e["neff"], "-s", ntff, "--output-format", fmt]
            p = subprocess.run(view, capture_output=True, text=True, env=env, timeout=900)
            path = os.path.join(OUT_DIR, f"{tag}.{suffix}")
            try:
                data = json.loads(p.stdout)
            except json.JSONDecodeError:
                print(f"  $ {' '.join(view)}\n  no JSON (exit {p.returncode}): "
                      + " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:]))
                continue
            json.dump(data, open(path, "w"), indent=1)
            print(f"  saved {os.path.relpath(path, HERE)}")
            if fmt == "summary-json":
                if len(data) == 1 and isinstance(next(iter(data.values())), dict):
                    data = next(iter(data.values()))          # {model: metrics}, as nki's own parser does
                summaries[tag] = data

    if not summaries:
        sys.exit("\nNo profile summary was captured.")
    keys = sorted({k for d in summaries.values() for k, v in d.items()
                   if PROFILE_KEYS.search(k) and isinstance(v, (int, float))})
    tags = list(summaries)
    print(f"\nPROFILE SUMMARY -- neuron-profile summary-json, numeric fields matching {PROFILE_KEYS.pattern}")
    print(f"  {'metric':<46}" + "".join(f"{t[:26]:>28}" for t in tags))
    for k in keys:
        print(f"  {k[:46]:<46}" + "".join(
            f"{summaries[t][k]:>28.6g}" if isinstance(summaries[t].get(k), (int, float))
            else f"{'-':>28}" for t in tags))
    print("\nFull summaries and timelines are in bench_out/*.summary.json and *.full.json.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", help="file[:entry][+opt]; default: the standard set")
    ap.add_argument("--calibrate", action="store_true", help="also run the calibration kernels")
    ap.add_argument("--all-shapes", action="store_true", help="the three level-8 shapes, not just 96x32")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--e2e", type=int, default=100, help="end-to-end calls per variant per round")
    ap.add_argument("--no-public", action="store_true", help="skip the first-call timing")
    ap.add_argument("--lnc", type=int, choices=(1, 2), default=None,
                    help="default: the runtime's NEURON_LOGICAL_NC_CONFIG (2 on the seat pods)")
    ap.add_argument("--probe-core", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--profile", action="store_true",
                    help="capture neuron-profile traces of the NEFFs a previous run saved")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.probe_core:
        probe_core()
    profile_run(a) if a.profile else device_run(a)


if __name__ == "__main__":
    main()
