#!/usr/bin/env python3
"""
bench_suites.py — run whole nkibench suites on the NeuronCore, time and profile each kernel, print the
gain or loss against the suite's baseline.

bench_device.py here times single-head attention variants (level 8, [seq, dim] inputs). This covers
the suites it does not: the tiled matmul (level 4) and multi-head attention (level 9, [seq, heads,
dim] inputs), where a version can pin its own LNC so one kernel is timed on one core and on both.
It was trainium-agent-labs' bench_device.py; its single-head suite is left out, as bench_device.py
covers it. Per kernel and shape:

    1. compile each version with nki's standalone path and run it on the device, checking the
       answer against the NumPy reference
    2. time it with the runtime's device-side benchmark: warmup, then many timed iterations
    3. profile the same NEFF with neuron-explorer (capture, then view --output-format summary-json)
       for per-engine active time and HBM bytes

    python bench_suites.py matmul             # reference_level4 against matmul_parallel
    python bench_suites.py mha                # multi-head attention: V0 and V6 per head vs mha_fast.py
    python bench_suites.py mha --iterations 2000 --no-profile

Results: a table on stdout, and one folder per run, profiles/<suite>/run-<time>/, holding results.json
and, per version and shape, the NEFF (build/kernel.neff), the trace (profile_exec_2.ntff) and the full
summary-json (metrics.json). Nothing is shared between runs: an earlier layout reused one folder per
version, and a timing-only rerun deleted the profiled run's traces twice.

Measured on seat-68, all of which the defaults encode:
  * The model server holds NeuronCores 0-1, not 2-3. Asking for 0 or 1 fails with "Logical Neuron
    Core(s) not available ... cores busy". So timing runs on 2 and the profiler on 3: this process
    keeps its core once the runtime starts, so the profiler cannot share it.
  * torch_xla cannot run NKI kernels in the seat image -- nki's torch path needs torch_neuronx,
    which is not installed. nki's own standalone compile-and-run path needs nothing extra.
  * Build at LNC 1 unless a version pins LNC 2. A kernel that does not split its work by program id
    runs whole on both physical cores at LNC 2: single-head attention showed exactly twice the HBM
    bytes, and timed 19.3 us against 17.8 us at LNC 1. mha_fast.py splits its heads, so it is timed
    at both.
  * Run to run, the same kernel differs by about 0.5% (19.29 vs 19.38 us on two cores). Treat a
    difference under 1% as noise.
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, file, entry point[, lnc]). The first one present is the baseline every other row is compared
# to. A fourth element pins that version's LNC; without it, --lnc applies.
SUITES = {
    # Level 4's reference and shapes. matmul_parallel exposes the same kernel under the level-5 name.
    "matmul": (4, [("level-4 reference", "reference_level4.py", "nki_matmul_tiled_"),
                   ("parallel", "matmul_parallel.py", "nki_matmul_hoist_load_")]),
    # One kernel call for every head. The fast kernel runs twice -- one core, then both cores of an
    # LNC 2 launch -- so the two-core gain is separate from the rest.
    "mha": (9, [("M0 V0 per head", "mha_v0_loop.py", "nki_mha_", 1),
                ("M6 V6 per head", "mha_v6_loop.py", "nki_mha_", 1),
                ("fast, 1 core", "mha_fast.py", "nki_mha_", 1),
                ("fast, 2 cores", "mha_fast.py", "nki_mha_", 2),
                ("fast balanced, 2 cores", "mha_fast.py", "nki_mha_balanced_", 2)]),
}

# summary-json fields. The *_percent ones are fractions of total_time despite the name.
ENGINES = [("TensorE", "tensor_engine_active_time_percent"),
           ("VectorE", "vector_engine_active_time_percent"),
           ("ScalarE", "scalar_engine_active_time_percent"),
           ("GpSimd", "gpsimd_engine_active_time_percent"),
           ("DMA", "dma_active_time_percent")]


def run_on_device(kernel, args, work, lnc, warmup, iterations):
    """Compile with nki's standalone path and time it with the runtime's device-side benchmark."""
    from nki.framework.compiled import StandaloneKernel

    seen = {}

    def benchmark(compiled, inputs, outputs):
        r = compiled.benchmark(warmup=warmup, iterations=iterations, **inputs)
        for name, arr in r.outputs.items():
            outputs[name][...] = arr
        seen.update(result=r, neff=compiled.neff_path)

    # neuronx-cc refuses an artifacts directory that is not empty.
    shutil.rmtree(work, ignore_errors=True)
    standalone = kernel._to_subclass(StandaloneKernel, _executor=benchmark, target="trn2",
                                     lnc=lnc, artifacts_dir=os.path.join(work, "build"))
    got = standalone(*args)
    return got, seen["result"], seen["neff"]


def profile(neff, work, core):
    """neuron-explorer capture + view. Returns the summary dict, or a string saying what failed."""
    tool = shutil.which("neuron-explorer")
    if not tool:
        return "neuron-explorer is not on PATH"
    env = dict(os.environ, NEURON_RT_VISIBLE_CORES=core)
    cap = subprocess.run([tool, "capture", "-n", neff, "-s", os.path.join(work, "profile.ntff"),
                          "--profile-nth-exec=2", "--enable-dge-notifs"],
                         capture_output=True, text=True, env=env)
    # capture names the trace after the execution it profiled: profile_exec_2.ntff, not profile.ntff
    traces = sorted(glob.glob(os.path.join(work, "*.ntff")))
    if cap.returncode or not traces:
        return f"capture failed: {(cap.stderr or cap.stdout).strip()[-300:]}"
    view = subprocess.run([tool, "view", "--output-format", "summary-json", "-n", neff,
                           "-s", traces[-1]], capture_output=True, text=True, env=env)
    raw = view.stdout
    try:
        summary = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except ValueError:
        return f"view returned no JSON: {(view.stderr or raw).strip()[-300:]}"
    with open(os.path.join(work, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    # One NEFF, one entry, keyed by a hash of it.
    return next(iter(summary.values())) if len(summary) == 1 else summary


def change(new, base):
    d = (new - base) / base * 100
    # Positive always means faster, so the sign reads the same on every row.
    if abs(d) < 1.0:
        return f"{-d:+.1f}%  (within noise)"
    return f"{-d:+.1f}% {'FASTER' if d < 0 else 'SLOWER'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=sorted(SUITES))
    ap.add_argument("--core", default="2", help="core to time on; the model server holds 0-1")
    ap.add_argument("--profile-core", default="3", help="core for neuron-explorer; must differ")
    ap.add_argument("--lnc", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()

    # Before the runtime starts: it reads this once, at first model load.
    os.environ["NEURON_RT_VISIBLE_CORES"] = a.core
    sys.path.insert(0, HERE)
    import nkibench

    level, candidates = SUITES[a.suite]
    present = [c for c in candidates if os.path.exists(os.path.join(HERE, c[1]))]
    for c in candidates:
        if c not in present:
            print(f"(skipping {c[0]}: {c[1]} is not here)")
    if len(present) < 2:
        sys.exit("need at least two versions to compare.")
    shapes = nkibench.LEVELS[level]["shapes"]
    root = os.path.join(HERE, "profiles", a.suite, time.strftime("run-%Y%m%d-%H%M%S"))
    print(f"{a.suite}: {len(present)} versions x {len(shapes)} shapes, baseline {present[0][0]}. "
          f"Timing on core {a.core} at LNC {a.lnc} unless a version pins it, {a.iterations} iterations after {a.warmup} "
          f"warmup; profiling on core {a.profile_core}.\n")

    results = []
    for case_i, case in enumerate(shapes):
        args, _ = nkibench.make_inputs(case, level)
        want = nkibench.LEVELS[level]["ref"](*args)
        print(f"=== {nkibench.label(case, level)}")
        base = None
        for label, path, entry, *pinned in present:
            work = os.path.join(root, f"{label.replace(' ', '_')}_case{case_i}")
            row = dict(version=label, shape=nkibench.label(case, level), case=case_i)
            t0 = time.time()
            try:
                got, r, neff = run_on_device(nkibench.load_kernel(os.path.join(HERE, path), entry),
                                             args, work, pinned[0] if pinned else a.lnc,
                                             a.warmup, a.iterations)
            except Exception as e:
                msg = str(e).strip().splitlines()
                print(f"  {label:<18} FAILED: {type(e).__name__}: {msg[-1] if msg else ''}")
                row["error"] = f"{type(e).__name__}: {e}"
                results.append(row)
                continue
            mismatch = nkibench.describe_mismatch(got, want)
            row.update(mean_us=r.latency * 1e6, min_us=r.latency_min * 1e6,
                       max_us=r.latency_max * 1e6, std_us=r.latency_std * 1e6,
                       correct=not mismatch, mismatch=mismatch or "", neff=neff)
            if base is None and row["correct"]:
                base = row
            delta = "baseline" if row is base else (change(row["mean_us"], base["mean_us"])
                                                     if base else "")
            verdict = "correct" if row["correct"] else "WRONG: " + mismatch.splitlines()[0]
            print(f"  {label:<18} {row['mean_us']:8.2f} us  +/-{row['std_us']:.2f}  "
                  f"(min {row['min_us']:.2f})  {delta}  [{verdict}]  {time.time() - t0:.0f}s")

            if not a.no_profile:
                s = profile(neff, work, a.profile_core)
                if isinstance(s, str):
                    print(f"  {'':<18} profile: {s}")
                else:
                    row["profile"] = {k: s.get(k) for _, k in ENGINES}
                    row["profile"].update(total_time_us=(s.get("total_time") or 0) * 1e6,
                                          hbm_read=s.get("hbm_read_bytes"),
                                          hbm_write=s.get("hbm_write_bytes"))
                    busy = "  ".join(f"{n} {s[k] * 100:.0f}%" for n, k in ENGINES
                                     if isinstance(s.get(k), (int, float)))
                    print(f"  {'':<18} busy: {busy}   HBM read {s.get('hbm_read_bytes'):,} "
                          f"write {s.get('hbm_write_bytes'):,}")
            results.append(row)
        print()

    # One number per version: the geometric mean of its speedup over the baseline across shapes.
    print("=========== overall, against", present[0][0], "===========")
    for label, *_ in present[1:]:
        ratios = []
        for case_i in range(len(shapes)):
            b = next((r for r in results if r["case"] == case_i and r["version"] == present[0][0]
                      and r.get("correct")), None)
            v = next((r for r in results if r["case"] == case_i and r["version"] == label
                      and r.get("correct")), None)
            if b and v:
                ratios.append(b["mean_us"] / v["mean_us"])
        if ratios:
            g = math.exp(sum(map(math.log, ratios)) / len(ratios))
            per = ", ".join(f"{x:.3f}x" for x in ratios)
            print(f"  {label:<18} {g:.3f}x  ({'faster' if g > 1 else 'slower'}; per shape {per})")
        else:
            print(f"  {label:<18} no shape where both it and the baseline ran correctly")

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "results.json"), "w") as f:
        json.dump(dict(suite=a.suite, core=a.core, lnc=a.lnc, warmup=a.warmup,
                       iterations=a.iterations, results=results), f, indent=1)
    print(f"\nresults: {os.path.relpath(root, HERE)}/ -- results.json, and per version and shape the "
          f"NEFF, trace and neuron-explorer summary")


if __name__ == "__main__":
    main()
