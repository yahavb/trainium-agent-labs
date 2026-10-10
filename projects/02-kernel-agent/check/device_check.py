"""Run NKI kernels on the real Trainium chip and compare them with the reference: the checker's layer 2,
which the organizers' harness leaves out ("real latency via nki.baremetal ... need the chip").

For each kernel, on every graded shape of its level (nkibench.make_inputs, the grader's own seed):
  1. the CPU simulator, as the agent was graded (nkibench.simulate_and_count);
  2. the device: in nki 0.6.0, calling a @nki.jit kernel with NumPy arrays compiles it and runs it
     standalone on a NeuronCore;
  3. both outputs against the NumPy reference, with nkibench's tolerance (worst error <= 2% of RMS);
  4. timing: the median of --reps device calls after one warm-up. This is wall time per call from
     Python, so it includes moving inputs and outputs and the launch, not only the kernel.

    cd $THEIRS && NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_LOGICAL_NC_CONFIG=1 \\
        PYTHONPATH=$THEIRS:$KIT/agent/nki:$KIT/agent/nki/check python device_check.py \\
        --out device.json 2:l2.py 3:l3.py ...

Needs NeuronCores no other process holds: the vLLM workers lock all four, so stop the model server first.
--sim-only skips the device (to test this script anywhere the simulator runs).
"""
import argparse
import json
import os
import statistics
import sys
import time
import traceback

import numpy as np

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
import nkibench  # noqa: E402
for _m in ("ops07", "ops08"):                  # our levels 9-14, through nkibench's extension point
    try:
        __import__(_m)
    except ImportError:
        pass


def compare(got, want, tol=2e-2):
    """(ok, worst error / RMS of the reference, why)."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if got.shape != want.shape:
        return False, None, f"shape {got.shape}, reference {want.shape}"
    if not np.all(np.isfinite(got)):
        return False, None, f"{int((~np.isfinite(got)).sum())} non-finite values"
    scale = float(np.sqrt((want ** 2).mean())) or 1.0
    worst = float(np.abs(got - want).max()) / scale
    return worst <= tol, worst, "" if worst <= tol else f"worst error {worst:.3g} of RMS > {tol}"


def run(level, path, reps, sim_only):
    spec = nkibench.LEVELS[level]
    kernel = nkibench.load_kernel(path, spec["entry"])
    rows = []
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        want = spec["ref"](*args)
        row = {"level": level, "kernel": path, "shape": nkibench.label(case, level)}
        try:
            got, _ = nkibench.simulate_and_count(kernel, [a.copy() if isinstance(a, np.ndarray) else a for a in args])
            row["sim_ok"], row["sim_err"], row["sim_why"] = compare(got, want)
        except Exception as e:  # noqa: BLE001
            row["sim_ok"], row["sim_err"], row["sim_why"] = False, None, f"{type(e).__name__}: {e}"[:300]
        if not sim_only:
            try:
                t0 = time.perf_counter()
                got = kernel(*[a.copy() if isinstance(a, np.ndarray) else a for a in args])
                row["first_call_s"] = round(time.perf_counter() - t0, 2)      # includes the compile
                row["dev_ok"], row["dev_err"], row["dev_why"] = compare(got, want)
                times = []
                for _ in range(reps):
                    fresh = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
                    t0 = time.perf_counter()
                    kernel(*fresh)
                    times.append(time.perf_counter() - t0)
                row["call_ms_median"] = round(statistics.median(times) * 1e3, 3)
                row["call_ms_min"] = round(min(times) * 1e3, 3)
            except Exception as e:  # noqa: BLE001
                row["dev_ok"], row["dev_err"] = False, None
                row["dev_why"] = (f"{type(e).__name__}: {e}"[:300] + " | " +
                                  traceback.format_exc().strip().splitlines()[-1][:200])
        rows.append(row)
        fmt = lambda ok, err, why: ("ok  " if ok else "FAIL") + (f" {err:.2e}" if err is not None else "") + \
            ("" if ok else f" ({why})")
        dev = ("" if sim_only else
               f" | device {fmt(row['dev_ok'], row['dev_err'], row['dev_why'])}" +
               (f" | {row['call_ms_median']:.3f} ms/call (first {row['first_call_s']} s)" if "call_ms_median" in row else ""))
        print(f"L{level} {os.path.basename(path)} {row['shape']}: sim {fmt(row['sim_ok'], row['sim_err'], row['sim_why'])}{dev}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="+", help="LEVEL:path, e.g. 4:nki_kernels/l4.py")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="device.json")
    ap.add_argument("--sim-only", action="store_true")
    a = ap.parse_args()
    print(f"nki target {os.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE')}, "
          f"NEURON_LOGICAL_NC_CONFIG={os.environ.get('NEURON_LOGICAL_NC_CONFIG', '(unset)')}", flush=True)
    rows = []
    for item in a.kernels:
        lv, path = item.split(":", 1)
        try:
            rows += run(int(lv), path, a.reps, a.sim_only)
        except Exception as e:  # noqa: BLE001
            print(f"L{lv} {path}: could not load: {type(e).__name__}: {e}", flush=True)
            rows.append({"level": int(lv), "kernel": path, "load_error": f"{type(e).__name__}: {e}"[:300]})
    json.dump(rows, open(a.out, "w"), indent=1)
    if not a.sim_only:
        dev = [r for r in rows if "dev_ok" in r]
        print(f"\ndevice: {sum(r['dev_ok'] for r in dev)} of {len(dev)} kernel-shapes match the reference "
              f"(simulator: {sum(r.get('sim_ok', False) for r in dev)}). Written to {a.out}")


if __name__ == "__main__":
    sys.exit(main())
