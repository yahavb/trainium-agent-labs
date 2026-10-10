"""hwcheck.py -- layers 2 and 3 of the kernel checker: run a kernel on the REAL chip and profile it.

nkibench.py is layer 1: rules, then numerics and HBM bytes on the CPU simulator. This file is the
part that needs a device. For every shape of a level it

  1. runs the kernel on a NeuronCore and compares the output with the same NumPy reference
     nkibench uses (fast and wrong still scores zero),
  2. captures a hardware trace with neuron-explorer,
  3. prints the numbers the simulator cannot see -- time on the chip, how busy each engine was --
     next to the ones it can, so the two can be cross-checked.

    python hwcheck.py --level 4 --check reference_level4.py
    python hwcheck.py --level 4 --check reference_level4.py --json out.json

Lives next to nkibench.py. THE CHIP MUST BE FREE: the model server takes every logical core at
LNC=2, so run this on a seat that is not serving the model (measured: "cores busy" otherwise).

Runs at LNC=1 by default, whatever the pod's own setting. Measured on a seat: at the pod's LNC=2
the kernel runs on both physical cores of the logical core, so every byte and transfer count comes
back exactly doubled and reads as "2.00x redundant traffic" when the kernel is at the floor.

Kernels are called with NumPy arrays, which is NKI's standalone mode. The torch route the AWS
profiling skill documents needs torch_neuronx, and the seat pods do not have it.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Fields copied out of neuron-explorer's summary. Names are the tool's own.
FIELDS = [
    "total_time", "tensor_engine_active_time_percent", "vector_engine_active_time_percent",
    "scalar_engine_active_time_percent", "gpsimd_engine_active_time_percent",
    "dma_active_time_percent", "dma_transfer_total_bytes", "dma_transfer_count",
    "dma_transfer_average_bytes", "inputs_outputs_weights_size_bytes", "hbm_read_bytes",
    "hbm_write_bytes", "mfu_estimated_percent", "mbu_estimated_percent", "hardware_flops",
    "mm_arithmetic_intensity", "peak_flops_bandwidth_ratio", "matmul_instruction_count",
]


# ---------------------------------------------------------------- worker: one shape, one process

def worker(job_path):
    """Run one shape on the chip. Separate process per shape because the compiler wants a clean
    artifacts directory each time, and because the runtime claims its core for the process."""
    job = pickle.load(open(job_path, "rb"))
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(job["core"])
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = job["target"]
    os.environ["NKI_ARTIFACTS_DIR"] = job["artifacts"]
    os.environ["NKI_DEBUG_INFO"] = "1"

    import nkibench as nb
    spec = nb.LEVELS[job["level"]]
    out = dict(ok=False, error=None, mismatch=None)
    try:
        kernel = nb.load_kernel(job["kernel"], spec["entry"])
        args, _ = nb.make_inputs(job["case"], job["level"], job["seed"])
        want = spec["ref"](*args)
        before = [np.array(a, copy=True) if hasattr(a, "shape") else a for a in args]
        # The LNC the kernel is launched with must match the runtime's, or the runtime errors.
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
        got = np.asarray((kernel[lnc] if lnc != 1 else kernel)(*args))
        out["mismatch"] = nb.describe_mismatch(got, want, job["tol"]) or nb.check_inputs_untouched(before, args)
        out["ok"] = not out["mismatch"]
        out["floor_bytes"] = nb.minimum_hbm_bytes(args, want)
    except Exception as e:  # reported to the caller, which decides what the agent is told
        out["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    pickle.dump(out, open(job["result"], "wb"))


# ---------------------------------------------------------------- profile

def profile(neff, workdir, core):
    """Capture one hardware trace of an already-compiled kernel and return its summary."""
    env = dict(os.environ, NEURON_RT_VISIBLE_CORES=str(core), NEURON_RT_ENABLE_DGE_NOTIFICATIONS="1")
    ntff = os.path.join(workdir, "profile.ntff")
    cap = subprocess.run(["neuron-explorer", "capture", "-n", neff, "-s", ntff, "--profile-nth-exec=2"],
                         env=env, capture_output=True, text=True, timeout=300)
    # With --profile-nth-exec=2 the file is written as profile_exec_2.ntff.
    traces = sorted(f for f in os.listdir(workdir) if f.endswith(".ntff"))
    if cap.returncode != 0 or not traces:
        raise RuntimeError(f"neuron-explorer capture failed: {(cap.stderr or cap.stdout)[-400:]}")
    ntff = os.path.join(workdir, traces[-1])
    view = subprocess.run(["neuron-explorer", "view", "--output-format", "summary-json", "-n", neff, "-s", ntff],
                          env=env, capture_output=True, text=True, timeout=300)
    if view.returncode != 0:
        raise RuntimeError(f"neuron-explorer view failed: {view.stderr[-400:]}")
    d = json.loads(view.stdout)
    if len(d) == 1 and isinstance(next(iter(d.values())), dict):
        d = next(iter(d.values()))
    return {k: d[k] for k in FIELDS if k in d}, ntff


def verdict(m, floor):
    """Turn the profile into sentences. A first cut on purpose: it states what was measured and
    names one change. Improve THIS before touching the prompt."""
    lines = []
    moved = m.get("dma_transfer_total_bytes")
    te = 100 * m.get("tensor_engine_active_time_percent", 0)
    dma = 100 * m.get("dma_active_time_percent", 0)
    if moved and floor:
        waste = moved / floor
        if waste > 1.05:
            lines.append(f"REDUNDANT TRAFFIC: {moved:,} bytes crossed the memory bus where {floor:,} "
                         f"would do ({waste:.2f}x). Some tile is loaded more than once -- load it once, "
                         f"outside the loop that does not change it, and reuse it.")
        else:
            lines.append(f"TRAFFIC at the floor: {moved:,} bytes moved, {waste:.2f}x the minimum. "
                         f"There is no reuse left to find at this shape.")
    n, avg = m.get("dma_transfer_count"), m.get("dma_transfer_average_bytes")
    if n and avg and avg < 4096:
        lines.append(f"ISSUE BOUND: {n:,} transfers averaging {avg:,.0f} bytes. The cost is the number "
                     f"of transfers, not the bytes. Move whole tiles.")
    if dma > te + 5:
        lines.append(f"WAITING ON DATA: data movement was busy {dma:.0f}% of the run and the tensor "
                     f"engine {te:.0f}%. The engine is idle while loads run.")
    elif te > dma + 5:
        lines.append(f"COMPUTE LIMITED: the tensor engine was busy {te:.0f}% of the run against "
                     f"{dma:.0f}% for data movement. Loads keep up; tuning them buys little.")
    return lines


# ---------------------------------------------------------------- driver

def check(kernel_path, level_n, core=0, target="trn2", tol=2e-2, seed=0, keep=None, only=None, lnc=1):
    import nkibench as nb
    # Inherited by the worker and by neuron-explorer, so the kernel and the runtime agree.
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    spec = nb.LEVELS[level_n]
    violations = nb.check_rules(open(kernel_path).read(), level_n)
    print(f"level {level_n}: {spec['op']}   [ON THE CHIP, core {core}, LNC={lnc}]")
    if violations:
        print("\nRULE VIOLATIONS -- not run on the chip:")
        for v in violations:
            print(f"  {v}")
        return []

    root = keep or tempfile.mkdtemp(prefix="hwcheck_")
    results = []
    for i, case in enumerate(spec["shapes"]):
        if only is not None and i != only:
            continue
        lbl = nb.label(case, level_n)
        work = os.path.join(root, f"shape{i}")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(os.path.join(work, "art"))
        job = dict(kernel=os.path.abspath(kernel_path), level=level_n, case=case, seed=seed, tol=tol,
                   core=core, target=target, artifacts=os.path.join(work, "art"),
                   result=os.path.join(work, "result.pkl"))
        job_path = os.path.join(work, "job.pkl")
        pickle.dump(job, open(job_path, "wb"))
        run = subprocess.run([sys.executable, os.path.abspath(__file__), "--_worker", job_path],
                             capture_output=True, text=True, timeout=900)
        rec = dict(shape=lbl, source="device")
        if not os.path.exists(job["result"]):
            # The worker died before reporting: surface the runtime's own reason, not a guess.
            why = [l for l in (run.stderr + run.stdout).splitlines() if "ERROR" in l or "Error" in l]
            rec["error"] = (why[0] if why else run.stderr[-300:]).strip()[:400]
        else:
            rec.update(pickle.load(open(job["result"], "rb")))
        if rec.get("error") or rec.get("mismatch"):
            print(f"\n  {lbl}: FAILED ON THE CHIP")
            print("    " + (rec.get("error") or rec["mismatch"]).replace("\n", "\n    "))
            results.append(rec)
            continue
        try:
            rec["metrics"], rec["ntff"] = profile(os.path.join(work, "art", "kernel.neff"), work, core)
            rec["neff"] = os.path.join(work, "art", "kernel.neff")
        except Exception as e:
            rec["error"] = f"correct on the chip, but profiling failed: {e}"
            print(f"\n  {lbl}: {rec['error']}")
            results.append(rec)
            continue
        m = rec["metrics"]
        rec["verdict"] = verdict(m, rec.get("floor_bytes"))
        print(f"\n  {lbl}: correct on the chip")
        print(f"    time on chip      {m['total_time'] * 1e6:,.1f} microseconds")
        print(f"    tensor engine     busy {100 * m['tensor_engine_active_time_percent']:.1f}% of the run")
        print(f"    data movement     busy {100 * m['dma_active_time_percent']:.1f}% of the run")
        print(f"    HBM traffic       {m['dma_transfer_total_bytes']:,} bytes in {m['dma_transfer_count']} transfers"
              f" (floor {rec.get('floor_bytes', 0):,})")
        print(f"    utilisation       compute {100 * m.get('mfu_estimated_percent', 0):.1f}%,"
              f" memory bandwidth {100 * m.get('mbu_estimated_percent', 0):.1f}%  (the tool's estimates)")
        for line in rec["verdict"]:
            print(f"    {line}")
        results.append(rec)
    if keep is None:
        shutil.rmtree(root, ignore_errors=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--_worker", help=argparse.SUPPRESS)
    ap.add_argument("--level", type=int)
    ap.add_argument("--check", metavar="KERNEL.py")
    ap.add_argument("--core", type=int, default=0, help="logical NeuronCore to run on")
    ap.add_argument("--target", default="trn2")
    ap.add_argument("--lnc", type=int, default=1, help="physical cores per logical core; 2 doubles every count")
    ap.add_argument("--shape", type=int, help="run only this shape index of the level")
    ap.add_argument("--keep", metavar="DIR", help="keep the NEFF and trace files here")
    ap.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    a = ap.parse_args()
    if a._worker:
        return worker(a._worker)
    if a.level is None or not a.check:
        ap.error("--level and --check are required")
    results = check(a.check, a.level, core=a.core, target=a.target, keep=a.keep, only=a.shape, lnc=a.lnc)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=2, default=str)
    failed = [r for r in results if r.get("error") or r.get("mismatch")]
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()
