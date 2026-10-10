"""chip.py -- measure one matmul kernel source on the real chip.

    python chip.py --kernel kernels/matmul_blocked.py --size 2048 --reps 3
    python chip.py --grid "1,1,1;16,2,8" --size 2048 --out grid.jsonl      # knob settings M,N,K

measure() is what the optimisation loop calls: it compiles the kernel, runs it on a NeuronCore,
checks the result against NumPy, then captures `reps` hardware profiles. It returns numbers only;
turning them into an instruction is the checker agent's job, not this file's.

Needs a seat whose chip is free (the model server takes every core), and hwcheck.py beside it.
"""

import argparse
import json
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ENTRY = "nki_matmul_"
KNOBS = ("TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K")


def get_knobs(src):
    return tuple(int(re.search(rf"^\s*{k}\s*=\s*(\d+)", src, re.M).group(1)) for k in KNOBS)


def set_knobs(src, m, n, k):
    for name, v in zip(KNOBS, (m, n, k)):
        src, hits = re.subn(rf"^(\s*{name}\s*=\s*)\d+", rf"\g<1>{int(v)}", src, count=1, flags=re.M)
        if hits != 1:
            raise ValueError(f"could not find the {name} line")
    return src


def parse_shape(text):
    """'2048' -> (2048, 2048, 2048); 'K4096_M512_N12288' -> (4096, 512, 12288)."""
    if str(text).isdigit():
        return (int(text),) * 3
    d = dict((p[0], int(p[1:])) for p in str(text).split("_"))
    return (d["K"], d["M"], d["N"])


def shape_tag(shape):
    k, m, n = shape
    return str(k) if k == m == n else f"K{k}_M{m}_N{n}"


def _worker(job_path):
    job = pickle.load(open(job_path, "rb"))
    os.environ.update(NEURON_RT_VISIBLE_CORES=str(job["core"]), NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
                      NKI_ARTIFACTS_DIR=job["art"], NKI_DEBUG_INFO="1")
    import importlib.util
    import numpy as np
    out = dict(correct=False, error=None)
    try:
        spec = importlib.util.spec_from_file_location("candidate", job["kernel"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        K, M, N = job["shape"]
        r = np.random.default_rng(0)
        lhsT = r.standard_normal((K, M)).astype(np.float32)
        rhs = r.standard_normal((K, N)).astype(np.float32)
        want = lhsT.T @ rhs
        got = np.asarray(getattr(mod, ENTRY)(lhsT, rhs))
        out["correct"] = bool(got.shape == want.shape and np.allclose(got, want, rtol=2e-2, atol=2e-2))
        if not out["correct"]:
            out["error"] = (f"WRONG RESULT on the chip: shape {got.shape} against {want.shape}, "
                            f"largest error {float(np.abs(got - want).max()) if got.shape == want.shape else 'n/a'}")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    pickle.dump(out, open(job["result"], "wb"))


def measure(source, size=2048, reps=3, core=0, keep=None):
    shape = size if isinstance(size, tuple) else (size,) * 3
    K, M, N = shape
    """Compile, run, verify and profile one kernel source. Returns a dict; never raises."""
    import hwcheck
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"   # at the pod's LNC=2 every count comes back doubled
    work = keep or tempfile.mkdtemp(prefix="chip_")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "art"))
    kpath = os.path.join(work, "kernel.py")
    open(kpath, "w").write(source)
    job = dict(kernel=kpath, shape=shape, core=core, art=os.path.join(work, "art"),
               result=os.path.join(work, "result.pkl"))
    jpath = os.path.join(work, "job.pkl")
    pickle.dump(job, open(jpath, "wb"))
    t0 = time.time()
    rec = dict(size=shape_tag(shape), correct=False, error=None)
    try:
        run = subprocess.run([sys.executable, os.path.abspath(__file__), "--_worker", jpath],
                             capture_output=True, text=True, timeout=1200)
        if os.path.exists(job["result"]):
            rec.update(pickle.load(open(job["result"], "rb")))
        else:
            lines = [l for l in (run.stderr + run.stdout).splitlines() if "Error" in l or "ERROR" in l]
            rec["error"] = (lines[-1] if lines else run.stderr[-300:]).strip()[:500] or "the run died with no message"
    except subprocess.TimeoutExpired:
        rec["error"] = "compile or run exceeded 20 minutes"
    rec["compile_run_s"] = round(time.time() - t0, 1)
    if rec["correct"]:
        try:
            times, m = [], None
            for i in range(reps):
                d = os.path.join(work, f"p{i}")
                os.makedirs(d)
                m, _ = hwcheck.profile(os.path.join(work, "art", "kernel.neff"), d, core)
                times.append(m["total_time"] * 1e6)
            rec.update(time_us=statistics.median(times), runs_us=[round(t, 1) for t in times],
                       te_busy=m["tensor_engine_active_time_percent"],
                       dma_busy=m["dma_active_time_percent"],
                       bytes=m["dma_transfer_total_bytes"], transfers=m["dma_transfer_count"],
                       avg_transfer_bytes=m.get("dma_transfer_average_bytes"),
                       floor_bytes=(K * M + K * N + M * N) * 4,
                       compute_util=m.get("mfu_estimated_percent"),
                       bandwidth_util=m.get("mbu_estimated_percent"))
        except Exception as e:
            rec["error"] = f"correct on the chip, but profiling failed: {str(e)[:300]}"
    if keep is None:
        shutil.rmtree(work, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker")
    ap.add_argument("--kernel", default=os.path.join(HERE, "kernels", "matmul_blocked.py"))
    ap.add_argument("--size", default="2048", help="2048 for square, or K4096_M512_N12288")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--grid", help='knob settings "M,N,K;M,N,K;..." applied to --kernel')
    ap.add_argument("--out", help="append one JSON line per measurement")
    a = ap.parse_args()
    if a._worker:
        return _worker(a._worker)
    base = open(a.kernel).read()
    todo = [tuple(int(x) for x in c.split(",")) for c in a.grid.split(";")] if a.grid else [get_knobs(base)]
    for knobs in todo:
        rec = measure(set_knobs(base, *knobs), parse_shape(a.size), a.reps)
        rec["knobs"] = list(knobs)
        line = json.dumps(rec)
        print(line, flush=True)
        if a.out:
            open(a.out, "a").write(line + "\n")


if __name__ == "__main__":
    main()
