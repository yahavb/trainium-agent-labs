"""Headroom check: AWS's own tutorial matmul kernels, naive to fully optimised, on the real chip.

Answers one question before any agent is built: is there a speed gap on the CHIP between these
versions, or does the compiler already close it?   python sweep_tutorial.py [SIZE] [REPEATS]
"""
import json, os, pickle, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/projects/02-kernel-agent")
KERNELS = "/workspace/hwprofile/tutorial_matmul.py"
VARIANTS = ["nki_matmul_tiled_", "nki_matmul_hoist_load_", "nki_matmul_block_free_dimension_",
            "nki_matmul_fully_optimized_"]


def worker(name, size, art, out):
    os.environ.update(NEURON_RT_VISIBLE_CORES="0", NEURON_PLATFORM_TARGET_OVERRIDE="trn2",
                      NKI_ARTIFACTS_DIR=art, NKI_DEBUG_INFO="1")
    import numpy as np, importlib.util
    spec = importlib.util.spec_from_file_location("k", KERNELS)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    r = np.random.default_rng(0)
    lhsT = r.standard_normal((size, size)).astype(np.float32)
    rhs = r.standard_normal((size, size)).astype(np.float32)
    got = np.asarray(getattr(mod, name)(lhsT, rhs))
    want = lhsT.T @ rhs
    pickle.dump(dict(ok=bool(np.allclose(got, want, rtol=2e-2, atol=2e-2)),
                     err=float(np.abs(got - want).max())), open(out, "wb"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]); sys.exit()
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
    import hwcheck
    root = f"/workspace/hwprofile/sweep_{size}"
    rows = []
    for name in VARIANTS:
        work = os.path.join(root, name); art = os.path.join(work, "art")
        subprocess.run(["rm", "-rf", work]); os.makedirs(art)
        res = os.path.join(work, "res.pkl")
        p = subprocess.run([sys.executable, __file__, "--worker", name, str(size), art, res],
                           capture_output=True, text=True, timeout=1500)
        if not os.path.exists(res):
            err = [l for l in (p.stderr + p.stdout).splitlines() if "Error" in l or "ERROR" in l]
            print(f"{name:36s} FAILED: {(err[-1] if err else p.stderr[-200:])[:200]}"); continue
        r = pickle.load(open(res, "rb"))
        times, m = [], None
        for i in range(reps):
            d = os.path.join(work, f"p{i}"); os.makedirs(d)
            m, _ = hwcheck.profile(os.path.join(art, "kernel.neff"), d, 0)
            times.append(m["total_time"] * 1e6)
        row = dict(kernel=name, correct=r["ok"], time_us=times, median_us=statistics.median(times),
                   te_busy=m["tensor_engine_active_time_percent"], dma_busy=m["dma_active_time_percent"],
                   bytes=m["dma_transfer_total_bytes"], transfers=m["dma_transfer_count"],
                   mfu=m.get("mfu_estimated_percent"), mbu=m.get("mbu_estimated_percent"))
        rows.append(row)
        print(f"{name:36s} correct={r['ok']}  time {row['median_us']:9.1f} us (runs {[round(t,1) for t in times]})  "
              f"TE {100*row['te_busy']:5.1f}%  DMA {100*row['dma_busy']:5.1f}%  "
              f"bytes {row['bytes']:,} in {row['transfers']}  compute-util {100*(row['mfu'] or 0):.1f}%", flush=True)
    json.dump(rows, open(os.path.join(root, "sweep.json"), "w"), indent=1)
