"""
Sweep tiled-matmul configurations x shapes, time each on the device, append to a CSV.

    # correctness of every config on the CPU simulator (fast, run once first)
    python bench.py --check

    # the sweep; two shards in parallel on the two free cores
    NEURON_RT_VISIBLE_CORES=2 nohup python bench.py --shard 0/2 > bench0.log 2>&1 &
    NEURON_RT_VISIBLE_CORES=3 nohup python bench.py --shard 1/2 > bench1.log 2>&1 &

Re-running skips rows already in the CSV, so a crash or a killed shard just resumes.
Each config is generated as a real .py file under _gen/, because neuronxcc re-reads kernel
source with inspect.getsource.
"""

import argparse
import csv
import importlib
import itertools
import os
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
GEN = HERE / "_gen"

# --- search space -----------------------------------------------------------------------------
TILE_N = [128, 256, 512]          # moving free size (<= 512)
TILE_K = [64, 128]                # contraction / partition size (<= 128)
TILE_M = [128]                    # stationary free size (<= 128)
VARIANT = ["naive", "hoist"]      # naive: reload lhsT for every n; hoist: load lhsT once per m
SHAPES = [                        # (K, M, N), all multiples of 128 and 512
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (1024, 2048, 1024),
    (1024, 1024, 2048),
    (2048, 2048, 2048),
]
DTYPE = "bf16"

HEADER = """import neuronxcc.nki as nki  # noqa: F401
import neuronxcc.nki.language as nl

TILE_M, TILE_K, TILE_N = {tm}, {tk}, {tn}

"""

NAIVE = """def kernel(lhsT, rhs, result):
  K, M = lhsT.shape
  _, N = rhs.shape
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        a = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        b = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        res_psum += nl.matmul(a, b, transpose_x=True)
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], value=res_sb)
"""

HOIST = """def kernel(lhsT, rhs, result):
  K, M = lhsT.shape
  _, N = rhs.shape
  for m in nl.affine_range(M // TILE_M):
    a_tiles = nl.ndarray((K // TILE_K, nl.par_dim(TILE_K), TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
    for k in nl.affine_range(K // TILE_K):
      a_tiles[k] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        b = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        res_psum += nl.matmul(a_tiles[k], b, transpose_x=True)
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], value=res_sb)
"""

FIELDS = ["variant", "tile_m", "tile_k", "tile_n", "K", "M", "N", "dtype",
          "p0_us", "p50_us", "p90_us", "p99_us", "compile_run_s", "status"]


def config_name(variant, tm, tk, tn):
    return f"k_{variant}_m{tm}_k{tk}_n{tn}"


def load_kernel(variant, tm, tk, tn, ret=False):
    """ret=False: writes into a `result` argument (what the simulator needs).
    ret=True: allocates and returns `result` (what nki.benchmark needs). Same loop body."""
    GEN.mkdir(exist_ok=True)
    # pid in the name: parallel shards must not write/import the same file at the same time
    name = config_name(variant, tm, tk, tn) + ("_ret" if ret else "") + f"_p{os.getpid()}"
    path = GEN / f"{name}.py"
    body = NAIVE if variant == "naive" else HOIST
    if ret:
        body = body.replace("def kernel(lhsT, rhs, result):", "def kernel(lhsT, rhs):")
        body = body.replace("  _, N = rhs.shape\n",
                            "  _, N = rhs.shape\n"
                            "  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n")
        body += "  return result\n"
    path.write_text(HEADER.format(tm=tm, tk=tk, tn=tn) + body)
    if str(GEN) not in sys.path:
        sys.path.insert(0, str(GEN))
    return importlib.import_module(name).kernel


def np_dtype():
    if DTYPE == "fp32":
        return np.float32
    from ml_dtypes import bfloat16
    return bfloat16


def configs():
    return list(itertools.product(VARIANT, TILE_M, TILE_K, TILE_N))


def check_all():
    """Every config on the CPU simulator at a small shape, against NumPy."""
    import neuronxcc.nki as nki
    K, M, N = 256, 256, 512
    rng = np.random.default_rng(0)
    a = rng.standard_normal((K, M)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    expect = a.T @ b
    sim = (lambda k, *x: nki.simulate_kernel(k, *x)) if hasattr(nki, "simulate_kernel") \
        else (lambda k, *x: nki.simulate(k)(*x))
    for cfg in configs():
        name = config_name(*cfg)
        try:
            out = np.zeros((M, N), np.float32)
            sim(load_kernel(*cfg), a, b, out)
            err = np.abs(out - expect).max() / np.abs(expect).max()
            print(f"  {name:28s} rel err {err:.1e}  {'OK' if err < 1e-3 else 'WRONG'}")
            if os.environ.get("CHECK_DEVICE") and cfg == configs()[0]:
                dev = np.zeros((M, N), np.float32)
                nki.baremetal(load_kernel(*cfg))(a, b, dev)
                derr = np.abs(dev - expect).max() / np.abs(expect).max()
                print(f"  {name:28s} ON DEVICE rel err {derr:.1e}  {'OK' if derr < 1e-3 else 'WRONG'}")
        except Exception as e:
            print(f"  {name:28s} FAILED {type(e).__name__}: {str(e)[:120]}")


def done_keys(path):
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["variant"], r["tile_m"], r["tile_k"], r["tile_n"], r["K"], r["M"], r["N"])
                for r in csv.DictReader(f) if r["status"] == "ok"}


def sweep(shard, nshards, out_path):
    import neuronxcc.nki as nki
    jobs = [(cfg, shp) for cfg in configs() for shp in SHAPES]
    jobs = jobs[shard::nshards]
    done = set().union(*(done_keys(q) for q in HERE.glob("results_*.csv")))  # any shard's rows
    new_file = not out_path.exists()
    dt = np_dtype()
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, ((variant, tm, tk, tn), (K, M, N)) in enumerate(jobs):
            key = (variant, str(tm), str(tk), str(tn), str(K), str(M), str(N))
            if key in done:
                continue
            row = dict(variant=variant, tile_m=tm, tile_k=tk, tile_n=tn, K=K, M=M, N=N, dtype=DTYPE)
            t0 = time.perf_counter()
            try:
                bench = nki.benchmark(warmup=5, iters=20)(load_kernel(variant, tm, tk, tn, ret=True))
                bench(np.zeros((K, M), dt), np.zeros((K, N), dt))
                lat = bench.benchmark_result.full_results["latency"]
                row.update(p0_us=lat["0"], p50_us=lat["50"], p90_us=lat["90"], p99_us=lat["99"],
                           status="ok")
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}: {str(e)[:150]}".replace("\n", " ")
                traceback.print_exc()
            row["compile_run_s"] = round(time.perf_counter() - t0, 1)
            w.writerow(row)
            f.flush()
            print(f"[{i + 1}/{len(jobs)}] {config_name(variant, tm, tk, tn)} K{K} M{M} N{N}: "
                  f"p50 {row.get('p50_us')} us  ({row['compile_run_s']} s)  {row['status'][:60]}",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="CPU-simulator correctness of every config")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shapes", nargs="+", default=None, metavar="KxMxN",
                    help="override the shape list, e.g. --shapes 4096x1024x1024 4096x512x512")
    ap.add_argument("--one", nargs=7, metavar=("VARIANT", "TM", "TK", "TN", "K", "M", "N"),
                    help="time a single config/shape (used to verify a search pick)")
    a = ap.parse_args()
    os.chdir(HERE)
    if a.one:
        import neuronxcc.nki as nki
        v, *nums = a.one
        tm, tk, tn, K, M, N = (int(x) for x in nums)
        dt = np_dtype()
        bench = nki.benchmark(warmup=5, iters=20)(load_kernel(v, tm, tk, tn, ret=True))
        bench(np.zeros((K, M), dt), np.zeros((K, N), dt))
        lat = bench.benchmark_result.full_results["latency"]
        print(f"MEASURED {config_name(v, tm, tk, tn)} K{K} M{M} N{N}: p50 {lat['50']} us")
    elif a.check:
        check_all()
    else:
        if a.shapes:
            SHAPES[:] = [tuple(int(v) for v in sh.lower().split("x")) for sh in a.shapes]
        i, n = (int(x) for x in a.shard.split("/"))
        sweep(i, n, pathlib.Path(a.out or f"results_{i}.csv"))
