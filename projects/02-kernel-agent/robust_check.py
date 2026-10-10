#!/usr/bin/env python3
"""robust_check.py -- the robust, post-hoc check for a minimum-traffic kernel.

Judges with the same official checker internals (`nkibench.accept_case`,
`minimum_hbm_bytes`, the level gates), but on cases beyond the four optimization shapes:

  * optimization shapes at levels 5, 6, 7 (gates 1.60 / 1.25 / 1.05), seeds 0 and 1
  * declared held-out aligned shapes (traffic_cases.json), seeds 17, 29, 43 -- level-5 bar
  * declared ragged correctness shapes, seeds 7, 11, 23 -- numerics judged, traffic reported
  * declared hostile value families on the optimization shapes -- numerics judged, traffic reported

Runs the independent simulations across processes (`--jobs`), so one seat uses several cores.

    python3 robust_check.py --kernel winner.py --entry nki_matmul_fully_optimized_ \
        --out robust_results.json --jobs 6
"""
import argparse
import json
import multiprocessing as mp

import numpy as np

import nkibench

OPT_SHAPES = [dict(K=128, M=128, N=512), dict(K=256, M=256, N=1024),
              dict(K=512, M=128, N=512), dict(K=256, M=512, N=1024)]
HELDOUT_SHAPES = [dict(K=384, M=256, N=512), dict(K=128, M=384, N=1024),
                  dict(K=512, M=256, N=1536)]
RAGGED_SHAPES = [dict(K=1, M=1, N=1), dict(K=129, M=127, N=513),
                 dict(K=257, M=131, N=519)]
FAMILIES = ["zeros", "negative_mixed_sign", "repeated_rows",
            "alternating_signs_cancellation", "moderate_finite_magnitudes"]


def hostile_args(spec, family, seed):
    K, M, N = spec["K"], spec["M"], spec["N"]
    rng = np.random.default_rng(seed + 991)
    if family == "zeros":
        return (np.zeros((K, M), np.float32), np.zeros((K, N), np.float32))
    if family == "negative_mixed_sign":
        return (rng.uniform(-1.0, 1.0, (K, M)).astype(np.float32),
                rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32))
    if family == "repeated_rows":
        return (np.tile(rng.standard_normal(M), (K, 1)).astype(np.float32),
                np.tile(rng.standard_normal(N), (K, 1)).astype(np.float32))
    if family == "alternating_signs_cancellation":
        return (np.ones((K, M), np.float32),
                np.tile((((-1.0) ** np.arange(K))).astype(np.float32)[:, None], (1, N)))
    return ((rng.standard_normal((K, M)) * 100.0).astype(np.float32),
            (rng.standard_normal((K, N)) * 100.0).astype(np.float32))


def run_one(job):
    kind, level, spec, seed, family, kernel_path, entry = job
    tag = (f"{kind} K={spec['K']} M={spec['M']} N={spec['N']} seed={seed}"
           + (f" {family}" if family else ""))
    try:
        args = hostile_args(spec, family, seed) if family else \
            nkibench.make_inputs(spec, level, seed)[0]
        want = nkibench.LEVELS[level]["ref"](*args)
        kernel = nkibench.load_kernel(kernel_path, entry)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        got, counted = nkibench.simulate_and_count(kernel, args)
        ok, msg, checks = nkibench.accept_case(level, got, counted, args, before, want, 2e-2)
        floor = int(nkibench.minimum_hbm_bytes(args, want) or 0)
        b = int(counted.get("bytes") or 0)
        return dict(case=tag, kind=kind, level=level, seed=seed, ok=bool(ok),
                    bytes=b, floor=floor, waste=(round(b / floor, 4) if floor else None),
                    failure=msg, warnings=counted.get("warnings", []))
    except Exception as e:
        return dict(case=tag, kind=kind, level=level, seed=seed, ok=False, bytes=None,
                    floor=None, waste=None, failure=f"raised {type(e).__name__}: {e}",
                    warnings=[])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--out", default="robust_results.json")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-hostile", action="store_true")
    a = ap.parse_args()

    jobs = []
    for level in (5, 6, 7):
        for seed in (0, 1):
            for spec in OPT_SHAPES:
                jobs.append(("opt", level, spec, seed, None, a.kernel, a.entry))
    for seed in (17, 29, 43):
        for spec in HELDOUT_SHAPES:
            jobs.append(("heldout", 5, spec, seed, None, a.kernel, a.entry))
    for seed in (7, 11, 23):
        for spec in RAGGED_SHAPES:
            jobs.append(("ragged", 5, spec, seed, None, a.kernel, a.entry))
    if not a.skip_hostile:
        for spec in OPT_SHAPES:
            for family in FAMILIES:
                jobs.append(("hostile", 5, spec, 0, family, a.kernel, a.entry))

    with mp.Pool(a.jobs) as pool:
        results = pool.map(run_one, jobs)

    for r in results:
        status = "PASS" if r["ok"] else "fail"
        w = f"{r['waste']:.2f}x" if r["waste"] is not None else "?"
        print(f"{status}  {r['case']:<48} bytes={r['bytes']} floor={r['floor']} waste={w}"
              + (f"  [{str(r['failure'])[:90]}]" if r["failure"] else ""))
        for wmsg in r["warnings"]:
            print(f"      WARNING: {wmsg[:110]}")

    def count(kind, level=None):
        sel = [r for r in results if r["kind"] == kind
               and (level is None or r["level"] == level)]
        return sum(1 for r in sel if r["ok"]), len(sel)

    summary = {
        "kernel": a.kernel, "entry": a.entry,
        "opt_l5": count("opt", 5), "opt_l6": count("opt", 6), "opt_l7": count("opt", 7),
        "heldout": count("heldout"), "ragged": count("ragged"), "hostile": count("hostile"),
        "results": results,
    }
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("\nsummary (passed/total):")
    print(f"  optimization shapes: level5 {summary['opt_l5'][0]}/{summary['opt_l5'][1]} | "
          f"level6 {summary['opt_l6'][0]}/{summary['opt_l6'][1]} | "
          f"level7 {summary['opt_l7'][0]}/{summary['opt_l7'][1]}")
    print(f"  held-out aligned:    {summary['heldout'][0]}/{summary['heldout'][1]}   "
          f"(level-5 bar)")
    print(f"  ragged correctness:  {summary['ragged'][0]}/{summary['ragged'][1]}   "
          f"(level-5 bar; numerics judged)")
    print(f"  hostile values:      {summary['hostile'][0]}/{summary['hostile'][1]}   "
          f"(level-5 bar)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
