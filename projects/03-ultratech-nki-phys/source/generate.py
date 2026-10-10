"""Generate development fixtures only; certify every FP64 oracle answer."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import uuid

# Each worker owns one CPU thread; otherwise BLAS can oversubscribe the pod quota.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy
from contact import build_fixture, diagnostics, oracle, projected_gradient, velocity_error_bound


def available_workers():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return min(8, count)


def evaluate(task):
    scene, count, seed, out, runs, stopping = task
    filename = f"{scene}-c{count}-s{seed}.npz"
    record = dict(file=filename, scene=scene, contacts=count, seed=seed, attempts=[])
    try:
        fixture = build_fixture(scene, count, seed)
        reference = oracle(fixture["A"], fixture["b"])
        proof = diagnostics(fixture["A"], fixture["b"], reference, reference, fixture)
        record["oracle"] = proof
        if not proof["passed"] or proof["residual"] > 1e-9:
            raise RuntimeError(f"Uncertified oracle: {proof}")
        record["condition_number"] = float(np.linalg.cond(fixture["A"]))
        np.savez_compressed(Path(out) / filename, **fixture, reference_impulses=reference)
        for repeat in range(runs):
            start = time.perf_counter()
            candidate, iterations = projected_gradient(fixture["A"], fixture["b"],
                                                       fixture=fixture if stopping == "velocity-bound" else None)
            elapsed = time.perf_counter() - start
            check = diagnostics(fixture["A"], fixture["b"], candidate, reference, fixture)
            record["attempts"].append(dict(repeat=repeat, score=int(check["passed"]),
                                           checker=check, iterations=iterations,
                                           stopping=stopping,
                                           velocity_error_bound=velocity_error_bound(fixture["A"], fixture["b"], candidate, fixture),
                                           at_iteration_limit=iterations == 4096,
                                           cpu_solve_with_setup_seconds=elapsed))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["attempts"].append(dict(score=0, error=record["error"]))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/development"))
    parser.add_argument("--contacts", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=available_workers())
    parser.add_argument("--stopping", choices=["residual", "velocity-bound"], default="velocity-bound")
    parser.add_argument("--attempt-log", type=Path, default=Path("data/cpu-attempts.jsonl"))
    args = parser.parse_args()
    if args.seeds < 1 or args.runs < 1 or args.workers < 1 or min(args.contacts) < 1:
        parser.error("Seeds, runs, workers and contact counts must be positive")
    if len(args.contacts) != len(set(args.contacts)):
        parser.error("Contact counts must be unique")
    args.out.mkdir(parents=True, exist_ok=True)
    args.attempt_log.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    source_hash = hashlib.sha256(Path(__file__).with_name("contact.py").read_bytes()).hexdigest()
    tasks = [(scene, count, seed, str(args.out), args.runs, args.stopping)
             for scene in ("plane", "pairs", "stack")
             for count in args.contacts for seed in range(args.seeds)]
    start = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.attempt_log.open("a") as log:
        for record in pool.map(evaluate, tasks):
            records.append(record)
            for attempt in record["attempts"]:
                entry = dict(run_id=run_id, candidate="cpu_fp32_projected_gradient",
                             source_sha256=source_hash, scene=record["scene"], contacts=record["contacts"],
                             seed=record["seed"], fixture=record["file"], **attempt)
                log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
    manifest = dict(schema=1, split="development", provenance="analytical sphere-contact snapshots",
                    units="normalized mass, length, time", epsilon=1e-3, restitution=0,
                    stabilization=0, numpy=np.__version__, scipy=scipy.__version__,
                    python=platform.python_version(), host=platform.node(), source_sha256=source_hash,
                    run_id=run_id, workers=args.workers, blas_threads=1, runs=args.runs,
                    stopping=args.stopping, fixtures=records)
    manifest["schema"] = 2
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    attempts = [a for r in records for a in r["attempts"]]
    passed = sum(a["score"] for a in attempts)
    times = [a["cpu_solve_with_setup_seconds"] for a in attempts if "cpu_solve_with_setup_seconds" in a]
    per_run_passes = [sum(a["score"] for a in attempts if a.get("repeat") == repeat)
                      for repeat in range(args.runs)]
    report = ["# CPU Development Run", "",
              f"Run ID: {run_id}. Host: {platform.node()}. Python: {platform.python_version()}.",
              f"NumPy {np.__version__}; SciPy {scipy.__version__}. Source SHA256: {source_hash}.", "",
              f"Work: {len(records)} development contact fixtures, {args.runs} repeats each.",
              f"Execution: {args.workers} worker processes, one BLAS thread each.",
              f"Stopping rule: {args.stopping}; iteration limit: 4096; FP64 convergence checks.",
              f"Result: {passed}/{len(attempts)} scored attempts passed all numerical gates.",
              f"Pass counts across repeats: {per_run_passes}; range {min(per_run_passes)}-{max(per_run_passes)}.",
              f"Generation and evaluation wall time: {time.perf_counter() - start:.3f} seconds."]
    if times:
        report.append(f"CPU solve time including eigenvalue setup: median {np.median(times):.6f}s; "
                      f"p95 {np.percentile(times, 95):.6f}s; min-max {min(times):.6f}-{max(times):.6f}s.")
    report += ["", "Checker: feasibility >= -1e-6; projected residual <= 1e-4; objective discrepancy <= 1e-5;",
               "normalized velocity error <= 1e-4; finite output and correct shape required.",
               f"Attempt log: {args.attempt_log}. Detailed results: manifest.json.", "",
               "These are deterministic CPU baseline attempts, not LLM-generated optimization attempts.",
               "Timing under concurrent workers is a throughput diagnostic, not an isolated latency benchmark.",
               "No Trainium physics kernel, MuJoCo rollout, training result or speedup is claimed."]
    (args.out / "run-note.md").write_text("\n".join(report) + "\n")
    print(f"Saved {sum('error' not in r for r in records)} certified fixtures to {args.out}")
    print(f"FP32 CPU baseline: {passed}/{len(attempts)} scored attempts pass all gates")
    print(f"Workers: {args.workers}; repeats: {args.runs}; attempt log: {args.attempt_log}")
    print(f"One-page run note: {args.out / 'run-note.md'}")
    print("These are contact snapshots, not MuJoCo exports or validated rollouts.")
    return int(any("error" in record for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
