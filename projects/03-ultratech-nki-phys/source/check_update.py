"""Validate fixed updates, logging update correctness separately from QP convergence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
from contact import build_fixture, diagnostics, oracle
from update_reference import fixed_steps, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "simulate", "device"), default="simulate")
    parser.add_argument("--contacts", nargs="+", type=int, default=[8, 33])
    parser.add_argument("--steps", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.contacts + args.steps + [args.seeds, args.repeats]) < 1:
        parser.error("Counts must be positive")
    if max(args.contacts) > 128:
        parser.error("Initial kernel supports at most 128 contacts")
    if args.backend == "device" and not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        parser.error("Set NEURON_RT_VISIBLE_CORES to confirmed available cores")
    executor = fixed_steps
    if args.backend != "cpu":
        import nki
        from nki_contact import contact_steps
        executor = nki.simulate(contact_steps) if args.backend == "simulate" else contact_steps
    run_id = uuid.uuid4().hex
    out = Path("data") / f"update-{args.backend}-{run_id}"
    out.mkdir(parents=True)
    log_path = Path("data/update-attempts.jsonl")
    source = "update_reference.py" if args.backend == "cpu" else "nki_contact.py"
    digest = hashlib.sha256(Path(__file__).with_name(source).read_bytes()).hexdigest()
    results = []
    with log_path.open("a") as log:
        for scene in ("plane", "pairs", "stack"):
            for count in args.contacts:
                for seed in range(args.seeds):
                    fixture = build_fixture(scene, count, seed)
                    reference = oracle(fixture["A"], fixture["b"])
                    for steps in args.steps:
                        entry = dict(run_id=run_id, backend=args.backend, scene=scene, contacts=count,
                                     seed=seed, steps=steps, source_sha256=digest, score=0)
                        print(f"[{args.backend}] {scene} C={count} seed={seed} steps={steps}", flush=True)
                        try:
                            inputs = prepare(fixture)
                            originals = [x.copy() for x in inputs[:3]]
                            expected = fixed_steps(*inputs, steps)
                            start = time.perf_counter()
                            actual = np.asarray(executor(*inputs, steps)).copy()
                            entry["first_call_seconds_including_possible_compilation"] = time.perf_counter() - start
                            if actual.shape != expected.shape or not np.isfinite(actual).all():
                                raise ValueError("Wrong output shape or nonfinite values")
                            error = float(np.max(np.abs(actual - expected)))
                            limit = 1e-4 * max(1.0, float(np.max(np.abs(expected))))
                            entry["update_max_absolute_error"] = error
                            entry["update_error_limit"] = limit
                            padding_ok = bool(np.all(np.abs(actual[count:]) <= 1e-6))
                            input_ok = all(np.array_equal(x, y) for x, y in zip(inputs[:3], originals))
                            entry.update(padding_ok=padding_ok, inputs_unchanged=input_ok,
                                         score=int(error <= limit and padding_ok and input_ok))
                            entry["physics_checker"] = diagnostics(fixture["A"], fixture["b"],
                                                                   actual[:count, 0], reference, fixture)
                            entry["physics_score"] = int(entry["physics_checker"]["passed"])
                            times = []
                            for _ in range(args.repeats):
                                start = time.perf_counter()
                                repeated = np.asarray(executor(*inputs, steps)).copy()
                                times.append(time.perf_counter() - start)
                                if not np.isfinite(repeated).all() or repeated.shape != expected.shape or not np.allclose(repeated, expected, rtol=0, atol=limit):
                                    entry["score"] = 0
                                if not all(np.array_equal(x, y) for x, y in zip(inputs[:3], originals)):
                                    entry["score"] = 0
                            entry["host_call_seconds"] = times
                            entry["median_host_call_seconds"] = float(np.median(times))
                        except Exception:
                            entry["error"] = traceback.format_exc()
                        results.append(entry)
                        log.write(json.dumps(entry, allow_nan=False) + "\n")
                        log.flush()
                        print(f"  update score={entry['score']} physics score={entry.get('physics_score', 'unavailable')}", flush=True)
                        if "error" in entry:
                            print(entry["error"], flush=True)
    passed = sum(r["score"] for r in results)
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    note = f"""# Fixed-Update Validation

Run: {run_id}; backend: {args.backend}; source SHA256: {digest}.
Workloads: {len(results)}; post-first-call repeats: {args.repeats}.
Update equivalence passes: {passed}/{len(results)}.
Each attempt records host-call timing samples and separate full-physics scores.

Score 1 means fixed-update agreement, finite output, zero padded impulses and
unchanged inputs. It does not mean a converged contact solve. Many fixed-step
stack outputs are expected to fail the full physics checker.
First-call timings may include compilation. Other timings include host overhead
and materialization; CPU and simulator timings are not device performance.
This smoke test has insufficient repetitions for a performance or speedup claim.
Detailed records: results.json; append-only attempts: {log_path}.
"""
    (out / "run-note.md").write_text(note)
    print(f"Update equivalence: {passed}/{len(results)}; artifacts: {out}", flush=True)
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
