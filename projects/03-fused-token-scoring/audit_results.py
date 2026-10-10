"""Offline evidence check: source hashes, raw timings, traces and reported gains."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def percentile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    lower = int(p)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (p - lower)


def require_close(actual, expected):
    if not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-10):
        raise AssertionError(f"Reported {actual}, reconstructed {expected}")


def audit(root, results):
    report = json.loads((results / "summary.json").read_text())
    raw = json.loads((results / "raw_samples.json").read_text())
    for name, expected in report["environment"]["source_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Source changed since measurement: {name}")
    buckets = {}
    for sample in raw:
        key = (sample["workload"], sample["kernel"], sample["mode"], sample["round"])
        buckets.setdefault(key, []).append(sample)
    for label, workload in report["workloads"].items():
        for kernel, measurements in workload["metrics"].items():
            for mode in ("device", "host"):
                values = []
                for r in range(report["rounds"]):
                    samples = sorted(buckets[(label, kernel, mode, r)], key=lambda x: x["iteration"])
                    if [s["iteration"] for s in samples] != list(range(report["iterations_per_round"])):
                        raise AssertionError("Missing or duplicate measured iteration")
                    values.extend(s["duration_ms"] for s in samples)
                    if mode == "device":
                        events = json.loads((results / "traces" / f"{label}-{kernel}-round{r}.json").read_text())
                        starts, executions = {}, {}
                        for e in events:
                            if e["phase"] == "start":
                                starts[e["tracking_id"]] = e
                            else:
                                start = starts.pop(e["tracking_id"])
                                group = executions.setdefault(start["data"]["exec_id"], [])
                                group.append((start["timestamp_ns"], e["timestamp_ns"], start["data"]["device_core_idx"]))
                        if starts or len(executions) != len(samples):
                            raise AssertionError("Incomplete execution trace")
                        for sample, intervals in zip(samples, (executions[i] for i in sorted(executions))):
                            if len(intervals) != 2 or len({x[2] for x in intervals}) != 2:
                                raise AssertionError("Expected two distinct physical cores")
                            span = (max(x[1] for x in intervals) - min(x[0] for x in intervals)) / 1e6
                            require_close(sample["duration_ms"], span)
                require_close(measurements[mode]["p50_ms"], median(values))
                require_close(measurements[mode]["p95_ms"], percentile(values, 0.95))
                if measurements[mode]["samples"] != len(values):
                    raise AssertionError("Sample count disagrees with report")
        best = min(("separate_score", "aws_separate_score"),
                   key=lambda k: workload["metrics"][k]["device"]["p50_ms"])
        if workload["strongest_baseline"] != best:
            raise AssertionError("Comparison did not use the fastest baseline")
        before = workload["metrics"][best]["device"]["p50_ms"]
        after = workload["metrics"]["fused_score"]["device"]["p50_ms"]
        require_close(workload["speedup"], before / after)
        require_close(workload["latency_reduction_percent"], (1 - after / before) * 100)
    print(f"PASS: source hashes, {len(raw)} raw timing samples, all physical-core traces and reported speedups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    audit(Path(__file__).resolve().parent, args.results)
