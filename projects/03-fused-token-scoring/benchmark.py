"""Tune independently, then retain device/host samples for fair comparisons."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import nki
from nrtpy import reset

from reference import make_inputs, score_reference, compare
from runtime import LoadedKernel, ROOT
from kernel import fused_score, separate_score
from baseline import aws_separate_score

KERNELS = {k.__name__: k for k in (fused_score, separate_score, aws_separate_score)}
SHAPES = {"A": (128, 8192), "B": (512, 32768), "C": (129, 8193), "D": (512, 151936)}
CONFIGS = ((64, 4096), (128, 4096), (128, 8192), (128, 16384))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def stats(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {"samples": len(values), "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(values.mean()),
            "std_ms": float(values.std()), "min_ms": float(values.min()), "max_ms": float(values.max())}


def environment():
    import nkilib.experimental.loss.cross_entropy as ce
    packages = {}
    for package in ("neuronx-cc", "numpy", "torch", "ml_dtypes"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pod": platform.node(), "python": platform.python_version(), "nki": nki.__version__,
            "packages": packages, "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
            "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
            "neuron_ls": subprocess.run(["neuron-ls"], capture_output=True, text=True).stdout,
            "aws_cross_entropy_sha256": hashlib.sha256(Path(ce.__file__).read_bytes()).hexdigest(),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}}


def tune(workloads, output):
    records = []
    selections = {}
    for label in workloads:
        logits, indices = make_inputs(*SHAPES[label], seed=2027)
        expected = score_reference(logits, indices)
        selections[label] = {}
        for name, kernel in KERNELS.items():
            choices = []
            for row_tile, vocab_tile in CONFIGS:
                record = {"workload": label, "kernel": name, "row_tile": row_tile, "vocab_tile": vocab_tile}
                try:
                    loaded = LoadedKernel(kernel, logits, indices, row_tile, vocab_tile)
                    record["errors"] = compare(loaded.run(), expected, logits.shape[1])
                    durations = loaded.measure(warmup=5, iterations=30)
                    record["device"] = stats(durations)
                    record["durations_ms"] = durations.tolist()
                    record["compile_seconds"] = loaded.compile_seconds
                    record["passed"] = True
                    choices.append(record)
                    del loaded
                except Exception as error:
                    record["passed"] = False
                    record["error"] = f"{type(error).__name__}: {error}"
                records.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "durations_ms"}), flush=True)
                # Release models/tensors, including SDK's own references.
                reset()
                save(output, {"selection_seed": 2027, "records": records, "selected": selections})
            if not choices:
                raise RuntimeError(f"No correct compilable config for {label}/{name}")
            winner = min(choices, key=lambda r: r["device"]["p50_ms"])
            selections[label][name] = {"row_tile": winner["row_tile"], "vocab_tile": winner["vocab_tile"]}
            save(output, {"selection_seed": 2027, "records": records, "selected": selections})


def benchmark(workloads, tuning, output, rounds, warmup, iterations):
    selections = json.loads(Path(tuning).read_text())["selected"]
    raw = []
    report = {"environment": environment(), "input_seed": 2026, "rounds": rounds,
              "warmup_per_round": warmup, "iterations_per_round": iterations,
              "timing": "nrtpy device nc_exec_running trace; host execute wall clock separately",
              "device_scope": "per exec_id: earliest PNC start to latest PNC finish using trace-synchronized device-event timestamps; two PNCs in LNC=2",
              "host_scope": "resident inputs/outputs; excludes transfers, host validation and compilation",
              "selected_configs": selections, "workloads": {}}
    for label in workloads:
        logits, indices = make_inputs(*SHAPES[label], seed=2026)
        expected = score_reference(logits, indices)
        loaded = {}
        validations = {}
        for name, kernel in KERNELS.items():
            config = selections[label][name]
            loaded[name] = LoadedKernel(kernel, logits, indices, **config)
            validations[name] = compare(loaded[name].run(), expected, logits.shape[1])
        samples = {name: {"device": [], "host": [], "rounds": []} for name in KERNELS}
        for round_id in range(rounds):
            order = list(KERNELS)
            order = order[round_id % len(order):] + order[:round_id % len(order)]
            for name in order:
                round_record = {"round": round_id}
                for mode in ("device", "host"):
                    values = loaded[name].measure(mode=mode, warmup=warmup, iterations=iterations)
                    if mode == "device":
                        save(output / "traces" / f"{label}-{name}-round{round_id}.json", loaded[name].last_device_trace)
                    samples[name][mode].extend(values.tolist())
                    round_record[mode] = stats(values)
                    for i, value in enumerate(values):
                        raw.append({"workload": label, "kernel": name, "round": round_id,
                                    "iteration": i, "mode": mode, "duration_ms": float(value)})
                samples[name]["rounds"].append(round_record)
                print(json.dumps({"workload": label, "kernel": name, **round_record}), flush=True)
        metrics = {name: {"device": stats(data["device"]), "host": stats(data["host"]),
                          "rounds": data["rounds"], "max_abs_error": validations[name],
                          "neff_sha256": hashlib.sha256(loaded[name].neff.read_bytes()).hexdigest(),
                          "compile_seconds_this_run": loaded[name].compile_seconds} for name, data in samples.items()}
        strongest = min(("separate_score", "aws_separate_score"), key=lambda name: metrics[name]["device"]["p50_ms"])
        before = metrics[strongest]["device"]["p50_ms"]
        after = metrics["fused_score"]["device"]["p50_ms"]
        report["workloads"][label] = {"shape": list(logits.shape), "metrics": metrics, "strongest_baseline": strongest,
                                      "speedup": before / after, "latency_reduction_percent": (1 - after / before) * 100,
                                      "meets_20_percent_target": after <= before * 0.8}
        save(output / "summary.json", report)
        save(output / "raw_samples.json", raw)
        save(output / "environment.json", report["environment"])
        print(json.dumps({"workload": label, "strongest_baseline": strongest, "speedup": before / after,
                          "latency_reduction_percent": (1 - after / before) * 100}), flush=True)
        del loaded
        reset()
    lines = ["# Measured results", "", "Device medians from real Trainium2 execution; see summary.json and raw_samples.json.", "",
             "| Workload | Shape | Strongest separate baseline | Before p50 (ms) | Fused p50 (ms) | Speedup | Reduction |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for label, record in report["workloads"].items():
        before = record["metrics"][record["strongest_baseline"]]["device"]["p50_ms"]
        after = record["metrics"]["fused_score"]["device"]["p50_ms"]
        lines.append(f"| {label} | {' × '.join(map(str, record['shape']))} | {record['strongest_baseline']} | {before:.6f} | {after:.6f} | {record['speedup']:.2f}x | {record['latency_reduction_percent']:.1f}% |")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--workloads", nargs="+", choices=list(SHAPES), default=["A", "B", "C"])
    parser.add_argument("--tuning", default="results/tuning.json")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tune:
        tune(args.workloads, args.tuning)
    else:
        benchmark(args.workloads, args.tuning, args.output, args.rounds, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
