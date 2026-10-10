#!/usr/bin/env python3
"""Measure a Trainium adapter and write a checker-compatible prediction.

An adapter is a Python file with this function:

    prepare(inputs: dict[str, numpy.ndarray], context: dict) -> object

The returned object must have ``run()``. It can also have ``synchronize()`` and
``close()``. ``run()`` returns either a prediction array or a dictionary with a
``prediction`` array. The adapter must keep compilation in ``prepare`` and must
make ``synchronize`` wait for all device work.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parents[1] / "fixtures" / "manifest.json",
    )
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--workload", choices=("single-step", "rollout"), default="single-step")
    parser.add_argument("--forecast-steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--precision", required=True)
    parser.add_argument(
        "--hardware",
        default=os.environ.get("TRAINIUM_HARDWARE", "aws-trainium"),
    )
    parser.add_argument(
        "--torch-profile",
        type=Path,
        default=None,
        help="opt-in: after timing, profile extra runs with torch.profiler into this "
        "directory. See PROFILING.md.",
    )
    parser.add_argument("--torch-profile-runs", type=int, default=3)
    return parser.parse_args()


def load_numpy() -> Any:
    extra_metrics = None
    prediction = None
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required. Run this script in the Samudra or Neuron environment."
        ) from error
    return np


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def load_adapter(path: Path) -> ModuleType:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("samudra_trainium_adapter", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "prepare", None)):
        raise ValueError("Adapter must define prepare(inputs, context)")
    return module


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def as_numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if args.forecast_steps < 1:
        raise ValueError("forecast-steps must be positive")
    if args.workload == "single-step" and args.forecast_steps != 1:
        raise ValueError("single-step workload requires --forecast-steps 1")

    np = load_numpy()
    manifest = load_json(args.manifest.resolve())
    prediction_key = manifest.get("prediction_key")
    coordinate_keys = manifest.get("coordinate_keys")
    if not isinstance(prediction_key, str) or not isinstance(coordinate_keys, list):
        raise ValueError("Manifest prediction and coordinate keys are invalid")

    with np.load(args.fixture.resolve(), allow_pickle=False) as archive:
        fixture = {key: archive[key] for key in archive.files}
    missing_coordinates = [key for key in coordinate_keys if key not in fixture]
    if missing_coordinates:
        raise ValueError(f"Fixture is missing coordinates: {missing_coordinates}")

    inputs = {
        key: value
        for key, value in fixture.items()
        if key not in {prediction_key, *coordinate_keys}
    }
    if not inputs:
        raise ValueError("Fixture has no candidate inputs")

    context = {
        "workload": args.workload,
        "forecast_steps": args.forecast_steps,
        "precision": args.precision,
        "hardware": args.hardware,
        "manifest": manifest,
    }
    adapter_module = load_adapter(args.adapter)

    # Wall-clock bounds of prepare(), so runners/profiling.py can find the
    # programs compiled here in the Neuron compile cache.
    compile_started_unix = time.time()
    compile_start = time.perf_counter()
    prepared = adapter_module.prepare(inputs, context)
    compile_seconds = time.perf_counter() - compile_start
    compile_finished_unix = time.time()
    run = getattr(prepared, "run", None)
    if not callable(run):
        raise ValueError("Prepared adapter must define run()")
    synchronize = getattr(prepared, "synchronize", lambda: None)
    close = getattr(prepared, "close", lambda: None)

    extra_metrics = None
    prediction = None
    try:
        warmup_start = time.perf_counter()
        output = None
        for _ in range(args.warmup):
            output = run()
            synchronize()
        warmup_seconds = time.perf_counter() - warmup_start

        latency_ms: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = run()
            synchronize()
            latency_ms.append((time.perf_counter() - start) * 1000)
        if output is None:
            raise RuntimeError("Adapter returned no output")
        if isinstance(output, dict):
            if prediction_key not in output:
                raise ValueError(f"Adapter output is missing {prediction_key!r}")
            prediction = as_numpy(output[prediction_key], np).copy()
        else:
            prediction = as_numpy(output, np).copy()
        if args.torch_profile is not None:
            # Runs after the timed loop, so profiler overhead never reaches
            # the reported latency.
            from profiling import torch_profile

            def profiled_step() -> None:
                run()
                synchronize()

            torch_profile(profiled_step, args.torch_profile_runs, args.torch_profile)
        adapter_metrics = getattr(prepared, "metrics", None)
        if callable(adapter_metrics):
            extra_metrics = adapter_metrics()
            if not isinstance(extra_metrics, dict):
                raise ValueError("Adapter metrics() must return a dictionary")
    finally:
        close()

    if prediction is None:
        raise RuntimeError("Adapter returned no prediction")
    if not np.isfinite(prediction).all():
        raise RuntimeError("Adapter returned NaN or infinite values")

    candidate_arrays = {prediction_key: prediction}
    candidate_arrays.update({key: fixture[key] for key in coordinate_keys})
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.candidate_output, **candidate_arrays)

    total_timed_seconds = sum(latency_ms) / 1000
    completed_steps = args.repeats * args.forecast_steps
    metrics = {
        "status": "passed",
        "workload": args.workload,
        "hardware": args.hardware,
        "precision": args.precision,
        "input_keys": sorted(inputs),
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "output_shape": list(prediction.shape),
        "compile_seconds": compile_seconds,
        "compile_started_unix": compile_started_unix,
        "compile_finished_unix": compile_finished_unix,
        "warmup_runs": args.warmup,
        "warmup_seconds": warmup_seconds,
        "timed_runs": args.repeats,
        "forecast_steps_per_run": args.forecast_steps,
        "timed_forecast_steps": completed_steps,
        "latency_ms": latency_ms,
        "median_ms": statistics.median(latency_ms),
        "p95_ms": percentile(latency_ms, 0.95),
        "steps_per_second": completed_steps / total_timed_seconds,
    }
    if extra_metrics is not None:
        metrics["adapter_metrics"] = extra_metrics
    if args.torch_profile is not None:
        metrics["torch_profile_dir"] = str(args.torch_profile)

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
