# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Synchronized forward-call timings without data I/O or forecast metrics."""

import resource
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForwardTiming:
    """Wall time and host process CPU time for one completed model call."""

    wall_seconds: float
    host_cpu_seconds: float


def time_forward(
    forward: Callable[[], Any],
    synchronize: Callable[[], None],
    compilation_count: Callable[[], int] = lambda: 0,
    cpu_fallback_count: Callable[[], int] = lambda: 0,
) -> tuple[Any, ForwardTiming]:
    """Time dispatch through completion, rejecting compilation in the interval.

    Inputs and weights must already reside on the execution device. Synchronize
    before starting both clocks; synchronize after dispatch before stopping them.
    Host CPU time is process-wide and includes its worker threads, not accelerator
    compute time. Reading compilation counters stays outside the timed interval.
    """
    synchronize()
    compilations = compilation_count()
    fallbacks = cpu_fallback_count()
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    result = forward()
    synchronize()
    cpu_end = time.process_time_ns()
    wall_end = time.perf_counter_ns()
    if compilation_count() != compilations:
        raise RuntimeError("Compilation occurred during a measured forward call")
    if cpu_fallback_count() != fallbacks:
        raise RuntimeError("CPU fallback occurred during a measured forward call")
    return result, ForwardTiming(
        (wall_end - wall_start) / 1e9, (cpu_end - cpu_start) / 1e9
    )


def summarize_timings(timings: list[ForwardTiming]) -> dict[str, Any]:
    """Summarize actual per-call samples; never extrapolate a rollout duration."""
    if not timings:
        raise ValueError("At least one measured forward call is required")
    wall = np.asarray([item.wall_seconds for item in timings])
    cpu = np.asarray([item.host_cpu_seconds for item in timings])
    return {
        "calls": len(timings),
        "total_forward_wall_seconds": float(wall.sum()),
        "total_host_cpu_seconds": float(cpu.sum()),
        "mean_wall_seconds": float(wall.mean()),
        "median_wall_seconds": float(np.median(wall)),
        "p95_wall_seconds": float(np.percentile(wall, 95)),
        "min_wall_seconds": float(wall.min()),
        "max_wall_seconds": float(wall.max()),
        "calls_per_forward_second": float(len(wall) / wall.sum()),
        "mean_host_cpu_seconds": float(cpu.mean()),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "samples": [asdict(item) for item in timings],
    }


def compare_reports(cpu: dict[str, Any], neuron: dict[str, Any]) -> dict[str, Any]:
    """Compare matched workloads, rejecting misleading speedup denominators."""
    if cpu["backend"] != "cpu" or neuron["backend"] != "neuron":
        raise ValueError("Expected a CPU report followed by a Neuron report")
    fields = (
        "torch_version",
        "dtype",
        "cpu_threads",
        "checkpoint_sha256",
        "prepared_case_sha256",
        "input_prognostic_shape",
        "input_boundary_shape",
        "output_shape",
        "requested_range",
        "target_time_start",
        "target_time_end",
        "complete_eval_range",
        "forecast_timesteps",
        "warmup_calls",
        "timing_contract",
    )
    for field in fields:
        if cpu[field] != neuron[field]:
            raise ValueError(f"Cannot compare reports with different {field}")
    for section in ("single_call", "eval_range"):
        if cpu[section]["calls"] != neuron[section]["calls"]:
            raise ValueError(f"Cannot compare different {section} call counts")
    return {
        "single_call_median_speedup_cpu_over_neuron": cpu["single_call"][
            "median_wall_seconds"
        ]
        / neuron["single_call"]["median_wall_seconds"],
        "eval_forward_speedup_cpu_over_neuron": cpu["eval_range"][
            "total_forward_wall_seconds"
        ]
        / neuron["eval_range"]["total_forward_wall_seconds"],
        "cpu": cpu,
        "neuron": neuron,
    }


def main() -> None:
    """Write a comparison without rerunning models or scoring forecasts."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("neuron", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_reports(
        json.loads(args.cpu.read_text()), json.loads(args.neuron.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in comparison.items() if k not in ("cpu", "neuron")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
