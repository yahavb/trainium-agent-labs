#!/usr/bin/env python3
"""Opt-in profiling utilities for Samudra attempts.

Library use (inside a runner or an adapter):

    from profiling import StageTimer, xla_synchronize

    timer = StageTimer(synchronize=xla_synchronize)
    with timer.time("model", sync=True):
        output = model(inputs)
    timer.write_json(path)

Command-line use, after ``trainium_runner.py`` has exited:

    # Profile the program compiled during a runner call on one NeuronCore.
    python runners/profiling.py neuron \
        --metrics-json runs/my-attempt/metrics.json \
        --out-dir runs/my-attempt/neuron-profile --core 2

    # Re-summarize an existing neuron-explorer summary.
    python runners/profiling.py summarize runs/my-attempt/neuron-profile/summary.json

See PROFILING.md for setup and for how to read the results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_NEURON_CACHE = Path("/var/tmp/neuron-compile-cache")


# --------------------------------------------------------------------------
# Stage timing
# --------------------------------------------------------------------------


def xla_synchronize() -> None:
    """Wait until queued torch-xla (Trainium) work has finished.

    torch-xla is lazy: a model call only records work, and the device runs it
    later. A timer that stops without this sync measures the recording, not
    the compute. Does nothing when torch-xla has not been imported.
    """
    if "torch_xla" in sys.modules:
        import torch_xla
        import torch_xla.core.xla_model as xm

        torch_xla.sync()
        xm.wait_device_ops()


class StageTimer:
    """Accumulates wall-clock durations per named stage.

    Each ``time(name)`` call appends one duration to ``name``. Summaries report
    the median with and without the first call, because the first call carries
    warm-up and, on Trainium, compilation.
    """

    def __init__(self, synchronize: Callable[[], None] = xla_synchronize) -> None:
        self.synchronize = synchronize
        self.durations: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def time(self, name: str, sync: bool = False) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                self.synchronize()
            self.durations[name].append(time.perf_counter() - start)

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, values in self.durations.items():
            rest = values[1:]
            out[name] = {
                "count": len(values),
                "total_s": sum(values),
                "median_s": statistics.median(values),
                "first_s": values[0],
                "median_excl_first_s": statistics.median(rest) if rest else None,
            }
        return out

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "durations_s": dict(self.durations)}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# torch.profiler (host side)
# --------------------------------------------------------------------------


def torch_profile(
    step: Callable[[], Any], runs: int, out_dir: Path, row_limit: int = 30
) -> Path:
    """Profile ``runs`` calls of ``step`` with torch.profiler and write reports.

    Writes ``torch_ops.txt`` (top operators by self CPU time),
    ``torch_trace.json`` (open in https://ui.perfetto.dev) and
    ``torch_stacks.txt`` (input for a flame graph tool such as speedscope).

    On CPU this is a per-operator breakdown of the model. On Trainium it only
    sees the host: graph recording and waiting. Use the Neuron profile for time
    spent on the device. Never quote timings from a profiled run.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    out_dir.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        # Without verbose=True, export_stacks writes an empty file.
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        for _ in range(runs):
            step()
    prof.export_chrome_trace(str(out_dir / "torch_trace.json"))
    prof.export_stacks(str(out_dir / "torch_stacks.txt"), "self_cpu_time_total")
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cpu_time_total", row_limit=row_limit
    )
    ops = out_dir / "torch_ops.txt"
    ops.write_text(table + "\n", encoding="utf-8")
    return ops


# --------------------------------------------------------------------------
# Neuron device profile
# --------------------------------------------------------------------------


def neuron_cache_dir() -> Path:
    url = os.environ.get("NEURON_COMPILE_CACHE_URL", "")
    if url and "://" not in url:
        return Path(url)
    return DEFAULT_NEURON_CACHE


def find_neffs(cache_dir: Path, newer_than: float | None = None) -> list[Path]:
    """Return compiled programs (*.neff) in the cache, largest first.

    ``newer_than`` is a Unix time. Only programs written at or after it are
    returned, so a runner's compile window selects the programs it built.
    """
    found = []
    for path in cache_dir.rglob("*.neff"):
        mtime = path.stat().st_mtime
        if newer_than is None or mtime >= newer_than:
            found.append(path)
    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)


def load_neuron_summary(path: Path) -> dict[str, Any]:
    """Load ``neuron-explorer view --output-format summary-json`` output.

    The file holds one object per profiling session. This returns the first.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"No profiling session in {path}")
    session = next(iter(data.values()))
    if not isinstance(session, dict):
        raise ValueError(f"Unexpected summary layout in {path}")
    return session


# Engine rows in the order the report prints them.
_ENGINES = [
    ("DMA (memory transfers)", "dma_active_time"),
    ("Tensor engine (matmul / conv)", "tensor_engine_active_time"),
    ("Sync engine", "sync_engine_active_time"),
    ("Scalar engine (activations)", "scalar_engine_active_time"),
    ("Vector engine (element-wise, norms)", "vector_engine_active_time"),
    ("GpSimd engine", "gpsimd_engine_active_time"),
]


def key_metrics(session: dict[str, Any]) -> dict[str, Any]:
    """Pick the numbers that explain where device time goes."""
    total_s = float(session["total_time"])
    spill_save = int(session.get("spill_save_bytes", 0))
    spill_reload = int(session.get("spill_reload_bytes", 0))
    real_bytes = int(session.get("inputs_and_weights_size_bytes", 0))
    hbm_bytes = int(session.get("hbm_read_bytes", 0)) + int(session.get("hbm_write_bytes", 0))
    intensity = session.get("mm_arithmetic_intensity")
    ridge = session.get("peak_flops_bandwidth_ratio")
    engines = {
        label: {
            "active_ms": 1000 * float(session.get(key, 0.0)),
            "fraction_of_pass": float(session.get(key, 0.0)) / total_s if total_s else 0.0,
        }
        for label, key in _ENGINES
    }
    bound = None
    if intensity is not None and ridge:
        bound = "memory" if float(intensity) < float(ridge) else "compute"
    return {
        "pass_ms": 1000 * total_s,
        "busy_fraction": session.get("total_active_time_percent"),
        "engines": engines,
        "mfu_fraction": session.get("mfu_estimated_percent"),
        "mbu_fraction": session.get("mbu_estimated_percent"),
        "arithmetic_intensity": intensity,
        "balance_point": ridge,
        "bound": bound,
        "spill_save_gb": spill_save / 1e9,
        "spill_reload_gb": spill_reload / 1e9,
        "inputs_and_weights_gb": real_bytes / 1e9,
        "hbm_traffic_gb": hbm_bytes / 1e9,
        "spill_fraction_of_hbm": (spill_save + spill_reload) / hbm_bytes if hbm_bytes else None,
        "compiler_version": session.get("compiler_version"),
        "runtime_version": session.get("runtime_version"),
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.0f}%"


def _ms(value: float) -> str:
    return f"{value:.1f}" if value >= 1 else f"{value:.4f}"


def _gb(value: float) -> str:
    return f"{value:.2f} GB" if value >= 0.01 else f"{1000 * value:.2f} MB"


def render_markdown(metrics: dict[str, Any], title: str = "Neuron device profile") -> str:
    lines = [
        f"# {title}",
        "",
        f"One pass on the device: {_ms(metrics['pass_ms'])} ms. "
        f"Busy: {_pct(metrics['busy_fraction'])} of the pass.",
        "",
        "Engines run in parallel, so fractions overlap and do not sum to 100%.",
        "",
        "| Engine | Active ms | Share of pass |",
        "| --- | ---: | ---: |",
    ]
    for label, row in metrics["engines"].items():
        lines.append(f"| {label} | {_ms(row['active_ms'])} | {_pct(row['fraction_of_pass'])} |")
    intensity = metrics["arithmetic_intensity"]
    ridge = metrics["balance_point"]
    lines += [
        "",
        "| Indicator | Value |",
        "| --- | ---: |",
        f"| Compute utilization (MFU) | {_pct(metrics['mfu_fraction'])} |",
        f"| Bandwidth utilization (MBU) | {_pct(metrics['mbu_fraction'])} |",
        f"| Arithmetic intensity / balance point | "
        f"{'n/a' if intensity is None else f'{float(intensity):.1f}'} / "
        f"{'n/a' if ridge is None else f'{float(ridge):.1f}'} |",
        f"| Bound by | {metrics['bound'] or 'n/a'} |",
        f"| Spill save + reload per pass | "
        f"{_gb(metrics['spill_save_gb'])} + {_gb(metrics['spill_reload_gb'])} |",
        f"| Inputs + weights | {_gb(metrics['inputs_and_weights_gb'])} |",
        f"| Spill share of HBM traffic | {_pct(metrics['spill_fraction_of_hbm'])} |",
        "",
        f"Compiler {metrics['compiler_version']}, runtime {metrics['runtime_version']}.",
        "",
    ]
    return "\n".join(lines)


def _run(command: list[str], env: dict[str, str], stdout: Any = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True, stdout=stdout)


def neuron_profile(neff: Path, out_dir: Path, core: str, explorer: str) -> dict[str, Any]:
    """Capture a device profile of ``neff`` on ``core`` and summarize it.

    Writes ``model.neff`` (a copy), ``profile.ntff`` (raw, large),
    ``summary.json`` (all metrics), ``metrics.json`` (key metrics) and
    ``summary.md`` (a readable table) into ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_neff = out_dir / "model.neff"
    shutil.copy2(neff, local_neff)
    ntff = out_dir / "profile.ntff"
    env = dict(os.environ, NEURON_RT_VISIBLE_CORES=core)
    _run([explorer, "capture", "-n", str(local_neff), "-s", str(ntff)], env)
    summary_path = out_dir / "summary.json"
    # summary-json prints to stdout; the tool rejects --output-file for it.
    with summary_path.open("w", encoding="utf-8") as handle:
        _run(
            [explorer, "view", "-n", str(local_neff), "-s", str(ntff),
             "--output-format", "summary-json"],
            env,
            stdout=handle,
        )
    metrics = key_metrics(load_neuron_summary(summary_path))
    metrics["neff"] = str(neff)
    metrics["neff_bytes"] = neff.stat().st_size
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(metrics), encoding="utf-8")
    return metrics


def _compile_window_start(metrics_json: Path) -> float:
    data = json.loads(metrics_json.read_text(encoding="utf-8"))
    started = data.get("compile_started_unix")
    if started is None:
        raise SystemExit(
            f"{metrics_json} has no compile_started_unix; rerun trainium_runner.py "
            "from this branch, or pass --neff"
        )
    return float(started)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    neuron = sub.add_parser("neuron", help="capture and summarize a device profile")
    source = neuron.add_mutually_exclusive_group(required=True)
    source.add_argument("--neff", type=Path, help="compiled program to profile")
    source.add_argument("--metrics-json", type=Path,
                        help="runner metrics; profiles the largest program it compiled")
    neuron.add_argument("--cache-dir", type=Path, default=None)
    neuron.add_argument("--out-dir", type=Path, required=True)
    neuron.add_argument("--core", default=os.environ.get("NEURON_RT_VISIBLE_CORES", "2"))
    neuron.add_argument("--explorer", default="neuron-explorer")

    summarize = sub.add_parser("summarize", help="summarize an existing summary.json")
    summarize.add_argument("summary_json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "summarize":
        print(render_markdown(key_metrics(load_neuron_summary(args.summary_json))))
        return 0

    neff = args.neff
    if neff is None:
        cache_dir = args.cache_dir or neuron_cache_dir()
        candidates = find_neffs(cache_dir, _compile_window_start(args.metrics_json))
        if not candidates:
            raise SystemExit(
                f"No .neff in {cache_dir} written since the runner started compiling. "
                "If the program came from the compile cache, pass --neff with its path."
            )
        neff = candidates[0]
        print(f"Profiling {neff} ({neff.stat().st_size / 1e6:.1f} MB)", flush=True)
    metrics = neuron_profile(neff, args.out_dir, args.core, args.explorer)
    print(render_markdown(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
