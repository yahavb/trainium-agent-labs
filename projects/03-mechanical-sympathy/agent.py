#!/usr/bin/env python3
"""Run one Trainium attempt, check it, and append the result to attempts.csv.

This controller enforces the project order:

    run candidate -> check against CPU -> record correctness -> expose timing

The controller never records a performance result for an incorrect candidate.
It is suitable for manual attempts and for a model-driven outer loop.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ATTEMPT_FIELDS = [
    "timestamp",
    "experiment",
    "runner_commit",
    "samudra_commit",
    "hardware",
    "precision",
    "input_shape",
    "compile_seconds",
    "warmup_runs",
    "timed_steps",
    "median_ms",
    "p95_ms",
    "steps_per_second",
    "correctness_score",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "fixtures" / "manifest.json",
    )
    parser.add_argument(
        "--attempts",
        type=Path,
        default=project_root / "results" / "attempts.csv",
    )
    parser.add_argument("--workload", choices=("single-step", "rollout"), default="single-step")
    parser.add_argument("--forecast-steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--hardware", default="aws-trainium")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_attempt(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ATTEMPT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in ATTEMPT_FIELDS})


def base_row(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    source_pin = read_json(project_root / "samudra-source.json")
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": args.experiment,
        "runner_commit": git_commit(project_root),
        "samudra_commit": source_pin["commit"],
        "hardware": args.hardware,
        "precision": args.precision,
        "warmup_runs": args.warmup,
        "status": "runner_failed",
        "notes": args.notes,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    row = base_row(args, project_root)

    with tempfile.TemporaryDirectory(prefix="samudra-attempt-") as directory:
        temporary = Path(directory)
        candidate = temporary / "candidate.npz"
        metrics_path = temporary / "metrics.json"
        check_path = temporary / "check.json"

        runner_command = [
            sys.executable,
            str(project_root / "runners" / "trainium_runner.py"),
            "--adapter",
            str(args.adapter.resolve()),
            "--fixture",
            str(args.fixture.resolve()),
            "--manifest",
            str(args.manifest.resolve()),
            "--candidate-output",
            str(candidate),
            "--metrics-json",
            str(metrics_path),
            "--workload",
            args.workload,
            "--forecast-steps",
            str(args.forecast_steps),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--precision",
            args.precision,
            "--hardware",
            args.hardware,
        ]
        runner = subprocess.run(runner_command, text=True, capture_output=True)
        if runner.returncode != 0:
            detail = (runner.stderr or runner.stdout).strip().replace("\n", " ")
            row["notes"] = f"{args.notes} runner failed: {detail[:500]}".strip()
            append_attempt(args.attempts.resolve(), row)
            print(runner.stdout, end="")
            print(runner.stderr, end="", file=sys.stderr)
            return runner.returncode

        metrics = read_json(metrics_path)
        check_command = [
            sys.executable,
            str(project_root / "checker.py"),
            "--manifest",
            str(args.manifest.resolve()),
            "--candidate",
            str(candidate),
            "--performance-json",
            str(metrics_path),
            "--json-out",
            str(check_path),
        ]
        checked = subprocess.run(check_command, text=True, capture_output=True)
        check = read_json(check_path)
        passed = checked.returncode == 0 and check.get("status") == "passed"

        row.update(
            {
                "input_shape": json.dumps(metrics.get("input_shapes", {}), sort_keys=True),
                "correctness_score": check.get("correctness_score", 0.0),
                "status": check.get("status", "checker_failed"),
                "notes": " ".join(
                    part
                    for part in (args.notes, str(check.get("reason", "")))
                    if part
                ),
            }
        )
        if passed:
            row.update(
                {
                    "compile_seconds": metrics.get("compile_seconds", ""),
                    "timed_steps": metrics.get("timed_forecast_steps", ""),
                    "median_ms": metrics.get("median_ms", ""),
                    "p95_ms": metrics.get("p95_ms", ""),
                    "steps_per_second": metrics.get("steps_per_second", ""),
                }
            )
        append_attempt(args.attempts.resolve(), row)
        print(checked.stdout, end="")
        if checked.stderr:
            print(checked.stderr, end="", file=sys.stderr)
        return checked.returncode


if __name__ == "__main__":
    raise SystemExit(main())
