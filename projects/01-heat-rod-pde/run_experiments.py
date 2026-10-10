#!/usr/bin/env python3
"""Repeat one fixed problem, snapshot its code, and summarize real or offline runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

PROJECT = Path(__file__).resolve().parent
DEFAULTS = dict(level=1, sub=3, seed=0, samples=4, rounds=4, max_tokens=1200,
                tool_steps=1, no_tools=False, think=False,
                feedback_style="baseline", repeats=5)


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    unknown = raw.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    config = DEFAULTS | raw
    for key, default in DEFAULTS.items():
        if type(config[key]) is not type(default):
            raise ValueError(f"{key} must have type {type(default).__name__}")
    for key in ("samples", "rounds", "max_tokens", "repeats"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["tool_steps"] < 0:
        raise ValueError("tool_steps must be nonnegative")
    if config["level"] not in (0, 1) or config["sub"] not in (1, 2, 3):
        raise ValueError("level must be 0 or 1, and sub must be 1, 2, or 3")
    if config["feedback_style"] not in ("baseline", "structured"):
        raise ValueError("feedback_style must be baseline or structured")
    return config


def agent_arguments(config):
    args = []
    for key, value in config.items():
        if key == "repeats":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend((flag, str(value)))
    return args


def inspect_attempts(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("agent produced no attempt records")
    failures = Counter(key for record in records for key, passed in record["parts"].items()
                       if not passed)
    return dict(solved=any(r["reward"] == 1.0 for r in records),
                best_reward=max(r["reward"] for r in records),
                rounds=len({r["round"] for r in records}),
                candidates=len(records),
                tool_calls=sum(r["tool_calls"] for r in records),
                failed_checks_by_candidate=dict(failures))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def summary_of(results, requested, offline):
    complete = [r for r in results if r["status"] == "complete"]
    # Failed or unstarted runs never disappear from the solve-rate denominator.
    solved = sum(r["solved"] for r in complete)
    success_rounds = [r["rounds"] for r in complete if r["solved"]]
    measured = not offline and len(complete) == requested
    return dict(kind="offline_preview" if offline else "live_experiment",
                requested_runs=requested, attempted_runs=len(results),
                completed_runs=len(complete),
                batch_complete=len(complete) == requested,
                solved_runs=None if offline else solved,
                solve_rate=solved / requested if measured else None,
                median_successful_rounds=statistics.median(success_rounds)
                if measured and success_rounds else None,
                median_wall_seconds=statistics.median(r["wall_seconds"] for r in complete)
                if measured else None,
                results=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/baseline.json")
    parser.add_argument("--repeats", type=int, help="override config for a short debugging run")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "runs")
    parser.add_argument("--offline", action="store_true", help="fake generator; not model evidence")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.repeats is not None:
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        config["repeats"] = args.repeats
    base = os.environ.get("HEATROD_BASE_URL", "")
    if not args.offline and not base:
        parser.error("HEATROD_BASE_URL is unset. Run inside your seat, or use --offline for preview.")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = ("offline-" if args.offline else "live-") + config["feedback_style"]
    batch = Path(tempfile.mkdtemp(prefix=f"{label}-{stamp}-", dir=root))
    source = batch / "source"
    source.mkdir()
    for path in PROJECT.glob("*.py"):
        shutil.copy2(path, source / path.name)
    write_json(batch / "config.json", config)
    versions = {}
    for package in ("numpy", "sympy", "httpx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(batch / "environment.json", dict(
        python=sys.version, packages=versions, offline=args.offline,
        model=os.environ.get("HEATROD_MODEL", "Qwen/Qwen3-8B"),
        endpoint_host=urlparse(base).hostname,
        note="Source snapshot is authoritative; environment credentials are not recorded."))
    print(f"Results directory: {batch}", flush=True)
    if args.offline:
        print("OFFLINE PREVIEW: fake answers. Do not report this as model performance.", flush=True)
    results = []
    write_json(batch / "summary.json", summary_of(results, config["repeats"], args.offline))
    for repeat in range(1, config["repeats"] + 1):
        run = batch / f"run-{repeat:02d}"
        run.mkdir()
        attempts = run / "attempts.jsonl"
        command = [sys.executable, "-u", str(source / "agent.py"),
                   *agent_arguments(config), "--log", str(attempts)]
        if args.offline:
            command.append("--offline")
        write_json(run / "command.json", command)
        print(f"\n===== run {repeat}/{config['repeats']} =====", flush=True)
        started = time.perf_counter()
        try:
            with (run / "console.log").open("w") as console:
                process = subprocess.Popen(command, cwd=source, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace")
                try:
                    for line in process.stdout:
                        console.write(line)
                        console.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            result = dict(run=repeat, exit_code=code,
                          wall_seconds=round(time.perf_counter() - started, 3))
            if code:
                result.update(status="error", error="agent failed; inspect console.log")
            else:
                result.update(inspect_attempts(attempts), status="complete")
        except KeyboardInterrupt:
            result = dict(run=repeat, status="interrupted", error="experiment interrupted",
                          wall_seconds=round(time.perf_counter() - started, 3))
        except (OSError, ValueError, KeyError) as error:
            result = dict(run=repeat, status="error", error=str(error),
                          wall_seconds=round(time.perf_counter() - started, 3))
        results.append(result)
        write_json(batch / "summary.json", summary_of(results, config["repeats"], args.offline))
        if result["status"] != "complete":
            print(f"Stopped: {result['error']}. Partial results: {batch}", file=sys.stderr)
            return 130 if result["status"] == "interrupted" else 1
    summary = summary_of(results, config["repeats"], args.offline)
    if args.offline:
        print("\nPreview finished. Real solve rate: unavailable (offline).")
    else:
        print(f"\nSolved {summary['solved_runs']}/{summary['requested_runs']} runs; "
              f"rate={summary['solve_rate']:.0%}; "
              f"median wall={summary['median_wall_seconds']:.1f}s")
    print(f"Summary: {batch / 'summary.json'}")
    return 0


if __name__ == "__main__":
    def stop_batch(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_batch)
    sys.exit(main())
