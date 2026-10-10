#!/usr/bin/env python3
"""Validate a pinned comparison batch and write JSON and Markdown reports.

Run with --allow-partial while collecting a live batch. Validation errors still
fail; only missing/incomplete runs and a final unfinished JSON line are allowed.
Infrastructure events never enter kernel-attempt counts.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ARMS = ("referee", "model_alone", "random_search")
FAILURES = ("rules", "wrong", "heldout_fail")
CORRECT = ("slower", "no_gain", "faster")
PRIOR_NOTE = ("Random search uses P2's expert-template cap prior. Agent arms start "
              "from the reference baseline. These results compare complete search "
              "setups; the three-arm comparison does not isolate feedback alone.")
VERSION_NOTE = ("This batch remains pinned to P1 434e5f9, P2 919c6be, and P3 2ce9416. "
                "It predates the subsequent representative-feedback fix and does not "
                "evaluate that fix. Keep these results separate from any future batch "
                "using the corrected feedback implementation.")


def read_jsonl(path, allow_partial=False):
    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8-sig")
    rows, notes = [], []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if allow_partial and index == len(lines) - 1 and not text.endswith("\n"):
                notes.append(f"{path.name}: ignored unfinished final JSON line")
                continue
            raise ValueError(f"{path.name}:{index + 1}: invalid JSON") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{index + 1}: expected an object")
        rows.append(row)
    return rows, notes


def aggregate(rows):
    counts = Counter(r["verdict"] for r in rows)
    n = len(rows)
    best = max([1.0] + [r["speedup"] for r in rows if r["verdict"] == "faster"])
    failed = sum(counts[v] for v in FAILURES)
    correct = sum(counts[v] for v in CORRECT)
    heldout_checked = counts["heldout_fail"] + counts["faster"]
    return {"attempts": n, "verdicts": dict(sorted(counts.items())),
            "correct": correct, "failed": failed,
            "failure_rate": failed / n if n else None,
            "heldout_checked": heldout_checked,
            "heldout_failure_rate": counts["heldout_fail"] / heldout_checked if heldout_checked else None,
            "best_verified_speedup": best}


def check_record(row, validate, where):
    problems = validate(row)
    for key in ("arm", "kernel", "run_id", "seat", "attempt_no", "verdict"):
        if row.get(key) is None:
            problems.append(f"{key} cannot be null in comparison records")
    if str(row.get("referee_message") or "").startswith(("(fake)", "(stub)")):
        problems.append("fake/stub record")
    timed = (row.get("speedup"), row.get("time_us_median"), row.get("baseline_us_same_session"))
    if any(v is not None for v in timed):
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and v > 0 for v in timed):
            problems.append("timing requires finite positive speedup, candidate and baseline times")
        elif not math.isclose(timed[0], timed[2] / timed[1], rel_tol=1e-6, abs_tol=1e-9):
            problems.append("speedup disagrees with baseline/candidate timing ratio")
    if row.get("verdict") in CORRECT:
        if not all(v is not None for v in timed):
            problems.append("correct timed verdict lacks timing")
        if row.get("sim_ok") is not True or row.get("chip_ok") is not True:
            problems.append("correct verdict lacks simulator/chip success")
        if row.get("source") != "chip":
            problems.append("timed comparison verdict is not chip measured")
    if row.get("verdict") == "faster" and isinstance(timed[0], (int, float)) and timed[0] <= 1:
        problems.append("faster verdict has speedup <= 1")
    if problems:
        raise ValueError(f"{where}: {'; '.join(problems)}")


def summarize(directory, validate, repeats=3, budget=8, allow_partial=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8-sig")) if state_path.exists() else {}
    notes, runs, all_rows, seen = [], [], [], set()
    expected = {f"{arm}-r{repeat}" for arm in ARMS for repeat in range(repeats)}
    unexpected = {p.stem for p in directory.glob("*-r*.jsonl")} - expected
    if unexpected:
        raise ValueError(f"Unexpected comparison logs: {sorted(unexpected)}")
    for arm in ARMS:
        for repeat in range(repeats):
            tag = f"{arm}-r{repeat}"
            rows, extra = read_jsonl(directory / f"{tag}.jsonl", allow_partial)
            notes.extend(extra)
            for i, row in enumerate(rows):
                check_record(row, validate, f"{tag}:{i + 1}")
                if row["arm"] != arm:
                    raise ValueError(f"{tag}: record belongs to {row['arm']}")
            identities = {(r["kernel"], r["arm"], r["seat"], r["run_id"]) for r in rows}
            if len(identities) > 1:
                raise ValueError(f"{tag}: contains more than one run")
            if identities & seen:
                raise ValueError(f"{tag}: duplicate run identity across files")
            seen.update(identities)
            # P3 increments before logging (1..8); P2 logs its initial candidate as 0.
            first_attempt = 0 if arm == "random_search" else 1
            if [r["attempt_no"] for r in rows] != list(range(first_attempt, first_attempt + len(rows))):
                raise ValueError(f"{tag}: attempt numbers must be unique and contiguous from {first_attempt}")
            if len(rows) > budget:
                raise ValueError(f"{tag}: {len(rows)} attempts exceed budget {budget}")
            complete = len(rows) == budget and not extra
            if not complete:
                notes.append(f"{tag}: partial ({len(rows)}/{budget} attempts)")
            runs.append(dict(tag=tag, arm=arm, repeat=repeat, complete=complete,
                             run_id=rows[0]["run_id"] if rows else None, **aggregate(rows)))
            all_rows.extend(rows)
    for field, expected_value in (("repeat", repeats), ("budget", budget)):
        if field in state and state[field] != expected_value:
            raise ValueError(f"state.{field}={state[field]} differs from requested {expected_value}")
    pinned = {"referee_commit": "434e5f9", "p2_commit": "919c6be", "p3_commit": "2ce9416"}
    for field, expected_value in pinned.items():
        if state and state.get(field) != expected_value:
            raise ValueError(f"state.{field} must identify pinned snapshot {expected_value}")
    if state and (not isinstance(state.get("core"), int) or state["core"] < 0):
        raise ValueError("state.core must identify the pinned nonnegative core")
    completed_tags = state.get("completed", [])
    if not isinstance(completed_tags, list) or any(not isinstance(tag, str) for tag in completed_tags):
        raise ValueError("state.completed must be a list of run tags")
    if len(completed_tags) != len(set(completed_tags)) or set(completed_tags) - expected:
        raise ValueError("state.completed contains duplicate or unknown runs")
    actual_complete = {r["tag"] for r in runs if r["complete"]}
    if set(completed_tags) - actual_complete:
        raise ValueError("state.completed claims runs whose copied logs are incomplete; refresh the snapshot")
    if state.get("phase") == "complete" and set(completed_tags) != expected:
        raise ValueError("state.complete requires all nine expected run tags in completed")
    if state.get("phase") not in {None, "preflight", "acceptance_retry", "failed", "complete"} | expected:
        raise ValueError(f"Unknown runner phase: {state['phase']}")
    acceptance_path = directory / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8-sig")) if acceptance_path.exists() else None
    gate = (state.get("acceptance") == "passed" and acceptance is not None
            and not validate(acceptance) and acceptance.get("verdict") == "no_gain"
            and acceptance.get("sim_ok") is True and acceptance.get("chip_ok") is True)
    if not gate:
        notes.append("Acceptance gate is not confirmed by state.json and acceptance.json")
    complete = all(r["complete"] for r in runs) and state.get("phase") == "complete" and gate
    if state.get("phase") != "complete":
        notes.append(f"Runner phase is {state.get('phase', 'missing')}")
    if state.get("phase") == "complete" and not complete:
        raise ValueError("Runner claims completion but acceptance or run counts are incomplete")
    if not complete and not allow_partial:
        raise ValueError("Incomplete batch; use --allow-partial for a provisional report: " + "; ".join(notes))
    infra, extra = read_jsonl(directory / "infrastructure.jsonl", allow_partial)
    notes.extend(extra)
    arms = {}
    for arm in ARMS:
        started = [r for r in runs if r["arm"] == arm and r["attempts"]]
        bests = [r["best_verified_speedup"] for r in started]
        stats = aggregate([r for r in all_rows if r["arm"] == arm])
        stats.pop("best_verified_speedup")
        arms[arm] = dict(stats, started_runs=len(started), completed_runs=sum(r["complete"] for r in started),
                         median_best_speedup=statistics.median(bests) if bests else None,
                         min_best_speedup=min(bests) if bests else None,
                         max_best_speedup=max(bests) if bests else None)
    return dict(status="complete" if complete else "partial", provisional=not complete,
                directory=str(directory.resolve()), expected_repeats=repeats, expected_budget=budget,
                attempts=len(all_rows), acceptance_passed=gate, runner_state=state,
                runs=runs, arms=arms, notes=notes, candidate_prior_note=PRIOR_NOTE,
                version_note=VERSION_NOTE,
                infrastructure={"events": len(infra), "by_kind": dict(Counter(r.get("kind", "unknown") for r in infra)),
                                "records": infra, "included_in_attempt_counts": False})


def markdown(report):
    def speed(value):
        return "—" if value is None else f"{value:.6f}x"
    lines = ["# Agent comparison report", "", f"Status: **{report['status'].upper()}**; {report['attempts']} recorded attempts.", ""]
    if report["provisional"]:
        lines += ["Provisional: aggregates include started runs only, including unfinished runs. Missing runs are not imputed as 1x.", ""]
    lines += ["| Arm | Runs complete/started | Attempts | Correct | Failed | Failure rate | Median best | Min best | Max best |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, s in report["arms"].items():
        rate = "—" if s["failure_rate"] is None else f"{s['failure_rate']:.1%}"
        lines.append(f"| {arm} | {s['completed_runs']}/{s['started_runs']} | {s['attempts']} | {s['correct']} | {s['failed']} | {rate} | {speed(s['median_best_speedup'])} | {speed(s['min_best_speedup'])} | {speed(s['max_best_speedup'])} |")
    lines += ["", "Best means maximum verified `faster` speedup per run, defaulting to 1x when no candidate improves. Correct means `slower`, `no_gain`, or `faster`; it does not imply held-out validation for every candidate. Failed means `rules`, `wrong`, or `heldout_fail`. Failure-rate denominator is recorded attempts. Held-out rejection rate uses only `heldout_fail` plus `faster` as its denominator.", "", "| Arm | rules | wrong | heldout_fail | slower | no_gain | faster |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, s in report["arms"].items():
        lines.append("| " + arm + " | " + " | ".join(str(s["verdicts"].get(v, 0)) for v in FAILURES + CORRECT) + " |")
    lines += ["", report["candidate_prior_note"], "", report["version_note"], "", f"Infrastructure: {report['infrastructure']['events']} separately logged events; excluded from all attempt counts and failure rates.", ""]
    if report["notes"]:
        lines += ["Notes:", ""] + [f"- {note}" for note in report["notes"]] + [""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--schema", type=Path, default=Path(__file__).resolve().parents[1] / "schema.py")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--markdown-out", type=Path, required=True)
    args = ap.parse_args()
    if args.repeats < 1 or args.budget < 1:
        ap.error("repeats and budget must be positive")
    spec = importlib.util.spec_from_file_location("comparison_schema", args.schema)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    try:
        result = summarize(args.directory, schema.validate, args.repeats, args.budget, args.allow_partial)
    except (ValueError, OSError) as exc:
        ap.exit(1, f"Comparison validation failed: {exc}\n")
    for path, content in ((args.json_out, json.dumps(result, indent=2, allow_nan=False) + "\n"),
                          (args.markdown_out, markdown(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"{result['status']}: {result['attempts']} attempts; wrote {args.json_out} and {args.markdown_out}")


if __name__ == "__main__":
    main()
