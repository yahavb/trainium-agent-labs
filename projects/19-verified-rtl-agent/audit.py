"""Calibration audit: does a PASS survive a proof? DESIGN.md 6.7.

The testbench samples a few hundred input vectors. Formal equivalence covers every one. So every
combinational PASS is re-checked with yosys, and a PASS that is not equivalent is a TESTBENCH
ESCAPE -- the agent claimed something the simulator could not catch. That count is the calibration
number: an agent that says PASS and is wrong is worse than one that says FAIL.

    python audit.py runs/C-1.jsonl [runs/B-1.jsonl ...]
    -> runs/C-1.audit.json, and a table
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import problems as P
from checker import formal_check

CATEGORIES = ("equivalent", "not_equivalent", "unsupported", "not_run")


def final_and_passing(path: str) -> dict:
    """{problem: (claim, kind, passing code or None)} from one run file."""
    out = {}
    passing = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("passed") and rec["problem"] not in passing:
                passing[rec["problem"]] = rec.get("code")
            if rec.get("claim"):
                out[rec["problem"]] = [rec["claim"], rec["kind"], None]
    for pid, v in out.items():
        v[2] = passing.get(pid)
    return out


def audit(path: str) -> dict:
    results = {}
    for pid, (claim, kind, code) in sorted(final_and_passing(path).items()):
        if claim != "PASS":
            continue
        if kind != "comb":
            results[pid] = "not_run"        # bounded model checking of seq designs is a stretch goal
            continue
        formal, cex, _, _ = formal_check(P.load(pid), code or "")
        results[pid] = formal if formal in CATEGORIES else "unsupported"
        if formal == "not_equivalent":
            results[pid + ":counterexample"] = cex
    counts = {c: sum(v == c for k, v in results.items() if ":" not in k) for c in CATEGORIES}
    name = os.path.basename(path)[: -len(".jsonl")]
    run, _, rep = name.partition("-")
    return {"run": run, "rep": rep, "file": os.path.basename(path), "results": results, "counts": counts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    print(f"{'run file':18s} {'comb PASS':>9s} {'equivalent':>10s} {'ESCAPES':>8s} {'unsupported':>11s} {'seq (not run)':>13s}")
    for path in a.files:
        rep = audit(path)
        c = rep["counts"]
        with open(path[: -len(".jsonl")] + ".audit.json", "w") as f:
            json.dump(rep, f, indent=2)
        comb = c["equivalent"] + c["not_equivalent"] + c["unsupported"]
        print(f"{rep['file']:18s} {comb:9d} {c['equivalent']:10d} {c['not_equivalent']:8d} "
              f"{c['unsupported']:11d} {c['not_run']:13d}")
        for pid, v in rep["results"].items():
            if v == "not_equivalent":
                print(f"    escape: {pid}  {rep['results'].get(pid + ':counterexample')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
