#!/usr/bin/env python3
"""
recheck_spoiled.py -- re-judge attempts the referee rejected because ITS OWN folder changed mid-check.

The hardened referee hashes every .py under projects/03-chipboost before and after a check and calls any
change tampering ("THE CANDIDATE MODIFIED THE REFEREE", verdict "rules"). A `git pull` in that folder
while a check runs does exactly that, so an honest candidate is logged as cheating. That is an
infrastructure failure, not a verdict on the kernel (REFEREE.md: retry referee failures, never log them).

This finds every such record, runs the SAME code through the same referee again, and puts the fresh record
in its place, under the same run id, attempt number and seat. The spoiled originals are not deleted: they
are moved to logs/seat-<N>/archive/spoiled-<file>.jsonl.bak, out of the dashboard's attempts*.jsonl.

Run it in the seat pod, from a clone nobody is pulling into, when no sweep or search is mid-run there:
    CHIPBOOST_CORE=3 python tools/recheck_spoiled.py logs/seat-102/*.jsonl --dry-run
    CHIPBOOST_CORE=3 python tools/recheck_spoiled.py logs/seat-102/*.jsonl
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import search  # noqa: E402  the same referee wrapper, record finisher and candidate folder search.py uses

MARK = "MODIFIED THE REFEREE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="attempts*.jsonl and sweep*.jsonl files")
    ap.add_argument("--dry-run", action="store_true", help="list the spoiled records; change nothing")
    a = ap.parse_args()

    found = []
    for path in a.logs:
        lines = open(path).read().splitlines()
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if MARK in (rec.get("referee_message") or ""):
                found.append((path, i, rec))
    print(f"{len(found)} record(s) rejected because the referee's folder changed mid-check")
    for path, i, rec in found:
        print(f"  {os.path.relpath(path)} line {i + 1}: run {rec.get('run_id')} attempt {rec.get('attempt_no')} "
              f"caps {search.caps_str(search.read_caps(rec['code']))}; {rec['referee_message'][:120]}")
    if a.dry_run or not found:
        return 0
    if any(r.get("arm") != "random_search" for _, _, r in found):
        sys.exit("only random_search records can be re-judged here: a model arm's record carries its prompt, "
                 "and replaying it is that arm owner's call.")

    import speedcheck
    referee, name = search.open_referee(speedcheck)
    print(f"referee: {name}, core {os.environ.get('CHIPBOOST_CORE', 'auto')}")
    replaced = {}
    run_dir = os.path.join(search.RUNS_ROOT, "recheck")
    os.makedirs(run_dir, exist_ok=True)
    try:
        for path, i, rec in found:
            src = rec["code"]
            cand = os.path.join(run_dir, hashlib.sha1(src.encode()).hexdigest()[:12] + ".py")
            with open(cand, "w") as f:
                f.write(src)
            new = search.call_referee(referee, cand)
            if new is None:
                print(f"  run {rec['run_id']} attempt {rec['attempt_no']}: the referee failed again; left as is")
                continue
            new = search.finish(new, rec["seat"], rec["run_id"], rec["attempt_no"], src)
            replaced.setdefault(path, {})[i] = (rec, new)
            print(f"  run {rec['run_id']} attempt {rec['attempt_no']}: was rules (spoiled), now {new['verdict']}"
                  + (f" {new['speedup']:.3f}x" if new.get("speedup") is not None else ""))
    finally:
        referee.close()

    for path, fixes in replaced.items():
        lines = open(path).read().splitlines()
        archive_dir = os.path.join(os.path.dirname(path), "archive")
        os.makedirs(archive_dir, exist_ok=True)
        archive = os.path.join(archive_dir, f"spoiled-{os.path.basename(path)}.bak")
        with open(archive, "a") as f:
            for i, (old, new) in sorted(fixes.items()):
                f.write(json.dumps(old) + "\n")
                lines[i] = json.dumps(new)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
        print(f"{os.path.relpath(path)}: {len(fixes)} record(s) replaced; originals in {os.path.relpath(archive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
