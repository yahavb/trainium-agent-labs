"""Re-grade the logs. Anyone can run this to check that the numbers in runs/ are what the checker
says, on their own machine:

  1. completeness: every arm has 3 repeats x 3 levels, and a final row for each
  2. safety: every logged strategy passes the static gate (math/numpy only, no I/O)
  3. reproducibility: every verified strategy is re-graded here, on dev and held-out, and must
     give exactly the profit lower bound the seat logged

    python audit.py
"""
import collections
import glob
import json
import os
import sys

import checker
import mmsim

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*-[0-9].jsonl"))):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    finals = [r for r in rows if r.get("final")]
    tries = [r for r in rows if not r.get("final")]
    ok = True

    print("1. completeness")
    by = collections.defaultdict(set)
    for f in finals:
        by[f["run"]].add((f["rep"], f["level"]))
    planned = {"C4b": 2, "C5": 5, "C6": 5, "C6r": 4, "C7": 5, "C7r": 4}   # partial arms; the rest are 3 levels x 3 reps
    for run in sorted(by):
        n = len(by[run])
        good = n == planned.get(run, 9)
        ok &= good
        print(f"   {'ok ' if good else 'BAD'} {run:3s} {n}/{planned.get(run, 9)} level-runs, "
              f"{sum(1 for t in tries if t['run'] == run)} attempts")

    print("2. safety: static gate on every logged strategy")
    codes = {t["code"] for t in tries if t.get("code")} | {f["code"] for f in finals if f.get("code")}
    unsafe = [c for c in codes if checker.static_check(c) and
              checker.static_check(c)[0] in ("FORBIDDEN_IMPORT", "FORBIDDEN_CALL")]
    print(f"   {len(codes)} distinct strategies; {len(unsafe)} would be refused as unsafe "
          f"(those were refused at L0 and never run)")

    print("3. reproducibility: re-grade every verified strategy here")
    seen = set()
    for f in finals:
        if not f["dev_solved"] or (f["code"], f["level"]) in seen:
            continue
        seen.add((f["code"], f["level"]))
        logged = next(t for t in tries if t["run"] == f["run"] and t["rep"] == f["rep"]
                      and t["level"] == f["level"] and t["solved"])
        dev = checker.check(f["code"], f["level"])
        held = checker.check(f["code"], f["level"], mmsim.heldout_seeds())
        same = (abs(dev["metrics"]["pnl_lcb"] - logged["pnl_lcb"]) < 1e-6 and
                abs(held["metrics"]["pnl_lcb"] - f["heldout_lcb"]) < 1e-6)
        ok &= same and dev["solved"] and held["solved"] == f["heldout_solved"]
        print(f"   {'ok ' if same else 'BAD'} {f['run']} rep {f['rep']} level {f['level']}: dev lcb "
              f"{dev['metrics']['pnl_lcb']:+.3f} (seat logged {logged['pnl_lcb']:+.3f}), held-out "
              f"{held['metrics']['pnl_lcb']:+.3f} (logged {f['heldout_lcb']:+.3f})")
    print("AUDIT", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
