"""A4: run the experiment -- levels x feedback A/B/C x N runs -- and summarise it.

    # on your seat, in the background (see SERVER.md):
    nohup python run_experiment.py --tag krish --levels levels/04_traffic_fsm levels/05_mac \\
        --runs 5 > run.log 2>&1 < /dev/null &

    python run_experiment.py --summary            # the table, from every results/*_summary.csv

Every run appends one row to results/<date>_<tag>_summary.csv and every attempt to
results/<date>_<tag>_attempts.jsonl -- one pair of files per person, so six seats never write the same file.
Re-running the same command SKIPS runs already in that summary, so a dropped connection costs nothing.

Fairness: the order is interleaved -- run 1 of every level x feedback, then run 2, ... -- so a slow or
busy moment on the server cannot fall on one feedback level only. Levels that fail the self-test are
refused: a broken level would grade the model wrongly.
"""

import argparse
import csv
import glob
import os
import socket
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent  # noqa: E402
import selftest  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIELDS = ["date", "tag", "seat", "model", "level", "feedback", "run", "solved", "rounds_used", "best_score",
          "attempts", "distinct_designs", "prompt_tokens", "completion_tokens", "seconds", "offline", "run_id",
          "max_rounds"]


def summary_path(tag, date):
    return os.path.join(RESULTS, f"{date}_{tag}_summary.csv")


def done_runs(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["level"], r["feedback"], int(r["run"])) for r in csv.DictReader(f)}


def run(a):
    date = time.strftime("%Y-%m-%d")
    os.makedirs(RESULTS, exist_ok=True)
    spath = summary_path(a.tag, date)
    apath = os.path.join(RESULTS, f"{date}_{a.tag}_attempts.jsonl")

    levels = a.levels or selftest.level_dirs(selftest.LEVELS)
    if not levels:
        sys.exit("No levels given and none in veriloop/levels/ yet.")
    for d in levels:
        problems, _ = selftest.check_level(d)
        if problems:
            sys.exit(f"REFUSED: {d} fails the self-test -- fix it first:\n  " + "\n  ".join(problems))

    already = done_runs(spath)
    plan = [(r, d, fb) for r in range(1, a.runs + 1) for d in levels for fb in a.feedback]
    todo = [p for p in plan if (os.path.basename(os.path.normpath(p[1])), p[2], p[0]) not in already]
    print(f"{len(plan)} runs planned, {len(plan) - len(todo)} already done, {len(todo)} to go. "
          f"Summary: {os.path.relpath(spath, HERE)}")

    new = not os.path.exists(spath)
    with open(spath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, (r, d, fb) in enumerate(todo, 1):
            level = os.path.basename(os.path.normpath(d))
            run_id = f"{a.tag}-{level}-{fb}-r{r}-{time.strftime('%H%M%S')}"
            print(f"\n[{k}/{len(todo)}] {level}  feedback {fb}  run {r}")
            s = agent.run_level(d, fb, a.rounds, a.samples, apath, a.offline, run_id=run_id, quiet=not a.verbose)
            row = dict(date=date, tag=a.tag, seat=socket.gethostname(), model=agent.MODEL, level=level,
                       feedback=fb, run=r, run_id=run_id, max_rounds=a.rounds,
                       **{k2: s[k2] for k2 in ("solved", "rounds_used", "best_score", "attempts", "distinct_designs",
                                               "prompt_tokens", "completion_tokens", "seconds", "offline")})
            w.writerow(row)
            f.flush()
            print(f"   -> {'SOLVED' if s['solved'] else 'not solved'} in {s['rounds_used']} round(s), "
                  f"best {s['best_score']:.2f}, {s['seconds']:.0f}s")
    summarise()


def summarise(paths=None, include_offline=False):
    """Solve rate per level x feedback across every summary file -- the table for TEAM.md and the graph."""
    paths = paths or sorted(glob.glob(os.path.join(RESULTS, "*_summary.csv")) +
                            glob.glob(os.path.join(HERE, "..", "results", "*_summary.csv")))
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [r for r in csv.DictReader(f) if include_offline or r["offline"] != "True"]
    if not rows:
        print("No results yet (offline test runs are left out; add --include-offline to see them).")
        return {}
    cell = defaultdict(list)
    for r in rows:
        cell[(r["level"], r["feedback"])].append(r)
    levels = sorted({r["level"] for r in rows})
    fbs = sorted({r["feedback"] for r in rows})
    print("\nsolved / runs   (mean rounds when solved)      " + str(len(rows)) + " runs from " +
          ", ".join(sorted({r['tag'] for r in rows})))
    print(f"{'level':<20}" + "".join(f"{'feedback ' + fb:>22}" for fb in fbs))
    table = {}
    for lv in levels:
        line = f"{lv:<20}"
        for fb in fbs:
            rs = cell.get((lv, fb), [])
            solved = [r for r in rs if r["solved"] == "True"]
            mean_rounds = (sum(int(r["rounds_used"]) for r in solved) / len(solved)) if solved else None
            table[(lv, fb)] = (len(solved), len(rs), mean_rounds)
            txt = f"{len(solved)}/{len(rs)}" + (f" ({mean_rounds:.1f})" if mean_rounds else "") if rs else "-"
            line += f"{txt:>22}"
        print(line)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="who/what, used in the file names, e.g. krish or krish-fsm")
    ap.add_argument("--levels", nargs="*", help="level folders (default: every level in veriloop/levels/)")
    ap.add_argument("--feedback", nargs="*", default=["A", "B", "C"], choices=["A", "B", "C", "D"])
    ap.add_argument("--runs", type=int, default=5, help="runs per level x feedback (default 5)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--offline", action="store_true", help="canned designs, no model (testing only)")
    ap.add_argument("--verbose", action="store_true", help="print every round")
    ap.add_argument("--summary", action="store_true", help="only print the table from existing results")
    ap.add_argument("--include-offline", action="store_true", help="count offline test runs in the table")
    a = ap.parse_args()
    if a.summary:
        summarise(include_offline=a.include_offline)
        return 0
    if not a.tag:
        ap.error("--tag is required (your name, so files from different seats never collide)")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
