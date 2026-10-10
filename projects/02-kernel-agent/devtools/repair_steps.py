#!/usr/bin/env python3
"""
repair_steps.py -- share of repair steps that moved, per condition and level.

A repair step is a round that answers a checker message. It MOVED if the score rose or the checker
named a fault not seen earlier in that run; it is STUCK if the fault is the one from the round
before; it WENT BACK if the fault was seen earlier but not in the round before. Faults are compared
by analyze_log.classify, with numbers masked. The definition follows the RESULTS.md of another team
on this project, so the two can be read on one scale.

    python devtools/repair_steps.py
"""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from analyze_log import classify, load, top_of  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
GROUPS = {
    "baseline": ["seat-154/baseline", "seat-158/baseline2"],
    "line only (locate)": ["seat-150/locate-l3"],
    "line + shapes (state)": ["seat-153/state-l3", "seat-156/state-l4"],
    "reportable checker (internal,origin)": ["seat-153/checker-l3", "seat-157/checker-l3b",
                                              "seat-150/checker-l3c", "seat-158/checker-l3c",
                                              "seat-159/checker-l4", "seat-150/checker-l1",
                                              "seat-151/checker-l2"],
    "ahead v1 (internal,origin,ahead @16dcf94)": ["seat-153/a-l3", "seat-157/a-l3"],
}


def key(fb):
    return re.sub(r"\d+", "#", classify(fb))


def main():
    print(f"{'condition':44s} {'level 1':>16s} {'level 2':>16s} {'level 3':>16s} {'level 4':>16s}")
    for name, stems in GROUPS.items():
        cells = []
        for level in (1, 2, 3, 4):
            c = collections.Counter()
            for stem in stems:
                path = os.path.join(R, stem + ".jsonl")
                if not os.path.exists(path):
                    continue
                for e in load(path):
                    if e["level"] != level:
                        continue
                    seen, prev, best = [], None, 0.0
                    for i, rnd in enumerate(e["rounds"]):
                        t = top_of(rnd)
                        k = key(t["feedback"])
                        if i:
                            if t["reward"] > best + 1e-9 or k not in seen:
                                c["moved"] += 1
                            elif k == prev:
                                c["stuck"] += 1
                            else:
                                c["back"] += 1
                        best, prev = max(best, t["reward"]), k
                        seen.append(k)
            n = sum(c.values())
            cells.append(f"{c['moved']}/{n} ({100 * c['moved'] / n:.0f}%)" if n else "-")
        print(f"{name:44s} " + " ".join(f"{x:>16s}" for x in cells))


if __name__ == "__main__":
    main()
