#!/usr/bin/env python3
"""
results.py -- per-experiment, per-level solve RATES from attempt logs (never the best run).

    python scripts/results.py runs/*/*.jsonl            # every experiment found
    python scripts/results.py --exp skel runs/*/*.jsonl

An experiment is the tag prefix before "_L" (base_L1 -> base, skel_L3_r2 -> skel). A RUN is one
(tag, run-index) pair; its score is the best reward it reached (what agent.py reports), and
rounds-to-solve is 1 + the first round that scored 1.0. Output matches RESULTS.md's cell format:
`solved k/n, mean x.xx, all=[...]`.
"""

import argparse
import collections
import glob
import json
import sys


def runs_of(paths):
    best = collections.defaultdict(float)
    solved_at = {}
    rounds = collections.defaultdict(int)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if "claim" in r:
                continue
            tag = r.get("tag") or "?"
            exp = tag.split("_L")[0]
            key = (exp, r["level"], tag, r.get("run", 0))
            best[key] = max(best[key], r["reward"])
            rounds[key] = max(rounds[key], r["round"] + 1)
            if r["reward"] >= 1.0 - 1e-9 and key not in solved_at:
                solved_at[key] = r["round"] + 1
    return best, solved_at, rounds


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--exp")
    a = ap.parse_args()
    paths = [p for g in a.logs for p in glob.glob(g)]
    best, solved_at, rounds = runs_of(paths)
    table = collections.defaultdict(list)
    for (exp, lv, tag, run), score in sorted(best.items()):
        if a.exp and exp != a.exp:
            continue
        table[(exp, lv)].append((score, solved_at.get((exp, lv, tag, run)),
                                 rounds[(exp, lv, tag, run)]))
    exps = sorted({e for e, _ in table})
    for exp in exps:
        print(f"\n== {exp}")
        cells = []
        for lv in sorted(l for e, l in table if e == exp):
            got = table[(exp, lv)]
            sc = [round(s, 2) for s, _, _ in got]
            k = sum(1 for s in sc if s >= 1.0 - 1e-9)
            att = [r for _, r, _ in got if r]
            rnd = [r for _, _, r in got]
            cell = f"solved {k}/{len(sc)}, mean {sum(sc) / len(sc):.2f}, all={sc}"
            print(f"  L{lv}: {cell}   rounds-to-solve={att or '-'}   rounds used={rnd}")
            cells.append(cell)
        print("  row: | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
