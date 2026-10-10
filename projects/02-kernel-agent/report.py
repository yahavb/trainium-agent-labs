#!/usr/bin/env python3
"""
report.py — one table comparing feedback modes, from the attempt logs.

    python report.py                                   # every *.jsonl under results*/
    python report.py results/directed.jsonl results-baselines/enriched.jsonl
    python report.py --cut 8                           # also score every run as if it had 8 rounds
    python report.py > RESULTS.md

For each feedback mode and level it reports the score, and -- because the score alone hides most of
what happens -- how the model RESPONDED to the feedback:

    repair steps   rounds after the first, each one an answer to a checker message
    moved          repair steps where the named mistake was gone in the next attempt
    stuck          repair steps that hit the same mistake again

A run that sits at 0.30 while fixing a different mistake every round and a run that sits at 0.30
repeating one mistake look identical by score. This table tells them apart.

Standard library only.
"""

import argparse
import glob
import json
import re
from collections import defaultdict

ORDER = ["raw", "enriched", "located", "directed", "directed2", "directed3"]
MARK = " The failing line is line "


def failure(rec):
    """The mistake, as the checker named it, with the quoted line and every number removed.

    Numbers go because "got src=4, dst=16384" and "got src=4, dst=64" are the same mistake made
    with different sizes. Without this, a feedback mode that prints more numbers would look as if
    the model were moving when it was only changing sizes.
    """
    text = rec.get("failure") or (rec.get("feedback") or "").split(MARK)[0]
    return re.sub(r"\d+", "N", text)


def load(paths):
    """{(mode, diverse, session, run): {level: [best attempt of each round, in order]}}"""
    runs = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("feedback_mode", "enriched"), bool(r.get("diverse")),
                       r.get("session", "?"), r.get("run", 0))
                rounds = runs[key][r["level"]]
                cur = rounds.get(r["round"])
                if cur is None or r["reward"] > cur["reward"]:
                    rounds[r["round"]] = r
    return {k: {lv: [rd[i] for i in sorted(rd)] for lv, rd in v.items()} for k, v in runs.items()}


def describe(seq, cut=None):
    if cut:
        seq = seq[:cut]
    best = max(r["reward"] for r in seq)
    solved = next((i for i, r in enumerate(seq) if r["reward"] >= 1 - 1e-9), None)
    steps = moved = 0
    for prev, nxt in zip(seq, seq[1:]):
        steps += 1
        if nxt["reward"] > prev["reward"] + 1e-9 or failure(nxt) != failure(prev):
            moved += 1
    kinds = len({failure(r) for r in seq if r["reward"] < 1 - 1e-9})
    return dict(best=best, solved=solved, rounds=len(seq), steps=steps, moved=moved,
                stuck=steps - moved, kinds=kinds)


def table(runs, cut=None):
    keys = sorted(runs, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1], k[2], k[3]))
    levels = sorted({lv for v in runs.values() for lv in v})
    out = ["| level | feedback | best score | solved | rounds | repair steps | moved | stuck | "
           "different mistakes |", "|---|---|---|---|---|---|---|---|---|"]
    for lv in levels:
        for k in keys:
            if lv not in runs[k]:
                continue
            d = describe(runs[k][lv], cut)
            name = k[0] + (" + varied prompts" if k[1] else "")
            out.append(f"| {lv} | {name} | {d['best']:.2f} | "
                       f"{'round ' + str(d['solved']) if d['solved'] is not None else 'no'} | "
                       f"{d['rounds']} | {d['steps']} | {d['moved']} | {d['stuck']} | {d['kinds']} |")
    return "\n".join(out)


def totals(runs, cut=None):
    keys = sorted(runs, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1], k[2], k[3]))
    out = ["| feedback | session | levels run | levels solved | repair steps | moved | stuck | "
           "moved share |", "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        ds = [describe(seq, cut) for seq in runs[k].values()]
        steps, moved = sum(d["steps"] for d in ds), sum(d["moved"] for d in ds)
        name = k[0] + (" + varied prompts" if k[1] else "")
        out.append(f"| {name} | {k[2]} | {len(ds)} | {sum(d['solved'] is not None for d in ds)} | "
                   f"{steps} | {moved} | {steps - moved} | "
                   f"{f'{100 * moved / steps:.0f}%' if steps else 'n/a'} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="attempt logs; default: every *.jsonl under results*/")
    ap.add_argument("--cut", type=int,
                    help="also show every run truncated to this many rounds, so runs given "
                         "different round budgets can be compared like for like")
    a = ap.parse_args()
    paths = a.logs or sorted(glob.glob("results*/*.jsonl") + glob.glob("results*/previous-*/*.jsonl"))
    if not paths:
        raise SystemExit("no attempt logs found")
    runs = load(paths)
    if not runs:
        raise SystemExit("the logs hold no attempts yet")
    print("# Feedback modes compared\n")
    print(f"From {len(paths)} log file(s): {', '.join(paths)}\n")
    print("Each row is one run of one level. A repair step is a round that answers a checker "
          "message. It *moved* if the score rose or the checker named a different mistake next "
          "time, and was *stuck* if the same mistake came back. Two messages count as the same "
          "mistake when they match after removing the quoted line and all numbers.\n")
    print("## Per level\n\n" + table(runs))
    print("\n## Per run\n\n" + totals(runs))
    if a.cut:
        print(f"\n## Per level, first {a.cut} rounds only\n\n" + table(runs, a.cut))
        print(f"\n## Per run, first {a.cut} rounds only\n\n" + totals(runs, a.cut))


if __name__ == "__main__":
    main()
