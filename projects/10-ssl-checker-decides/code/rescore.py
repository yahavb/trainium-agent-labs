"""H5, offline: does the reward rank a degenerate running kernel above a crash, and how often does it matter?

Today's reward gives 0.2 for "runs" even when the output is NaN or (mostly) zeros, i.e. the kernel ran
but never wrote a result. Measured on seat 49: such a kernel (0.50) outranked a nearly-correct one that
crashed (0.30), and the loop then repaired the wrong attempt.

r2 = today's reward minus the 0.2 "runs" credit when the feedback says the output was non-finite or
mostly zeros. No other change. For every log: how many attempts are degenerate, how often the attempt
chosen for repair changes, and how the per-run best score changes.
"""
import collections
import glob
import json
import os
import re
import sys

DEGENERATE = re.compile(r"NON-FINITE OUTPUT|OUTPUT IS \d+% ZEROS|% of the output is zero")


def r2(row):
    r = row.get("reward")
    if r is None:
        return None
    return round(r - 0.2, 2) if DEGENERATE.search(row.get("feedback") or "") else r


def runs_of(rows):
    """Split a log into runs: an explicit `run` field, or a new run whenever round goes back to 0."""
    if rows and "run" in rows[0]:
        by = collections.defaultdict(list)
        for r in rows:
            by[(r.get("run"), r.get("level"))].append(r)
        return list(by.values())
    out, cur, last = [], [], None
    for r in rows:
        key = (r.get("level"), r.get("gate"))
        if cur and (r["round"] < cur[-1]["round"] or key != last):
            out.append(cur)
            cur = []
        cur.append(r)
        last = key
    if cur:
        out.append(cur)
    return out


def main(paths):
    print(f"{'log':<52} {'att':>4} {'degen':>5} {'pick changes':>13}  best per run: today -> r2")
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if r.get("reward") is not None]
        if not rows:
            continue
        degen = sum(1 for r in rows if DEGENERATE.search(r.get("feedback") or ""))
        changes, rounds = 0, 0
        bests = []
        for run in runs_of(rows):
            by_round = collections.defaultdict(list)
            for r in run:
                by_round[r["round"]].append(r)
            for rnd, cands in by_round.items():
                if len(cands) < 2:
                    continue
                rounds += 1
                a = max(range(len(cands)), key=lambda i: cands[i]["reward"])
                b = max(range(len(cands)), key=lambda i: r2(cands[i]))
                # a different attempt only matters if it is a different error
                if (cands[a]["feedback"] or "")[:80] != (cands[b]["feedback"] or "")[:80]:
                    changes += 1
            bests.append((round(max(r["reward"] for r in run), 2), round(max(r2(r) for r in run), 2)))
        pick = f"{changes}/{rounds}" if rounds else "n/a (1 sample)"
        today = [b[0] for b in bests]
        new = [b[1] for b in bests]
        print(f"{os.path.basename(p):<52} {len(rows):>4} {degen:>5} {pick:>13}  {today} -> {new}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("attempts-rishabh-*.jsonl")))
