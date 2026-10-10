#!/usr/bin/env python3
"""Summarize rl_trace_agent.py JSONL logs (v3 and v4 rows) without pandas.

  python summarize_rl_trace.py rl_level2_v4.jsonl [more.jsonl ...]
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def load(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"warning: skipped malformed line in {path}")
                    continue
                r["_file"] = path
                rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    rows = load(p.parse_args().logs)
    if not rows:
        raise SystemExit("No valid JSONL rows found.")

    print(f"attempts: {len(rows)}")
    for key in ("kernel_reward", "trace_reward", "rl_reward"):
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            print(f"{key}: mean={mean(vals):.4f} best={max(vals):.4f}")

    print("\nBy level:")
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("level")].append(r)
    for level, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        kr = [float(r["kernel_reward"]) for r in rs]
        print(f"  level {level}: n={len(rs)} mean_kernel_reward={mean(kr):.3f} "
              f"solved_attempts={sum(x >= .999 for x in kr)}/{len(rs)}")

    # An episode is one (file, run_id, level, episode). v3 rows have no run_id: one episode per file+level.
    print("\nEpisodes (attempts to first verified kernel; round is 0-based):")
    eps = defaultdict(list)
    for r in rows:
        eps[(r["_file"], r.get("run_id", "v3"), r.get("level"), r.get("episode", 0))].append(r)
    by_level = defaultdict(list)
    for (f, run, level, ep), rs in sorted(eps.items(), key=lambda x: str(x[0])):
        solved = [r["round"] for r in rs if float(r["kernel_reward"]) >= .999]
        by_level[level].append(min(solved) if solved else None)
        best = max(float(r["kernel_reward"]) for r in rs)
        print(f"  {f} run={run} level={level} ep={ep}: best={best:.2f} "
              f"first_verified_round={min(solved) if solved else 'never'} "
              f"rounds={1 + max(r['round'] for r in rs)}")
    for level, firsts in sorted(by_level.items(), key=lambda x: str(x[0])):
        got = [x for x in firsts if x is not None]
        print(f"  level {level}: verified in {len(got)}/{len(firsts)} episodes"
              + (f", mean first round {mean(got):.1f}" if got else ""))

    print("\nBy strategy:")
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("action", "unknown")].append(r)
    for action, rs in sorted(groups.items()):
        print(f"  {action:18s} n={len(rs):3d} mean_RL={mean([float(r['rl_reward']) for r in rs]):.3f} "
              f"mean_kernel={mean([float(r['kernel_reward']) for r in rs]):.3f}")

    temps = [r for r in rows if "temperature" in r]
    if temps:
        print("\nBy temperature:")
        groups = defaultdict(list)
        for r in temps:
            groups[r["temperature"]].append(r)
        for t, rs in sorted(groups.items()):
            print(f"  t={t:<4} n={len(rs):3d} mean_RL={mean([float(r['rl_reward']) for r in rs]):.3f} "
                  f"mean_kernel={mean([float(r['kernel_reward']) for r in rs]):.3f}")

    print("\nFailure categories:")
    cats = Counter(r.get("failure_category", "unknown") for r in rows)
    for c, n in cats.most_common():
        print(f"  {c:14s} {n:3d}  ({100 * n / len(rows):.0f}%)")

    v4 = [r for r in rows if r.get("schema_version", 0) >= 4]
    if v4:
        print("\nHygiene (v4 rows):")
        print(f"  repeated identical code: {sum(bool(r.get('repeated')) for r in v4)}/{len(v4)}")
        print(f"  token-limit cut-offs:    {sum(bool(r.get('truncated')) for r in v4)}/{len(v4)}")
        print(f"  module-level code stripped: {sum(bool(r.get('sanitizer_dropped')) for r in v4)}/{len(v4)}")
        adv = [float(r["advantage"]) for r in v4 if "advantage" in r]
        if adv and any(abs(a) > 1e-9 for a in adv):
            print(f"  mean |advantage| within sample groups: {mean([abs(a) for a in adv]):.3f}")


if __name__ == "__main__":
    main()