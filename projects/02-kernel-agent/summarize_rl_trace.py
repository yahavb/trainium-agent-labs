#!/usr/bin/env python3
"""Summarize rl_trace_agent.py JSONL logs without pandas."""
import argparse, json, statistics
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    a = p.parse_args()
    rows = []
    for path in a.logs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    r["_file"] = path
                    rows.append(r)
                except json.JSONDecodeError:
                    print(f"warning: skipped malformed line in {path}")
    if not rows:
        raise SystemExit("No valid JSONL rows found.")
    print(f"attempts: {len(rows)}")
    for key in ("kernel_reward", "trace_reward", "rl_reward"):
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            print(f"{key}: mean={statistics.mean(vals):.4f} best={max(vals):.4f}")
    print("\\nBy level:")
    groups = defaultdict(list)
    for r in rows: groups[r.get("level")].append(r)
    for level, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        kr = [float(r["kernel_reward"]) for r in rs]
        solved = sum(x >= .999 for x in kr)
        print(f"  level {level}: n={len(rs)} mean_kernel_reward={statistics.mean(kr):.3f} "
              f"solved_attempts={solved}/{len(rs)}")
    print("\\nBy strategy:")
    groups = defaultdict(list)
    for r in rows: groups[r.get("action", "unknown")].append(r)
    for action, rs in sorted(groups.items()):
        vals = [float(r["rl_reward"]) for r in rs]
        kernels = [float(r["kernel_reward"]) for r in rs]
        print(f"  {action:18s} n={len(rs):3d} mean_RL={statistics.mean(vals):.3f} "
              f"mean_kernel={statistics.mean(kernels):.3f}")

if __name__ == "__main__":
    main()
