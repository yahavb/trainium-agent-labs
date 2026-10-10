#!/usr/bin/env python3
"""read_failures.py -- print the failing kernels and their feedback from a log, to READ them.

    python read_failures.py logs/base_L3.jsonl              # first 6 attempts
    python read_failures.py logs/base_L3.jsonl --level 3 -n 10
    python read_failures.py logs/base_L3.jsonl --worst      # only reward < 1.0

The most important hour in the plan is reading the actual failing kernels and writing, for each
wall, one sentence: what does the model believe that is false? This dumps exactly what you need
for that, from the `code` and `feedback` fields the agent logs.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--worst", action="store_true", help="only attempts that did not solve")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.log) if l.strip()]
    if a.level is not None:
        rows = [r for r in rows if r["level"] == a.level]
    if a.worst:
        rows = [r for r in rows if r["reward"] < 1.0 - 1e-9]
    for r in rows[:a.n]:
        print(f"--- exp={r.get('exp')} run={r.get('run')} level={r['level']} "
              f"round={r['round']} sample={r.get('sample')} reward={r['reward']}")
        print(f"FEEDBACK: {r['feedback'][:400]}")
        print(f"CODE:\n{r['code']}\n")


if __name__ == "__main__":
    main()
