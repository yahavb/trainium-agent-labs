#!/usr/bin/env python3
"""pilot_summary.py — the pilot table from an attempt log, for runs that were stopped early.
    python pilot_summary.py pilot_attempts.jsonl"""
import collections, json, sys
import agent

rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r.get("note") == "verify":
        continue
    by[r["item"].split(":")[0] if ":" in r["item"] else "halworld"][r["item"]].append(r)
print(f"{'dataset':<14} {'items':>5}  {'r0 correct':>10} {'r0 halluc':>10} {'r0 over-abst':>12}"
      f"  {'loop correct':>12} {'loop halluc':>11}  most common round-0 failures")
for ds, items in by.items():
    z = [r["label"] for its in items.values() for r in its if r["round"] == 0]
    fin = []
    for its in items.values():
        if any(r["reward"] == 1.0 for r in its):
            fin.append("correct")
        else:
            last = max(r["round"] for r in its)
            fin.append(max((r for r in its if r["round"] == last), key=lambda r: r["reward"])["label"])
    top = collections.Counter(l for l in z if l != "correct").most_common(2)
    print(f"{ds:<14} {len(items):>5}  {agent.rate(z, {'correct'}):>10.0%} {agent.rate(z, agent.HALLUCINATED):>10.0%} "
          f"{agent.rate(z, {'over_abstain'}):>12.0%}  {agent.rate(fin, {'correct'}):>12.0%} "
          f"{agent.rate(fin, agent.HALLUCINATED):>11.0%}  {', '.join(f'{k} {v}' for k, v in top)}")
print("\nThe last item may be cut off mid-loop. Small n: use this to rank datasets, not to report.")
