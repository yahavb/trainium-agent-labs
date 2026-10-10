#!/usr/bin/env python3
"""Fill section 5 of analysis/l2_regression.md from v7's level-2 log.

    python analysis/l2_regression_fill.py runs/seat-116/Ev7_L2/<attempts>.jsonl [usage.jsonl]

Prints: runs, solved, round-0 samples correct, repair samples correct, any 0.95 (gate hold), the
round-0 prompt length (v7 should be 3,911 characters; the baseline's is 2,673), truncations, and the
carried failure mode per round of every run.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import taxonomy  # noqa: E402
import usage     # noqa: E402

att = sys.argv[1]
eps = taxonomy.load_attempts([att])
if len(sys.argv) > 2:
    q = usage.load(sys.argv[2:])
    for a in eps.values():
        usage.apply(a, q)
runs = [a for (_, _, lv), a in eps.items() if lv == 2]
r0 = [r for a in runs for r in a if r["round"] == 0]
rep = [r for a in runs for r in a if r["round"] > 0]
ok = lambda rs: sum(r["reward"] >= 1 - 1e-9 for r in rs)
print(f"runs {len(runs)}, solved {sum(any(r['reward'] >= 1 - 1e-9 for r in a) for a in runs)}")
print(f"round-0 samples correct {ok(r0)}/{len(r0)}; repair samples correct {ok(rep)}/{len(rep)} "
      f"(baseline + replica: 7/40 and 0/104)")
print(f"0.95 (gate hold): {sum(abs(r['reward'] - 0.95) < 1e-6 for a in runs for r in a)}")
print(f"round-0 prompt_chars: {sorted({r['prompt_chars'] for r in r0})} (v7 expected 3911, baseline 2673)")
print(f"truncated (finish=length): {sum(r.get('finish') == 'length' for a in runs for r in a)}")
for i, a in enumerate(runs, 1):
    rounds = collections.OrderedDict()
    for r in a:
        rounds.setdefault(r["round"], []).append(r)
    tops = [taxonomy.mode_of(max(rs, key=lambda r: r["reward"])) for rs in rounds.values()]
    print(f"  run {i}: round-0 rewards {[round(r['reward'], 2) for r in rounds[0]]}; carried {tops}")
