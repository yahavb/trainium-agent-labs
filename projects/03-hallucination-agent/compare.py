#!/usr/bin/env python3
"""compare.py — two label-free runs side by side, per dataset: right / hallucinated / refused.
    python compare.py b_plain b_fc"""
import collections, json, sys
import agent
HALL = agent.HALLUCINATED


def finals(path):
    rows = [json.loads(l) for l in open(path)]
    last, kind, calls = {}, {}, collections.Counter()
    for r in rows:
        kind[r["item"]] = r["kind"]
        if r.get("note") != "fail_closed":
            calls[r["item"]] += 1
        if r.get("note") != "verify":
            last[r["item"]] = r["label"]
    by = collections.defaultdict(list)
    for i, l in last.items():
        by[i.split(":")[0] + ":" + kind[i]].append((i, l))
    return by, calls


def stats(xs):
    return dict(n=len(xs), cor=sum(l == "correct" for _, l in xs), hal=sum(l in HALL for _, l in xs),
                ref=sum(l == "over_abstain" for _, l in xs))


base, test = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("b_plain", "b_fc")
(pb, pc), (fb, fc) = finals(base + ".jsonl"), finals(test + ".jsonl")
print(f"{'':<26} {'---- ' + base + ' ----':^26} {'---- ' + test + ' ----':^26} {'change':^20}")
print(f"{'dataset:kind':<22} {'n':>3}  {'right':>6} {'halluc':>6} {'refused':>7}   "
      f"{'right':>6} {'halluc':>6} {'refused':>7}   {'right':>6} {'halluc':>6}")
T = collections.Counter()
for k in sorted(set(pb) | set(fb)):
    a, b = stats(pb.get(k, [])), stats(fb.get(k, []))
    print(f"{k:<22} {a['n']:>3}  {a['cor']:>6} {a['hal']:>6} {a['ref']:>7}   {b['cor']:>6} {b['hal']:>6} "
          f"{b['ref']:>7}   {b['cor'] - a['cor']:>+6} {b['hal'] - a['hal']:>+6}")
    for s, d in (("a", a), ("b", b)):
        for f in ("n", "cor", "hal", "ref"):
            T[s + f] += d[f]
print(f"{'TOTAL':<22} {T['an']:>3}  {T['acor']:>6} {T['ahal']:>6} {T['aref']:>7}   {T['bcor']:>6} "
      f"{T['bhal']:>6} {T['bref']:>7}   {T['bcor'] - T['acor']:>+6} {T['bhal'] - T['ahal']:>+6}")
print(f"\ncalls per question: {base} {sum(pc.values()) / max(1, len(pc)):.1f}   "
      f"{test} {sum(fc.values()) / max(1, len(fc)):.1f}")
