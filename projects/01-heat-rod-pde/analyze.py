#!/usr/bin/env python3
"""analyze.py -- one table per arm from results/*.jsonl (written by agent.py).

    python analyze.py                  # every results/*.jsonl
    python analyze.py results/T2-*.jsonl

Counts only model answers as solves; a splice (source="splice") is reported in its own column.
"""
import collections
import glob
import json
import statistics as st
import sys

paths = sys.argv[1:] or sorted(glob.glob("results/*.jsonl"))
probs = collections.OrderedDict()     # (arm, run_id, problem) -> summary
for path in paths:
    for line in open(path):
        r = json.loads(line)
        if "run_id" not in r:
            continue
        key = (r.get("arm"), r["run_id"], r["problem"])
        s = probs.setdefault(key, dict(arm=r.get("arm"), seed=r.get("seed"), problem=r["problem"],
                                       cands=[], rounds=[], done=None, splice_round=None))
        if r["kind"] == "cand" and r.get("source", "model") == "model":
            s["cands"].append(r)
        elif r["kind"] == "cand":
            if r["reward"] == 1.0 and s["splice_round"] is None:
                s["splice_round"] = r["round"]
        elif r["kind"] == "round":
            s["rounds"].append(r)
        elif r["kind"] == "problem":
            s["done"] = r

arms = collections.OrderedDict()
for s in probs.values():
    if not s["cands"]:
        continue
    solved_round = min((c["round"] for c in s["cands"] if c["reward"] == 1.0), default=None)
    n_rounds = max(c["round"] for c in s["cands"]) + 1
    r0 = [c["reward"] for c in s["cands"] if c["round"] == 0]
    s.update(solved=solved_round is not None,
             rtext=str(solved_round + 1) if solved_round is not None else f">{n_rounds}",
             m1_r0=st.mean(r0), m1=st.mean(c["reward"] for c in s["cands"]),
             tokens=sum(c.get("completion_tokens", 0) for c in s["cands"]),
             wall=sum(r["wall_s"] for r in s["rounds"]),
             cut=sum(c.get("finish_reason") == "length" for c in s["cands"]),
             specfail=sum(bool(c.get("spec_error")) for c in s["cands"]),
             n=len(s["cands"]))
    arms.setdefault(s["arm"], []).append(s)

print(f"{'arm':<6} {'problem':<9} {'seed':>4} {'solved':>6} {'rounds':>6} {'M1 r0':>6} {'M1':>5} "
      f"{'tokens':>7} {'wall s':>7} {'cut':>4} {'specerr':>7} {'splice':>6}")
for arm, ss in arms.items():
    for s in ss:
        print(f"{arm:<6} {s['problem']:<9} {s['seed']:>4} {('yes' if s['solved'] else 'no'):>6} {s['rtext']:>6} "
              f"{s['m1_r0']:>6.2f} {s['m1']:>5.2f} {s['tokens']:>7} {s['wall']:>7.0f} {s['cut']:>4} "
              f"{s['specfail']:>7} {('r' + str(s['splice_round'] + 1)) if s['splice_round'] is not None else '-':>6}")
print()
print(f"{'arm':<6} {'problems':>8} {'solved':>7} {'rounds to solve':<22} {'M1 r0':>6} {'tok/solve':>9} "
      f"{'s/solve':>8} {'solved/h':>8}")
for arm, ss in arms.items():
    k = sum(s["solved"] for s in ss)
    wall = sum(s["wall"] for s in ss)
    tok = sum(s["tokens"] for s in ss)
    print(f"{arm:<6} {len(ss):>8} {k:>4}/{len(ss):<2} {','.join(s['rtext'] for s in ss):<22} "
          f"{st.mean(s['m1_r0'] for s in ss):>6.2f} {tok / max(k, 1):>9.0f} {wall / max(k, 1):>8.0f} "
          f"{3600 * k / wall if wall else 0:>8.1f}")
