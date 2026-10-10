#!/usr/bin/env python3
"""
pilot.py — which dataset makes Qwen3-8B hallucinate, and on which one does the loop fix it?

Runs the same N items of each dataset twice: round 0 alone (the model, no help) and the oracle loop
(--rounds). Prints one row per dataset. Pick the dataset for your demo from this table, not from
a paper: the one with high round-0 hallucination AND a large drop after the loop. High hallucination
with no drop means the fix is not in the context, so no update rule can show anything there.

    python halsets.py --fetch squad2 --n 200      # once per dataset, see halsets.py
    python pilot.py --data squad2 faith-unans+squad2 faith-cf hotpot popqa --n 50
    python pilot.py --data halworld --n 28        # the contamination-free control

About 20 minutes for 5 datasets x 50 items at --samples 4 --rounds 3 on one seat. Run it with nohup.
"""

import argparse
import collections
import io
import random
import sys

import agent
import halsets
import halworld


def run(spec, a):
    if spec == "halworld":
        items = [halworld.make(lv, sub, 0) for lv in halworld.LEVELS for sub in halworld.SUBS][:a.n]
    else:
        items = halsets.load(spec, n=a.n, max_chars=a.max_chars)
    argv = ["--samples", str(a.samples), "--rounds", str(a.rounds), "-q", "--max-tokens",
            str(a.max_tokens)] + (["--offline"] if a.offline else [])
    if a.base:
        argv += ["--base", a.base]
    ag = agent.build_parser().parse_args(argv)
    log = open(a.log, "a") if a.log else io.StringIO()
    res = [agent.solve(it, ag, log, 0, random.Random(0), None) for it in items]
    z = [l for r in res for l in r["zero_shot"]]
    fin = [r["final"] for r in res]
    kinds = collections.Counter(it["kind"] for it in items)
    return dict(n=len(items), kinds=dict(kinds),
                r0_correct=agent.rate(z, {"correct"}),
                r0_halluc=agent.rate(z, agent.HALLUCINATED),
                r0_over=agent.rate(z, {"over_abstain"}),
                fin_correct=agent.rate(fin, {"correct"}),
                fin_halluc=agent.rate(fin, agent.HALLUCINATED),
                top=collections.Counter(l for l in z if l != "correct").most_common(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--base")
    ap.add_argument("--log", default="pilot_attempts.jsonl")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if a.offline:
        print("*** OFFLINE: fake generator, numbers are meaningless ***")
    rows = []
    for spec in a.data:
        print(f"running {spec} ...", flush=True)
        rows.append((spec, run(spec, a)))
    print(f"\n{'dataset':<22} {'n':>4}  {'r0 correct':>10} {'r0 halluc':>10} {'r0 over-abst':>12}"
          f"  {'loop correct':>12} {'loop halluc':>11}  most common round-0 failures")
    for spec, r in rows:
        print(f"{spec:<22} {r['n']:>4}  {r['r0_correct']:>10.0%} {r['r0_halluc']:>10.0%} "
              f"{r['r0_over']:>12.0%}  {r['fin_correct']:>12.0%} {r['fin_halluc']:>11.0%}  "
              f"{', '.join(f'{k} {v}' for k, v in r['top'])}")
    print("\nr0 = every round-0 sample. loop = the best answer after the oracle loop, per item.")
    print("A good demo dataset: high r0 halluc AND loop halluc much lower. Then confirm with --repeat 5.")


if __name__ == "__main__":
    main()
