#!/usr/bin/env python3
"""
build_examples.py — turn the attempt log into a bank of worked examples for the prompt. Inference
only: nothing is trained. The model's own successful replies come back as in-context examples on
items it has not seen.

The question this answers: can what the oracle loop discovered on seeds 0-199 make the model
right on ROUND 0 for seeds 500+, with no loop and no answer key at all?

    python build_examples.py attempts.jsonl --out bank.jsonl
    python build_examples.py --gold 1-7 --gold-seeds 0-49 --out gold_bank.jsonl

    python agent.py --all --rounds 1 --seed 500 --seeds 5 --repeat 5 -q                          # before
    python agent.py --all --rounds 1 --seed 500 --seeds 5 --repeat 5 -q --examples bank.jsonl    # after

--gold builds the bank from the generator's ideal replies instead. That is the baseline to beat:
do examples the loop DISCOVERED (in the model's own wording, including ones it reached only after
feedback) help more than templated gold answers? Report both.

--hard keeps only items the model got wrong in round 0 and right later. Those are the examples
that show it the traps it actually falls into.

Seeds >= 1000 are judging seeds. This script refuses them.
"""

import argparse
import collections
import json

import halworld


def _range(s):
    lo, _, hi = s.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--out", default="bank.jsonl")
    ap.add_argument("--hard", action="store_true",
                    help="only items wrong in round 0 and solved later")
    ap.add_argument("--gold", help="levels to take generator gold replies from, e.g. 1-7")
    ap.add_argument("--gold-seeds", default="0-49")
    a = ap.parse_args()
    if not a.logs and not a.gold:
        ap.error("give attempt logs, or --gold")

    bank = []
    by_item = collections.defaultdict(list)
    for path in a.logs:
        for line in open(path):
            r = json.loads(line)
            if isinstance(r["level"], int) and r["seed"] >= 1000:
                raise SystemExit(f"{path} holds judging seed {r['seed']}; refusing to use it")
            if r.get("note") in ("verify", "fail_closed"):
                continue   # the verifier's verdict, not an answer
            if r.get("examples"):
                continue   # replies written with examples in the prompt would echo them
            by_item[(r["item"], r["seed"])].append(r)

    hard = 0
    for (iid, seed), rows in sorted(by_item.items()):
        good = [r for r in rows if r["reward"] == 1.0]
        if not good:
            continue
        r0_wrong = any(r["round"] == 0 and r["reward"] < 1.0 for r in rows)
        if a.hard and not (r0_wrong and any(r["round"] > 0 for r in good)):
            continue
        hard += r0_wrong
        g = min(good, key=lambda r: r["round"])
        bank.append(dict(item=iid, level=g["level"], sub=g["sub"], seed=seed, kind=g["kind"],
                         reply=g["reply"], source="loop", round=g["round"]))

    if a.gold:
        for lv in _range(a.gold):
            for seed in _range(a.gold_seeds):
                if seed >= 1000:
                    raise SystemExit("gold seeds must be < 1000")
                for sub in halworld.SUBS:
                    it = halworld.make(lv, sub, seed)
                    bank.append(dict(item=it["id"], level=lv, sub=sub, seed=seed, kind=it["kind"],
                                     reply=it["ideal"], source="gold", round=0))

    with open(a.out, "w") as f:
        for e in bank:
            f.write(json.dumps(e) + "\n")
    kinds = collections.Counter(e["kind"] for e in bank)
    print(f"{len(by_item)} items in the logs; wrote {len(bank)} examples to {a.out}")
    print(f"  by kind: {dict(kinds)}")
    if a.logs:
        print(f"  {hard} of the loop examples had at least one wrong round-0 sample")


if __name__ == "__main__":
    main()
