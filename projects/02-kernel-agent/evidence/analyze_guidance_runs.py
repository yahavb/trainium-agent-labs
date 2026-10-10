#!/usr/bin/env python3
"""Summarize guidance-classifier runs from their rounds.jsonl files.

    python3 analyze_guidance_runs.py evidence/guidance223_rounds.jsonl [more.jsonl ...]
"""
import json
import sys


def summarize(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    rounds = [r for r in recs if "chosen" in r]
    print(f"\n== {path} ==")
    print(f"rounds with a classifier decision: {len(rounds)}")
    n_match = sum(1 for r in rounds if r.get("match"))
    n_expected = sum(1 for r in rounds if r.get("expected") and not r.get("parse_failed"))
    fallbacks = sum(1 for r in rounds if r.get("parse_failed"))
    for r in rounds:
        mark = {True: "match ", False: "MISS  ", None: "?     "}[r.get("match")]
        print(f"  r{r['round']}: worst={r.get('worst')} valid={r.get('valid')} "
              f"| {mark} chose {r.get('chosen')} (expected {r.get('expected')}) "
              f"conf={r.get('confidence')} "
              f"| clf {r.get('classifier_tokens')}tok, applier {r.get('applier_tokens')}tok")
        if r.get("reason"):
            print(f"      reason: {r['reason'][:110]}")
    acc = f"{n_match}/{n_expected}" if n_expected else "n/a"
    print(f"classifier agreement: {acc} | parse fallbacks: {fallbacks}")
    solved = [r for r in recs if r.get("outcome") == "solved"]
    if solved:
        print(f"SOLVED at round {solved[0]['round']} | worst={solved[0]['worst']}")
    else:
        tail = [r for r in recs if r.get("outcome") == "unsolved"]
        if tail:
            print(f"unsolved: best progress {tail[0].get('best_progress')}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            summarize(p)
        except FileNotFoundError:
            print(f"(missing: {p})")
