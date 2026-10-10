"""A3: calibration -- does the model KNOW when its design is wrong?

For designs the model wrote (read from attempt logs), ask the model -- in a fresh conversation, never
showing it the checker's verdict -- how confident it is that the design meets the spec. Then compare with
the checker's real verdict. Being confidently wrong is the worst behaviour the rubric names: on hardware
a wrong design does not crash, it ships.

    python calibrate.py results/2026-10-10_krish-L4_attempts.jsonl          # every distinct design
    python calibrate.py results/*_attempts.jsonl --max 60 --out results/calibration.csv
    python calibrate.py LOG --offline                                         # no model, tests the script

Run it when the experiments are finished: it shares the model server with them.

The AGENT's own claim needs no such test: run_level() only ever reports "solved" when the checker
passed every test, and otherwise "could not verify" -- it cannot claim success the checker did not grant.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent  # noqa: E402

SEARCH = [os.path.join(HERE, "levels"), os.path.join(HERE, "tests", "sim_fixtures")]
CONFIDENT = 80        # "confidently wrong" = at least this confident, and the checker says wrong


def find_spec(level):
    for root in SEARCH:
        p = os.path.join(root, level, "spec.txt")
        if os.path.isfile(p):
            return open(p).read()
    raise FileNotFoundError(f"no spec.txt for level {level!r} in levels/ or tests/sim_fixtures/")


def confidence_prompt(spec, design):
    return (f"Specification:\n{spec.strip()}\n\n"
            f"A design:\n```verilog\n{design.strip()}\n```\n\n"
            f"Does this design meet the specification exactly, on every input and every clock cycle? "
            f"Reply with exactly one line: CONFIDENCE: <a number from 0 to 100>")


def parse_confidence(reply):
    m = re.search(r"CONFIDENCE\s*[:=]?\s*(\d{1,3})", reply or "", re.I) or re.search(r"\b(\d{1,3})\s*%?", reply or "")
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 100 else None


def load_designs(paths, max_n, seed=0):
    """Distinct (level, design) pairs from the logs, with the checker's verdict. Offline rows are skipped."""
    seen, rows = set(), []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("offline") or not r.get("verilog"):
                continue
            key = (r["level"], r["verilog"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    if max_n and len(rows) > max_n:
        # Keep every correct design (they are rare on hard levels) and sample the rest, so both sides exist.
        good = [r for r in rows if r["solved"]]
        bad = [r for r in rows if not r["solved"]]
        random.Random(seed).shuffle(bad)
        rows = good[:max_n] + bad[:max(0, max_n - len(good[:max_n]))]
    return rows


def ask_confidence(row, offline):
    if offline:
        return f"CONFIDENCE: {random.Random(row['verilog']).randint(0, 100)}"
    reply, _ = agent.ask(confidence_prompt(find_spec(row["level"]), row["verilog"]), temperature=0.0)
    return reply


def report(results):
    have = [r for r in results if r["confidence"] is not None]
    right = [r["confidence"] for r in have if r["solved"]]
    wrong = [r["confidence"] for r in have if not r["solved"]]
    conf_wrong = [r for r in have if not r["solved"] and r["confidence"] >= CONFIDENT]
    print(f"\n{len(results)} designs; confidence parsed for {len(have)}")
    print(f"  checker says CORRECT: {len(right):>4}   mean confidence {sum(right)/len(right):5.1f}" if right
          else "  checker says CORRECT:    0")
    print(f"  checker says WRONG:   {len(wrong):>4}   mean confidence {sum(wrong)/len(wrong):5.1f}" if wrong
          else "  checker says WRONG:      0")
    if wrong:
        print(f"  confidently wrong (>= {CONFIDENT} and wrong): {len(conf_wrong)} of {len(wrong)} wrong designs "
              f"({100 * len(conf_wrong) / len(wrong):.0f}%)")
    if have:
        brier = sum((r["confidence"] / 100 - int(r["solved"])) ** 2 for r in have) / len(have)
        print(f"  Brier score {brier:.3f}  (0 = perfect; 0.25 = always saying 50)")
    print("\n  confidence   designs   actually correct")
    for lo in range(0, 100, 20):
        hi = 100 if lo == 80 else lo + 19
        b = [r for r in have if lo <= r["confidence"] <= hi]
        if b:
            print(f"  {lo:>3}-{hi:<3}      {len(b):>5}   {sum(r['solved'] for r in b):>4} "
                  f"({100 * sum(r['solved'] for r in b) / len(b):.0f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="attempt logs (*_attempts.jsonl)")
    ap.add_argument("--max", type=int, default=0, help="at most this many designs (0 = all)")
    ap.add_argument("--out", help="write one CSV row per design here")
    ap.add_argument("--offline", action="store_true", help="random confidences, no model (testing only)")
    a = ap.parse_args()

    rows = load_designs(a.logs, a.max)
    if not rows:
        sys.exit("No (non-offline) designs in those logs.")
    print(f"asking for confidence on {len(rows)} distinct designs ...")
    with cf.ThreadPoolExecutor(max_workers=4) as ex:      # the server runs 4 sequences at once
        replies = list(ex.map(lambda r: ask_confidence(r, a.offline), rows))
    results = [dict(level=r["level"], run=r["run"], round=r["round"], attempt=r["attempt"],
                    feedback=r["feedback"], score=r["score"], solved=r["solved"],
                    confidence=parse_confidence(reply), reply=reply.strip()[:200], offline=a.offline)
               for r, reply in zip(rows, replies)]
    report(results)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
