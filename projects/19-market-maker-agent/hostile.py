"""Post-hoc robustness check: the held-out seeds again, but with the market spread set to 1 tick.

Why: the real-data replay (lobster.py) showed MSFT and INTC at a 1-tick spread 99% of the day, while
every synthetic level used 2 ticks. Strategies "verified" on 2 ticks improved the price by one tick,
which on a 1-tick market takes liquidity. The held-out seeds came from the same generator as dev,
so they could not catch it. This is the hostile set they should have included.

Fair by the same test as the levels: the reference passes all three at spread 1, and the naive
quoter passes level 1 and fails 2 and 3, as it does at spread 2. Added after runs A, B and C were
frozen; it changes no run, it only re-scores their final strategies.

    python hostile.py            # writes runs/hostile.md
"""
import glob
import json
import os

import checker
import mmsim

HERE = os.path.dirname(os.path.abspath(__file__))


def check_spread1(code, level):
    bad = checker.static_check(code)
    if bad:
        return "static", None
    saved = mmsim.LEVELS[level]
    mmsim.LEVELS[level] = dict(saved, spread=1)
    try:
        out = mmsim.evaluate(code, level, mmsim.heldout_seeds())
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}", None
    finally:
        mmsim.LEVELS[level] = saved
    if out["error"]:
        return f"exception: {out['error']['type']}", None
    m = checker.aggregate(out["episodes"])
    if out["violations"]:
        rules = sorted({v[2] for v in out["violations"]})
        return f"rules: {', '.join(rules)} x{len(out['violations'])}", m
    if m["obligation"] < mmsim.OBLIGATION:
        return f"obligation {m['obligation']:.0%}", m
    if m["pnl_lcb"] <= 0:
        return f"profit lcb {m['pnl_lcb']:+.0f}", m
    return "PASS", m


def main():
    entries = [(f"{f} (level {lv})", lv, open(os.path.join(HERE, f)).read())
               for f in ("baseline.py", "reference.py") for lv in mmsim.CALIBRATED]
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r.get("final") and r.get("code") and r["dev_solved"]:
                entries.append((f"{r['run']} rep {r['rep']}", r["level"], r["code"]))
    lines = ["# Hostile held-out: the 32 held-out seeds at a 1-tick spread\n",
             "Only strategies the agent claimed as verified (plus the baseline and reference). "
             "Same pass rule as the checker.\n",
             "| strategy | level | spread 2 (held-out) | spread 1 (hostile) | lcb at spread 1 | "
             "hostile states (252 probes) |",
             "|---|---|---|---|---|---|"]
    for name, lv, code in entries:
        normal = checker.check(code, lv, mmsim.heldout_seeds())
        verdict, m = check_spread1(code, lv)
        pr = checker.probe(code)
        if pr.get("error"):
            states = pr["error"][:60]
        elif pr["findings"]:
            rules = sorted({f["rule"] for f in pr["findings"]})
            states = f"{len(pr['findings'])} fail ({', '.join(rules)})"
        else:
            states = "PASS"
        lcb = f"{m['pnl_lcb']:+.0f}" if m else "-"
        lines.append(f"| {name} | {lv} | {'PASS' if normal['solved'] else 'fail'} | {verdict} | "
                     f"{lcb} | {states} |")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    open(os.path.join(HERE, "runs", "hostile.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
