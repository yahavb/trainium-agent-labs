#!/usr/bin/env python3
"""Calibration after the fact: what the agent would have claimed about runs that already happened.

    python scripts/calibrate.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl \
        -o analysis/calibration_baseline_seat116

For every (run, level) in the logs it takes that run's best kernel -- the one the agent carried as
best: the highest reward, first to reach it -- and passes it through agent.verdict(), the same code
a live run calls when a level ends. So the confidence comes from agent.confidence() and the claim
from the held-out set exactly as in a live run; nothing is re-implemented here. Needs the NKI
simulator (a seat pod, or the container in SETUP_PYTHON.md). Works on old logs without a run field
(split by order, as taxonomy.py does) and on new ones.

Writes <out>.md (per-run table, calibration, Brier) and <out>.csv (one row per (run, level)).
"""
import argparse
import csv
import io
import os
import sys
import types
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "projects", "02-kernel-agent"))
import agent      # noqa: E402  the live agent's confidence() and verdict()
import taxonomy   # noqa: E402  the same episode split the taxonomy uses

BUCKETS = ((0.0, 0.01), (0.01, 0.5), (0.5, 0.8), (0.8, 1.01))


def best_attempt(attempts):
    """The kernel solve() returned as best: highest reward, the first attempt to reach it."""
    top = max(r["reward"] for r in attempts)
    return next(r for r in attempts if r["reward"] == top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("-o", "--out", default="analysis/calibration")
    ap.add_argument("--note", action="append", default=[], help="a line appended under Notes")
    a = ap.parse_args()

    episodes = taxonomy.load_attempts(a.attempts)
    rows = []
    for (fi, run, level), atts in episodes.items():
        best = best_attempt(atts)
        spent = dict(prompt=sum(r.get("prompt_tokens") or 0 for r in atts),
                     completion=sum(r.get("completion_tokens") or 0 for r in atts),
                     rounds=len({r["round"] for r in atts}))
        ns = types.SimpleNamespace(no_eval=False, run=run)
        with redirect_stdout(io.StringIO()):
            v = agent.verdict(ns, level, best["reward"], spent["rounds"], best.get("code", ""),
                              spent)
        v["file"], v["best_round"] = a.attempts[fi], best["round"]
        rows.append(v)
        print(f"run {run + 1} level {level}: loop {best['reward']:.2f}  confidence "
              f"{v['confidence']:.2f}  held-out {v['heldout_passed']}/{v['heldout_total']}  "
              f"{v['claim']}")

    scored = [v for v in rows if v["heldout_total"]]
    ok = lambda v: v["heldout_passed"] == v["heldout_total"]
    brier = (sum((v["confidence"] - ok(v)) ** 2 for v in scored) / len(scored)) if scored else None
    over = [v for v in scored if v["confidence"] >= 0.5 and not ok(v)]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["run", "level", "loop_reward", "best_round", "rounds", "confidence", "reasons",
            "heldout_passed", "heldout_total", "claim", "first_failure", "error"]
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for v in rows:
            w.writerow([v["run"] + 1, v["level"], v["reward"], v["best_round"], v["rounds"],
                        v["confidence"], "; ".join(v["reasons"]), v["heldout_passed"],
                        v["heldout_total"], v["claim"], v["first_failure"] or "", v["error"] or ""])

    esc = lambda s: str(s or "").replace("|", "\\|")
    md = ["# Calibration after the fact", "",
          f"Source: {', '.join(a.attempts)}. {len(rows)} (run, level) pairs. For each, the run's "
          f"best kernel went through `agent.verdict()`: the confidence from `agent.confidence()` "
          f"(stated from the loop's evidence only), then the held-out set "
          f"(`nkibench.evaluate`, new shapes x 4 value kinds) in the NKI 0.6.0 CPU simulator.",
          "",
          "| run | level | loop reward | confidence | why | held-out | claim | first held-out failure |",
          "|---|---|---|---|---|---|---|---|"]
    for v in rows:
        held = f"{v['heldout_passed']}/{v['heldout_total']}" if v["heldout_total"] else "-"
        md.append(f"| {v['run'] + 1} | {v['level']} | {v['reward']:.2f} | {v['confidence']:.2f} | "
                  f"{esc('; '.join(v['reasons']))} | {held} | {v['claim']} | "
                  f"{esc((v['first_failure'] or v['error'] or '')[:120])} |")
    md += ["", "## Calibration", "",
           "| confidence said | verdicts | mean confidence | passed held-out |", "|---|---|---|---|"]
    for lo, hi in BUCKETS:
        b = [v for v in scored if lo <= v["confidence"] < hi]
        if b:
            md.append(f"| {lo:.2f}–{min(hi, 1):.2f} | {len(b)} | "
                      f"{sum(v['confidence'] for v in b) / len(b):.2f} | "
                      f"{sum(ok(v) for v in b)}/{len(b)} |")
    if brier is not None:
        md += ["", f"**Brier score {brier:.3f}** over {len(scored)} verdicts (0 is perfect; always "
                   f"saying 0.5 scores 0.25). **Confident (≥ 0.5) but failed held-out: {len(over)}.**"]
    solved = [v for v in scored if v["solved"]]
    if solved:
        sb = sum((v["confidence"] - ok(v)) ** 2 for v in solved) / len(solved)
        md += ["", f"Unsolved runs are confidence 0 and fail held-out by construction, so they pull the "
                   f"Brier score toward 0. Over the {len(solved)} solved run(s) alone, where the "
                   f"confidence is a real prediction, the Brier score is **{sb:.3f}**."]
        md += ["", "## Solved runs: did they hold up?", ""]
        for v in solved:
            md.append(f"- run {v['run'] + 1}, level {v['level']} (best kernel from round "
                      f"{v['best_round']}): confidence {v['confidence']:.2f}; held-out "
                      f"{v['heldout_passed']}/{v['heldout_total']}, "
                      + ("**passes every case**." if ok(v) else
                         f"**fails**, first: {esc(v['first_failure'])}."))
    if a.note:
        md += ["", "## Notes", ""] + [f"- {n}" for n in a.note]
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n{len(rows)} verdicts, Brier {brier if brier is None else round(brier, 3)}, "
          f"confident-but-wrong {len(over)} -> {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
