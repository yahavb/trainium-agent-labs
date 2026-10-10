#!/usr/bin/env python3
"""The final numbers: one row per level, from any number of attempt logs, in about a second.

    python scripts/summarize.py final_L1a.jsonl final_L1b.jsonl final_L2.jsonl ... \
        --verdicts verdicts_*.jsonl --baseline runs/seat-116/latest/.../attempts.jsonl \
        -o analysis/summary_final

Runs are taken in file order and numbered per level across files, so a level split over several
files (final_L1a, final_L1b) is simply joined. Old logs (no run field) and new ones both read; the
split is taxonomy.load_attempts, shared with taxonomy.py and calibrate.py. Scores are the logged
rewards; logs from before c39c0ce should go through scripts/calibrate.py (re-grade) first.

Per level: runs, solved x/n, every run's best score, mean/min/max, for each solve the attempts and
the round it took, tokens per run (n/a for old logs), the held-out claims, mean confidence, Brier,
and confident-but-wrong. Ends with every input file's path, md5 and last commit if tracked.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taxonomy  # noqa: E402

FULL = 1.0 - 1e-9
CLAIMS = ("VERIFIED", "PASSES THE LOOP'S SHAPES ONLY", "NOT SOLVED", "UNVERIFIED")


def runs_by_level(paths):
    """{level: [attempts of run 1, attempts of run 2, ...]} in file order."""
    out = collections.defaultdict(list)
    for (_, _, level), atts in taxonomy.load_attempts(paths).items():
        out[level].append(atts)
    return out


def run_stats(atts):
    best = max(r["reward"] for r in atts)
    solve = next(((i + 1, r["round"]) for i, r in enumerate(atts) if r["reward"] >= FULL), None)
    has_tokens = all("prompt_tokens" in r for r in atts)
    tokens = ((sum(r["prompt_tokens"] or 0 for r in atts),
               sum(r["completion_tokens"] or 0 for r in atts)) if has_tokens else None)
    return dict(best=best, solve=solve, tokens=tokens)


def claim_of(v):
    if v.get("claim"):
        return v["claim"]
    s = v.get("status", "")          # verdicts from 7868c08 have no claim field
    return next((c for c in CLAIMS if s.startswith(c)), "UNVERIFIED")


def verdict_stats(vs):
    if not vs:
        return None
    claims = collections.Counter(claim_of(v) for v in vs)
    scored = [v for v in vs if v.get("heldout_total")]
    ok = lambda v: v["heldout_passed"] == v["heldout_total"]
    brier = (sum((v["confidence"] - ok(v)) ** 2 for v in scored) / len(scored)) if scored else None
    return dict(n=len(vs), claims=claims,
                confidence=sum(v["confidence"] for v in vs) / len(vs), brier=brier,
                over=sum(1 for v in scored if v["confidence"] >= 0.5 and not ok(v)))


def provenance(path):
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", os.path.abspath(path)],
                                capture_output=True, text=True, timeout=10,
                                cwd=os.path.dirname(os.path.abspath(path))).stdout.strip()
    except Exception:
        commit = ""
    return md5, commit or "untracked"


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+")
    ap.add_argument("--verdicts", nargs="*", default=[])
    ap.add_argument("--baseline", nargs="*", default=[], help="attempt logs to compare against")
    ap.add_argument("-o", "--out", default="analysis/summary")
    a = ap.parse_args()

    levels = runs_by_level(a.attempts)
    base = runs_by_level(a.baseline) if a.baseline else {}
    verdicts = collections.defaultdict(list)
    for p in a.verdicts:
        for line in open(p):
            if line.strip():
                v = json.loads(line)
                verdicts[v["level"]].append(v)

    rows = []
    for lv in sorted(levels):
        st = [run_stats(x) for x in levels[lv]]
        scores = [s["best"] for s in st]
        solved = [i for i, s in enumerate(st) if s["solve"]]
        vs = verdict_stats(verdicts.get(lv))
        row = dict(level=lv, runs=len(st), solved=len(solved),
                   scores=" ".join(fmt(x) for x in scores),
                   mean=fmt(sum(scores) / len(scores)), min=fmt(min(scores)), max=fmt(max(scores)),
                   solves="; ".join(f"run {i + 1}: attempt {st[i]['solve'][0]}, round "
                                    f"{st[i]['solve'][1]}" for i in solved) or "-",
                   tokens="; ".join(f"{s['tokens'][0]:,}+{s['tokens'][1]:,}" if s["tokens"]
                                    else "n/a" for s in st))
        if vs:
            row.update(verdicts=vs["n"],
                       claims=", ".join(f"{c} {vs['claims'][c]}" for c in CLAIMS
                                        if vs["claims"][c]),
                       confidence=fmt(vs["confidence"]),
                       brier="-" if vs["brier"] is None else fmt(vs["brier"], 3),
                       confident_but_wrong=vs["over"])
        if base:
            bs = [run_stats(x) for x in base.get(lv, [])]
            row["baseline"] = (f"{sum(1 for s in bs if s['solve'])}/{len(bs)}, mean "
                               f"{fmt(sum(s['best'] for s in bs) / len(bs))}" if bs else "-")
        rows.append(row)

    cols = ["level", "runs", "solved", "scores", "mean", "min", "max", "solves", "tokens",
            "verdicts", "claims", "confidence", "brier", "confident_but_wrong", "baseline"]
    cols = [c for c in cols if any(c in r for r in rows)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

    head = {"level": "level", "runs": "runs", "solved": "solved", "scores": "best score per run",
            "mean": "mean", "min": "min", "max": "max", "solves": "solve: attempts / round",
            "tokens": "tokens per run (prompt+answer)", "verdicts": "verdicts",
            "claims": "held-out claims", "confidence": "mean confidence", "brier": "Brier",
            "confident_but_wrong": "confident (>=0.5) but wrong", "baseline": "baseline"}
    md = ["# Results by level", "",
          "| " + " | ".join(head[c] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(
            (f"{r['solved']}/{r['runs']}" if c == "solved" else str(r.get(c, "")))
            .replace("|", "\\|") for c in cols) + " |")
    md += ["", "*solve: attempts* counts every attempt in that run up to and including the first 1.0 "
               "(all samples of every earlier round). Scores are the best loop reward per run. Held-out "
               "claims, confidence and Brier are counted per level over every verdict in the "
               "--verdicts files, so pass the verdict files that belong to these runs.", "", "## Inputs", "",
           "| role | file | md5 | last commit |", "|---|---|---|---|"]
    for role, paths in (("attempts", a.attempts), ("verdicts", a.verdicts),
                        ("baseline", a.baseline)):
        for p in paths:
            md5, commit = provenance(p)
            md.append(f"| {role} | `{p}` | `{md5}` | {commit} |")
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    for r in rows:
        print(f"level {r['level']}: solved {r['solved']}/{r['runs']}  [{r['scores']}]  "
              f"mean {r['mean']}" + (f"  claims: {r['claims']}" if r.get("claims") else ""))
    print(f"-> {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
