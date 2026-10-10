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
import usage  # noqa: E402

FULL = 1.0 - 1e-9
CLAIMS = ("VERIFIED", "PASSES THE LOOP'S SHAPES ONLY", "NOT SOLVED", "UNVERIFIED")
NKI_CLAIMS = ("VERIFIED", "SIMULATOR ONLY", "DOUBTFUL", "REJECTED BY COMPILER", "FAILED", "UNVERIFIED")


def read_jsonl(paths):
    out = collections.defaultdict(list)
    for p in paths:
        for line in open(p):
            if line.strip():
                v = json.loads(line)
                out[v["level"]].append(v)
    return out


def pair_verdicts(ours, theirs):
    """Pair each of feedback_v7's NKI_VERDICTS lines with agent.py's verdict for the same level-run.

    v7 writes its verdict as solve() returns and agent.py states its confidence right after, so when
    both carry a time (v7: `t`, ours: `confidence_time`) each v7 verdict pairs with the first of ours
    at or after it, within 10 minutes. Otherwise (older verdicts without a time) they pair by order
    within each level, so pass the files in the same order. Both record the level's rounds, which
    must agree. Anything left over is reported, never dropped."""
    pairs, unpaired = collections.defaultdict(list), []
    for lv in sorted(set(ours) | set(theirs)):
        a, b = ours.get(lv, []), theirs.get(lv, [])
        timed = a and b and all("confidence_time" in x for x in a) and all("t" in y for y in b)
        if timed:
            used, matched = set(), []
            for y in sorted(b, key=lambda y: y["t"]):
                cands = [i for i, x in enumerate(a) if i not in used
                         and -2 <= x["confidence_time"] - y["t"] <= 600]
                i = min(cands, key=lambda i: a[i]["confidence_time"]) if cands else None
                if i is None:
                    unpaired.append(f"level {lv}: v7 only ({y.get('status')}, t={y['t']})")
                else:
                    used.add(i)
                    matched.append((a[i], y))
            unpaired += [f"level {lv}: ours only (run {x.get('run', 0) + 1}, {x.get('claim')})"
                         for i, x in enumerate(a) if i not in used]
        else:
            matched = list(zip(a, b))
            unpaired += [f"level {lv} #{k + 1}: ours only ({a[k].get('claim')})"
                         for k in range(len(b), len(a))]
            unpaired += [f"level {lv} #{k + 1}: v7 only ({b[k].get('status')})"
                         for k in range(len(a), len(b))]
        for x, y in matched:
            if x.get("rounds") is not None and y.get("rounds") is not None \
                    and x["rounds"] != y["rounds"]:
                unpaired.append(f"level {lv} run {x.get('run', 0) + 1}: rounds differ (ours "
                                f"{x['rounds']}, v7 {y['rounds']}) -- not paired")
            else:
                pairs[lv].append((x, y))
    return pairs, unpaired


def scored_stats(pairs, side):
    """Mean confidence, Brier and confident-but-wrong for one side, against OUR held-out result."""
    s = [(x if side == "ours" else y)["confidence"] for x, y in pairs if x.get("heldout_total")]
    ok = [x["heldout_passed"] == x["heldout_total"] for x, y in pairs if x.get("heldout_total")]
    if not s:
        return None
    return dict(n=len(s), conf=sum(s) / len(s),
                brier=sum((c - o) ** 2 for c, o in zip(s, ok)) / len(s),
                over=sum(1 for c, o in zip(s, ok) if c >= 0.5 and not o))


def runs_by_level(paths, queues=None):
    """{level: [attempts of run 1, attempts of run 2, ...]} in file order. With queues (a usage
    log), token fields are replaced by the server's counts where an attempt matches."""
    out = collections.defaultdict(list)
    for (_, _, level), atts in taxonomy.load_attempts(paths).items():
        if queues is not None:
            usage.apply(atts, queues)
        out[level].append(atts)
    return out


def run_stats(atts):
    best = max(r["reward"] for r in atts)
    solve = next(((i + 1, r["round"]) for i, r in enumerate(atts) if r["reward"] >= FULL), None)
    has_tokens = all("prompt_tokens" in r for r in atts)
    tokens = ((sum(r["prompt_tokens"] or 0 for r in atts),
               sum(r["completion_tokens"] or 0 for r in atts)) if has_tokens else None)
    unmatched = sum(1 for r in atts if r.get("usage_matched") is False)
    return dict(best=best, solve=solve, tokens=tokens, unmatched=unmatched)


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
    ap.add_argument("--usage", nargs="*", default=None,
                    help="USAGE_LOG files (feedback_v5+): exact server token counts")
    ap.add_argument("--nki-verdicts", nargs="*", default=None,
                    help="feedback_v7's NKI_VERDICTS files, in the same order as --verdicts")
    a = ap.parse_args()

    levels = runs_by_level(a.attempts, usage.load(a.usage) if a.usage is not None else None)
    base = runs_by_level(a.baseline) if a.baseline else {}
    verdicts = collections.defaultdict(list)
    for p in a.verdicts:
        for line in open(p):
            if line.strip():
                v = json.loads(line)
                verdicts[v["level"]].append(v)

    if a.nki_verdicts is not None:
        nki = read_jsonl(a.nki_verdicts)
        pairs, unpaired = pair_verdicts(verdicts, nki)
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
                   tokens="; ".join((f"{s['tokens'][0]:,}+{s['tokens'][1]:,}"
                                     + (f" ({s['unmatched']} est.)" if s["unmatched"] else ""))
                                    if s["tokens"] else "n/a" for s in st))
        if vs:
            row.update(verdicts=vs["n"],
                       claims=", ".join(f"{c} {vs['claims'][c]}" for c in CLAIMS
                                        if vs["claims"][c]),
                       confidence=fmt(vs["confidence"]),
                       brier="-" if vs["brier"] is None else fmt(vs["brier"], 3),
                       confident_but_wrong=vs["over"])
        if a.nki_verdicts is not None:
            nv = nki.get(lv, [])
            st2 = scored_stats(pairs.get(lv, []), "v7")
            row.update(
                nki_claims=", ".join(f"{c} {sum(1 for v in nv if v.get('status') == c)}"
                                     for c in NKI_CLAIMS if any(v.get("status") == c for v in nv))
                or "-",
                nki_confidence=fmt(sum(v["confidence"] for v in nv) / len(nv)) if nv else "-",
                nki_brier=fmt(st2["brier"], 3) if st2 else "-",
                nki_confident_but_wrong=st2["over"] if st2 else "-")
        if base:
            bs = [run_stats(x) for x in base.get(lv, [])]
            row["baseline"] = (f"{sum(1 for s in bs if s['solve'])}/{len(bs)}, mean "
                               f"{fmt(sum(s['best'] for s in bs) / len(bs))}" if bs else "-")
        rows.append(row)

    cols = ["level", "runs", "solved", "scores", "mean", "min", "max", "solves", "tokens",
            "verdicts", "claims", "confidence", "brier", "confident_but_wrong", "nki_claims",
            "nki_confidence", "nki_brier", "nki_confident_but_wrong", "baseline"]
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
            "confident_but_wrong": "confident (>=0.5) but wrong", "nki_claims": "v7 verdicts",
            "nki_confidence": "v7 mean confidence", "nki_brier": "v7 Brier (vs our held-out)",
            "nki_confident_but_wrong": "v7 confident but failed our held-out",
            "baseline": "baseline"}
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
    if a.nki_verdicts is not None:
        cmp_rows = []
        for lv in sorted(pairs):
            o, v = scored_stats(pairs[lv], "ours"), scored_stats(pairs[lv], "v7")
            if o:
                cmp_rows.append(f"| {lv} | {o['n']} | {fmt(o['conf'])} | {fmt(o['brier'], 3)} | "
                                f"{o['over']} | {fmt(v['conf'])} | {fmt(v['brier'], 3)} | {v['over']} |")
        block = ["## Two verdicts, one yardstick", "",
                 "Both verdicts after every level, scored against the same outcome: did the kernel "
                 "pass agent.py's held-out set (nkibench `EVAL_SHAPES` x `VALUE_KINDS`). Only pairs "
                 "with a held-out result count. Ours: confidence from `agent.confidence()`. v7: "
                 "`verdict_nki.py`'s confidence (extra hostile cases, compiler-risk scan, lowering).",
                 "", "| level | pairs scored | ours: mean confidence | ours: Brier | ours: confident "
                 "but wrong | v7: mean confidence | v7: Brier | v7: confident but wrong |",
                 "|---|---|---|---|---|---|---|---|"] + (cmp_rows or ["| - | 0 | | | | | | |"])
        block += ["", "Unpaired verdicts: " + ("; ".join(unpaired) if unpaired else "none") + ".", ""]
        i = md.index("## Inputs")
        md[i:i] = block
    if a.usage is not None:
        allr = [r for runs in levels.values() for x in runs for r in x]
        n_un = sum(1 for r in allr if r.get("usage_matched") is False)
        md.insert(md.index("## Inputs"),
                  f"Tokens: server counts from the usage log for {len(allr) - n_un} of {len(allr)} "
                  f"attempts; {n_un} unmatched kept chars/4 estimates (marked \"est.\" per run).\n")
    for role, paths in (("attempts", a.attempts), ("verdicts", a.verdicts),
                        ("baseline", a.baseline), ("usage", a.usage or []),
                        ("v7 verdicts", a.nki_verdicts or [])):
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
