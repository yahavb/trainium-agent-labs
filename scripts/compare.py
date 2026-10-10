#!/usr/bin/env python3
"""Configurations x levels in one table: which agent version did what.

    python scripts/compare.py LABEL=DIR[,DIR...] [LABEL=DIR ...] [--ref LABEL] [--match REGEX] -o OUT

Each LABEL is one configuration; its DIRs are searched like report.py does (files classified by
content: attempts, agent verdicts, USAGE_LOG). Per configuration and level, one cell:

    solved/runs · first-solve round per run · minutes per run · truncated attempts ·
    distinct trajectories · held-out claims · diverges from REF at round r

- minutes and truncation need USAGE_LOG (feedback_v5+); old logs show "n/a" for minutes, and
  truncation only where the attempts log carries finish.
- a trajectory is a run's sequence of rounds, each round the set of sha1s of its samples' code.
  "distinct trajectories" counts different sequences among the runs.
- divergence: for each run, the longest round-by-round common prefix with any run of the reference
  configuration on the same level; the cell shows the round where they first differ (min-max over
  runs), "same" if a run equals a reference run all the way.
Writes OUT.md and OUT.csv.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report    # noqa: E402  (file discovery)
import taxonomy  # noqa: E402
import usage     # noqa: E402

FULL = 1.0 - 1e-9
SHORT = {"VERIFIED": "V", "PASSES THE LOOP'S SHAPES ONLY": "LOOP-ONLY", "NOT SOLVED": "NS",
         "UNVERIFIED": "U"}


def sha(code):
    return hashlib.sha1((code or "").encode()).hexdigest()[:10]


def trajectory(atts):
    rounds = collections.OrderedDict()
    for r in atts:
        rounds.setdefault(r["round"], []).append(sha(r.get("code")))
    return tuple(tuple(sorted(v)) for _, v in sorted(rounds.items()))


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load_config(dirs, match):
    files, logs = report.find(dirs, match)
    rows, problems = report.checks(files, logs)
    problems = problems + [f"level {lv}: {'; '.join(fl)}" for lv, *_, fl in rows
                           if any(not f.startswith(("no verdicts", "no v7 verdicts", "no usage"))
                                  for f in fl)]
    eps = taxonomy.load_attempts(files["attempts"]) if files["attempts"] else {}
    q = usage.load(files["usage"]) if files["usage"] else None
    runs = collections.defaultdict(list)
    for (_, _, lv), atts in eps.items():
        if q is not None:
            usage.apply(atts, q)
        runs[lv].append(atts)
    verdicts = collections.defaultdict(list)
    for p in files["verdicts"]:
        for line in open(p):
            if line.strip():
                v = json.loads(line)
                verdicts[v["level"]].append(v.get("claim") or v.get("status", "").split(":")[0])
    return dict(files=files, runs=runs, verdicts=verdicts, problems=problems)


def level_stats(runs, verdicts, ref_runs):
    solved, first, minutes, trunc = 0, [], [], 0
    for atts in runs:
        hit = next((r for r in atts if r["reward"] >= FULL), None)
        solved += hit is not None
        first.append(str(hit["round"]) if hit else "-")
        timed = [r for r in atts if r.get("request_t") is not None]
        minutes.append(f"{(max(r['request_t'] + (r['request_seconds'] or 0) for r in timed) - min(r['request_t'] for r in timed)) / 60:.1f}"
                       if timed else "n/a")
        trunc += sum(1 for r in atts if r.get("finish") == "length")
    trajs = [trajectory(a) for a in runs]
    div = "-"
    if ref_runs is not None:
        ref_t = [trajectory(a) for a in ref_runs]
        if ref_t:
            points = []
            for t in trajs:
                best = max(ref_t, key=lambda u: common_prefix(t, u))
                k = common_prefix(t, best)
                points.append("same" if t == best else k)
            nums = [p for p in points if p != "same"]
            div = ("same" if not nums else
                   (f"r{min(nums)}" if min(nums) == max(nums) else f"r{min(nums)}-{max(nums)}")
                   + (f" ({points.count('same')} same)" if "same" in points else ""))
        else:
            div = "no ref runs"
    claims = collections.Counter(verdicts)
    return dict(runs=len(runs), solved=solved, first=",".join(first), minutes=" ".join(minutes),
                truncated=trunc, distinct=len(set(trajs)),
                heldout=" ".join(f"{SHORT.get(c, c)}{n}" for c, n in claims.most_common()) or "-",
                diverge=div)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="LABEL=DIR[,DIR...]")
    ap.add_argument("--ref", default=None, help="label to measure divergence from (default: first)")
    ap.add_argument("--match", default="", help="regex on file names to keep (as report.py)")
    ap.add_argument("-o", "--out", default="analysis/compare")
    a = ap.parse_args()

    configs = collections.OrderedDict()
    for c in a.configs:
        label, _, dirs = c.partition("=")
        if not dirs:
            sys.exit(f"expected LABEL=DIR[,DIR...], got {c!r}")
        configs[label] = load_config(dirs.split(","), a.match)
    ref = a.ref or next(iter(configs))
    if ref not in configs:
        sys.exit(f"--ref {ref!r} is not one of {list(configs)}")
    levels = sorted({lv for c in configs.values() for lv in c["runs"]})

    table, flat = {}, []
    for label, c in configs.items():
        for lv in levels:
            if lv not in c["runs"]:
                continue
            ref_runs = None if label == ref else configs[ref]["runs"].get(lv, [])
            s = level_stats(c["runs"][lv], c["verdicts"].get(lv, []), ref_runs)
            table[(label, lv)] = s
            flat.append(dict(config=label, level=lv, **s))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["config", "level", "runs", "solved", "first", "minutes",
                                          "truncated", "distinct", "heldout", "diverge"])
        w.writeheader()
        w.writerows(flat)

    def cell(s):
        return (f"**{s['solved']}/{s['runs']}** · first 1.0 r{s['first']} · min {s['minutes']} · "
                f"trunc {s['truncated']} · traj {s['distinct']} · {s['heldout']} · div {s['diverge']}")

    md = ["# Configurations compared", "",
          f"Divergence is measured from **{ref}**. Cell: solved/runs · round of the first 1.0 per run "
          "(- = never) · minutes per run (USAGE_LOG; n/a without it) · attempts cut off by max_tokens "
          "· distinct trajectories among the runs · held-out claims (V verified, LOOP-ONLY passes the "
          "loop's shapes only, NS not solved, U unverified) · first round whose code differs from the "
          "reference's runs.", "",
          "| config | " + " | ".join(f"L{lv}" for lv in levels) + " |",
          "|---|" + "---|" * len(levels)]
    for label in configs:
        md.append(f"| {label} | " + " | ".join(cell(table[(label, lv)]) if (label, lv) in table
                                              else "-" for lv in levels) + " |")
    md += ["", "## Checks (from report.py; missing verdicts or usage logs are not listed)", ""]
    md += [f"- **{label}**: {p}" for label, c in configs.items() for p in c["problems"]] or ["none"]
    md += ["", "## Inputs", ""]
    for label, c in configs.items():
        for k in ("attempts", "verdicts", "usage"):
            for p in c["files"][k]:
                md.append(f"- {label} {k}: `{p}`")
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:6 + len(configs)]))
    for label, c in configs.items():
        for p in c["problems"]:
            print(f"  CHECK {label}: {p}")
    print(f"\n-> {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
