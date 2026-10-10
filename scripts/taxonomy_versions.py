#!/usr/bin/env python3
"""Failure modes across agent versions: one table, modes x (version, level).

    python scripts/taxonomy_versions.py baseline=a.jsonl,b.jsonl v7=c.jsonl,... -o analysis/taxonomy_versions

Each VERSION=FILES names the attempts files that make up that version (give files, not directories, so
side experiments in the same folder stay out). USAGE_LOG files next to each attempts file are found
by content and joined (usage.apply), so cut-off answers count as `truncated`. Every failed attempt
(reward < 1) is classified with taxonomy.mode_of. Rows: the 10 most frequent modes overall, plus
`truncated` and `transpose_whole_input` if not already in; the rest is summed as "other modes".
Writes OUT.md (the table and a per-version context line) and OUT.csv (every mode).
"""
import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report    # noqa: E402
import taxonomy  # noqa: E402
import usage     # noqa: E402

ALWAYS = ("truncated", "transpose_whole_input")


def load_version(files):
    usage_files = []
    for d in sorted({os.path.dirname(f) for f in files}):
        found, _ = report.find([d], "")
        usage_files += [u for u in found["usage"] if u not in usage_files]
    q = usage.load(usage_files) if usage_files else None
    eps = taxonomy.load_attempts(files)
    counts, attempts, runs, solved = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for (_, _, lv), atts in eps.items():
        if q is not None:
            usage.apply(atts, q)
        runs[lv] += 1
        solved[lv] += any(r["reward"] >= 1 - 1e-9 for r in atts)
        for r in atts:
            attempts[lv] += 1
            if r["reward"] < 1 - 1e-9:
                counts[(taxonomy.mode_of(r), lv)] += 1
    return dict(counts=counts, attempts=attempts, runs=runs, solved=solved, usage=usage_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="+", help="VERSION=FILE[,FILE...]")
    ap.add_argument("-o", "--out", default="analysis/taxonomy_versions")
    a = ap.parse_args()
    vers = collections.OrderedDict()
    for spec in a.versions:
        name, _, files = spec.partition("=")
        vers[name] = load_version(files.split(","))
    cols = [(v, lv) for v, d in vers.items() for lv in sorted(d["runs"])]
    total = collections.Counter()
    for d in vers.values():
        for (m, lv), n in d["counts"].items():
            total[m] += n
    top = [m for m, _ in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]
    top += [m for m in ALWAYS if m not in top]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mode", "family"] + [f"{v} L{lv}" for v, lv in cols] + ["total"])
        for m in sorted(total, key=lambda m: (-total[m], m)):
            w.writerow([m, taxonomy.FAMILY.get(m, "other")] + [vers[v]["counts"][(m, lv)] for v, lv in cols]
                       + [total[m]])
    head = "| mode | " + " | ".join(f"{v} L{lv}" for v, lv in cols) + " | total |"
    md = ["# Failure modes across versions", "",
          "Failed attempts (reward < 1) per failure mode, by version and level. A version's runs per level and "
          "solves are in the last rows. Modes from `scripts/taxonomy.py`; cut-off answers come from USAGE_LOG.", "",
          head, "|---|" + "---|" * (len(cols) + 1)]
    for m in top:
        md.append(f"| `{m}` | " + " | ".join(str(vers[v]["counts"][(m, lv)] or "") for v, lv in cols)
                  + f" | {total[m]} |")
    rest = {(v, lv): sum(n for (m, l2), n in vers[v]["counts"].items() if l2 == lv and m not in top)
            for v, lv in cols}
    md.append("| other modes | " + " | ".join(str(rest[c] or "") for c in cols)
              + f" | {sum(rest.values())} |")
    md.append("| **failed / all attempts** | " + " | ".join(
        f"{sum(n for (m, l2), n in vers[v]['counts'].items() if l2 == lv)}/{vers[v]['attempts'][lv]}"
        for v, lv in cols) + " | |")
    md.append("| **runs solved** | " + " | ".join(f"{vers[v]['solved'][lv]}/{vers[v]['runs'][lv]}"
                                                for v, lv in cols) + " | |")
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
