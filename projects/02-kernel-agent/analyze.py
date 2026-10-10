#!/usr/bin/env python3
"""analyze.py -- turn one or more attempts.jsonl logs into the numbers the judges ask for.

    python analyze.py logs/base_L4.jsonl
    python analyze.py logs/base_*.jsonl logs/doc_tool_*.jsonl     # side by side

Per log and level: solve rate over runs with a 95% Wilson interval (needs the `run` field from
the A1 logging patch in agent.py), best reward per run, rounds-to-solve, prompt sizes, the
prompt-segment breakdown (needs `seg_chars`), and a failure taxonomy from the checker's feedback.
Writes taxonomy.csv next to the first log.

Needs only the Python standard library, so it runs anywhere -- including off the pod.
"""
import collections
import csv
import json
import math
import os
import re
import sys

FULL = 1.0 - 1e-9

# Ordered: the first pattern that matches names the failure class. Extend it as you read logs;
# an attempt that matches nothing is counted as "other", and the tool tells you how many there are.
TAXONOMY = [
    ("no_code",          r"No code came back"),
    ("syntax",           r"does not parse"),
    ("rule_violation",   r"Rule violations"),
    ("framework_cheat",  r"hands the whole operation to a framework"),
    ("invented_module",  r"There is no module named"),
    ("invented_api",     r"has no attribute"),
    ("memregion_call",   r"MemoryRegion' object is not callable"),
    ("bad_kwarg",        r"unexpected keyword argument"),
    ("tile_1d",          r"must have at least 2 dimensions"),
    ("reshape",          r"cannot reshape array"),
    ("partition_gt_128", r"partition dimension \d+ exceeds"),
    ("contraction_gt",   r"contraction dimension \d+ exceeds"),
    ("wrong_buffer",     r"must be in \["),
    ("out_of_bounds",    r"Out-of-bound access"),
    ("dma_size",         r"same number of elements"),
    ("broadcast",        r"could not be broadcast"),
    ("input_mutated",    r"MODIFIED ITS INPUT"),
    ("wrong_shape",      r"WRONG SHAPE"),
    ("non_finite",       r"NON-FINITE"),
    ("output_zeros",     r"OUTPUT IS \d+% ZEROS|of the output is zero"),
    ("ragged_edge",      r"ragged edge"),
    ("hw_hazard",        r"WRONG ON HARDWARE"),
    ("traffic_unmeasured", r"NOT MEASURED|CANNOT CONFIRM THE TRAFFIC"),
    ("traffic_bar",      r"TOO MUCH HBM TRAFFIC"),
    ("numerics",         r"NUMERICAL MISMATCH"),
    ("solved",           r"Correct on every shape"),
]


def classify(feedback):
    for name, pat in TAXONOMY:
        if re.search(pat, feedback or ""):
            return name
    return "other"


def wilson(k, n, z=1.96):
    """95% confidence interval for a solve rate. A rate without an interval is a point estimate
    masquerading as a result, and these intervals are wide at the N a hackathon can afford."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(path, rows, writer):
    print(f"\n==== {os.path.basename(path)}  ({len(rows)} attempts)")
    by_level = collections.defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    for lv in sorted(by_level):
        rs = by_level[lv]
        # Group by (run) if present, else everything is one run.
        runs = collections.defaultdict(list)
        for r in rs:
            runs[r.get("run", 0)].append(r)
        best_per_run = [max(x["reward"] for x in v) for v in runs.values()]
        solved = sum(b >= FULL for b in best_per_run)
        lo, hi = wilson(solved, len(best_per_run))
        rounds_to_solve = [min(x["round"] for x in v if x["reward"] >= FULL)
                           for v in runs.values() if any(x["reward"] >= FULL for x in v)]
        tax = collections.Counter(classify(x["feedback"]) for x in rs)
        print(f"  level {lv}: solved {solved}/{len(best_per_run)} runs "
              f"(95% CI {lo:.2f}-{hi:.2f})  best/run {[round(b, 2) for b in best_per_run]}  "
              f"mean attempt reward {sum(x['reward'] for x in rs) / len(rs):.2f}"
              + (f"  rounds-to-solve {rounds_to_solve}" if rounds_to_solve else ""))
        pc = [x.get("prompt_chars", 0) for x in rs]
        if any(pc):
            print(f"    prompt chars: min {min(pc)} max {max(pc)} mean {sum(pc) // len(pc)}"
                  f"  (~{max(pc) // 4} tokens at the largest)")
        # Prompt-segment breakdown: where the tokens went, averaged over attempts that logged it.
        segs = collections.defaultdict(list)
        for x in rs:
            for k, v in (x.get("seg_chars") or {}).items():
                segs[k].append(v)
        if segs:
            parts = "  ".join(f"{k}:{sum(v) // len(v)}" for k, v in
                              sorted(segs.items(), key=lambda kv: -sum(kv[1])))
            print(f"    prompt segments (mean chars): {parts}")
        for name, n in tax.most_common():
            print(f"    {n:4d}x  {name}")
            writer.writerow([os.path.basename(path), lv, name, n])


def main(paths):
    if not paths:
        sys.exit(__doc__)
    out = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "taxonomy.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log", "level", "failure_class", "count"])
        for p in paths:
            summarize(p, load(p), w)
    print(f"\ntaxonomy written to {out}")
    unknown = sum(classify(r["feedback"]) == "other" for p in paths for r in load(p))
    if unknown:
        print(f"{unknown} attempts classified 'other' -- read them and add a pattern to TAXONOMY.")


if __name__ == "__main__":
    main(sys.argv[1:])
