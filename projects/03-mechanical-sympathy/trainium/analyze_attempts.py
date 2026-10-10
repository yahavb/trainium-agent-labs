#!/usr/bin/env python3
"""Turn agent attempt logs into the tables for the write-up.

    python analyze_attempts.py attempts_v1.jsonl attempts_v2.jsonl

Prints (as markdown, ready to paste):
  1. summary per log: runs, solve rate, rounds to solve, attempts, mean / best reward
  2. failure taxonomy: every attempt classified into one named failure mode, with counts
  3. mean and best reward per round (does feedback move the model, round by round?)
  4. the wall: the failure mode each run ended on
Each attempt is classified from the checker's own feedback text, so the taxonomy is exactly what
the model was told.
"""
import json, re, sys
from collections import Counter, defaultdict

# (mode, regex on the feedback); first match wins, order matters
MODES = [
    ("solved", r"^Correct on every shape"),
    ("no code / empty reply", r"No code came back"),
    ("syntax error", r"does not parse"),
    ("forbidden import", r"are not allowed"),
    ("numpy/torch inside kernel", r"The kernel uses \["),
    ("missing @nki.jit / wrong name", r"@nki\.jit|no top-level function"),
    ("could not load", r"could not be loaded|There is no module named"),
    # simulator errors: the checker's message always contains "raised <ExceptionType>"
    ("1-D tile (needs 2 dims)", r"raised .*at least 2 dimensions"),
    ("more than 128 partitions", r"raised .*(partition[^.]*(128|exceeds)|exceeds maximum 128)"),
    ("invented API name", r"raised .*(has no attribute|is not defined|not a function|cannot import)"),
    ("wrong arguments", r"raised .*(unexpected keyword|missing \d+ required|takes \d+ positional)"),
    ("other simulator error", r"raised "),
    ("wrong output shape", r"Output shape is"),
    ("NaN / inf", r"NaN/inf"),
    ("nothing computed (copy)", r"equals the input"),
    ("cap missing", r"except the cap|apply the cap"),
    ("partial last tile skipped", r"rows >= 128 are wrong"),
    ("GELU missing", r"without GELU"),
    ("unbiased variance", r"looks unbiased"),
    ("other wrong values", r"Wrong values"),
    ("checker error", r"CHECKER ERROR"),
]


def mode_of(rec):
    fb = rec.get("feedback", "")
    if rec.get("reward", 0) >= 1.0:
        return "solved"
    for name, rx in MODES:
        if re.search(rx, fb):
            return name
    return "unclassified"


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    for r in recs:
        r["mode"] = mode_of(r)
        r.setdefault("feedback_version", "v1")
        r.setdefault("prompt_version", "v1")
    return recs


def main(paths):
    data = {p: load(p) for p in paths}
    print("## 1. Summary\n")
    print("| Log | Version | Runs | Solved | Rounds to solve | Attempts | Mean reward | Best reward |")
    print("|---|---|---|---|---|---|---|---|")
    walls = {}
    for p, recs in data.items():
        runs = defaultdict(list)
        for r in recs:
            runs[r["run"]].append(r)
        solved = {k: min(x["round"] for x in v if x["reward"] >= 1.0)
                  for k, v in runs.items() if any(x["reward"] >= 1.0 for x in v)}
        fbv = ",".join(sorted({f"feedback {r['feedback_version']}, prompt {r['prompt_version']}"
                               for r in recs}))
        print(f"| {p} | {fbv} | {len(runs)} | {len(solved)}/{len(runs)} | "
              f"{sorted(solved.values()) or '-'} | {len(recs)} | "
              f"{sum(r['reward'] for r in recs) / max(len(recs), 1):.2f} | "
              f"{max((r['reward'] for r in recs), default=0):.2f} |")
        walls[p] = {k: ("solved" if k in solved else
                        Counter(x["mode"] for x in v if x["round"] == max(y["round"] for y in v))
                        .most_common(1)[0][0]) for k, v in runs.items()}

    print("\n## 2. Failure taxonomy (every attempt, by the checker's feedback)\n")
    modes = [m for m, _ in MODES] + ["unclassified"]
    counts = {p: Counter(r["mode"] for r in recs) for p, recs in data.items()}
    print("| Failure mode | " + " | ".join(paths) + " |")
    print("|---|" + "---|" * len(paths))
    for m in modes:
        if any(counts[p][m] for p in paths):
            print(f"| {m} | " + " | ".join(
                f"{counts[p][m]} ({100 * counts[p][m] / max(len(data[p]), 1):.0f}%)" for p in paths) + " |")

    print("\n## 3. Reward by round (mean / best over all runs and samples)\n")
    rounds = sorted({r["round"] for recs in data.values() for r in recs})
    print("| Round | " + " | ".join(paths) + " |")
    print("|---|" + "---|" * len(paths))
    for k in rounds:
        cells = []
        for p in paths:
            rr = [r["reward"] for r in data[p] if r["round"] == k]
            cells.append(f"{sum(rr) / len(rr):.2f} / {max(rr):.2f} (n={len(rr)})" if rr else "-")
        print(f"| {k} | " + " | ".join(cells) + " |")

    print("\n## 4. The wall each run ended on\n")
    for p in paths:
        print(f"- {p}: " + ", ".join(f"{k}: {w}" for k, w in sorted(walls[p].items())))

    un = [r["feedback"][:150] for recs in data.values() for r in recs if r["mode"] == "unclassified"]
    if un:
        print("\nUnclassified feedback (add a rule for these):")
        for u in un[:10]:
            print("  - " + u)


if __name__ == "__main__":
    main(sys.argv[1:] or ["attempts_samudra.jsonl"])
