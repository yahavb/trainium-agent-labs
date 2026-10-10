#!/usr/bin/env python3
"""Turn agent attempts.jsonl into spreadsheets for failure classification.

    python scripts/attempts_to_csv.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl -o analysis/baseline

writes <out>_attempts.csv (one row per attempt, with a guessed category to correct by hand) and
<out>_summary.csv (category x level counts). UTF-8 with BOM so Excel / WPS / Tencent Docs open it.
"""
import argparse, collections, csv, json, os, re, sys

# (category, pattern on the feedback text) — first match wins. Order matters.
CATEGORIES = [
    ("correct",               r"^Correct on every shape"),
    ("invented_name",         r"has no attribute"),
    ("copy_size_mismatch",    r"same number of elements"),
    ("wrong_buffer",          r"dst must be in \["),
    ("wrong_signature",       r"unexpected keyword argument|not callable|missing \d+ required|takes \d+ positional"),
    ("tile_1d",               r"at least 2 dimensions"),
    ("partition_over_128",    r"partition dimension \d+ exceeds|exceeds the maximum of 128|exceeds maximum 128"),
    ("out_of_bounds",         r"Out-of-bound access"),
    ("reshape",               r"cannot reshape"),
    ("broadcast",             r"shape mismatch|could not be broadcast|operands could not"),
    ("numeric_mismatch",      r"NUMERICAL MISMATCH|mismatch"),
    ("rule_violation",        r"Rule violations|banned|not decorated|no function named"),
    ("does_not_parse",        r"does not parse|SyntaxError|No code came back"),
]

def categorize(feedback):
    for name, pat in CATEGORIES:
        if re.search(pat, feedback):
            return name
    return "other"

def error_text(feedback):
    m = re.search(r"raised (\w+: .*)", feedback, re.S)
    return (m.group(1) if m else feedback).strip()

def level_names():
    here = os.path.join(os.path.dirname(__file__), "..", "projects", "02-kernel-agent")
    sys.path.insert(0, os.path.abspath(here))
    try:
        import nkibench
        return {str(k): v["op"] for k, v in nkibench.LEVELS.items()}
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("-o", "--out", default="analysis/attempts")
    a = ap.parse_args()
    names = level_names()

    rows, run, prev_level = [], 0, None
    for path in a.jsonl:
        prev_level = None
        run += 1  # each file starts a new run sequence
        for line in open(path):
            r = json.loads(line)
            lvl = int(r["level"])
            if prev_level is not None and lvl < prev_level:
                run += 1  # --repeat: levels start over at the lowest one
            prev_level = lvl
            rows.append((run, r))

    sample = collections.Counter()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out_rows = []
    with open(a.out + "_attempts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "run", "level", "level_name", "round", "sample", "score",
                    "parses", "rules_pass", "runs", "correct",
                    "category_guess", "category_checked", "feedback_text", "notes"])
        for i, (run_i, r) in enumerate(rows, 1):
            key = (run_i, r["level"], r["round"])
            sample[key] += 1
            p = r.get("parts", {})
            cat = categorize(r["feedback"])
            out_rows.append((r["level"], cat))
            w.writerow([i, run_i, r["level"], names.get(str(r["level"]), ""), r["round"], sample[key],
                        f'{r["reward"]:.2f}',
                        *("yes" if p.get(k) else "no" for k in ("parses", "rules", "runs", "correct")),
                        cat, "", error_text(r["feedback"])[:500], ""])

    levels = sorted({lvl for lvl, _ in out_rows}, key=int)
    counts = collections.Counter(out_rows)
    with open(a.out + "_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["category"] + [f"level {l}" for l in levels] + ["total"])
        cats = sorted({c for _, c in out_rows}, key=lambda c: -sum(counts[(l, c)] for l in levels))
        for c in cats:
            n = [counts[(l, c)] for l in levels]
            w.writerow([c] + n + [sum(n)])
        w.writerow(["total"] + [sum(counts[(l, c)] for c in cats) for l in levels] + [len(out_rows)])

    print(f"{len(out_rows)} attempts, {run} run(s) -> {a.out}_attempts.csv, {a.out}_summary.csv")

if __name__ == "__main__":
    main()
