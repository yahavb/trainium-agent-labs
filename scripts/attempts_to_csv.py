#!/usr/bin/env python3
"""Turn agent attempts.jsonl into spreadsheets for failure classification.

    python scripts/attempts_to_csv.py runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl -o analysis/baseline

writes <out>_attempts.csv (one row per attempt, with a guessed category to correct by hand) and
<out>_summary.csv (category x level counts). UTF-8 with BOM so Excel / WPS / Tencent Docs open it.
"""
import argparse, collections, csv, json, os, re, sys

# (category, pattern on the feedback text) — first match wins. Order matters.
CATEGORIES = [
    ("正确",               r"^Correct on every shape"),
    ("编造不存在的函数",     r"has no attribute"),
    ("拷贝两边大小不一致",   r"same number of elements"),
    ("内存位置放错(SBUF/PSUM)", r"dst must be in \["),
    ("函数参数用错",        r"unexpected keyword argument|not callable|missing \d+ required|takes \d+ positional"),
    ("tile 只有一维",       r"at least 2 dimensions"),
    ("tile 超过 128 行",    r"partition dimension \d+ exceeds|exceeds the maximum of 128|exceeds maximum 128"),
    ("下标越界",           r"Out-of-bound access"),
    ("乱用 reshape",       r"cannot reshape"),
    ("形状不匹配/广播错误",  r"shape mismatch|could not be broadcast|operands could not"),
    ("能运行但数值算错",     r"NUMERICAL MISMATCH|mismatch"),
    ("违反规则(静态检查)",   r"Rule violations|banned|not decorated|no function named"),
    ("代码无法解析",        r"does not parse|SyntaxError|No code came back"),
]

def categorize(feedback):
    for name, pat in CATEGORIES:
        if re.search(pat, feedback):
            return name
    return "其他"

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
        w.writerow(["编号", "第几次run", "level", "level名称", "轮次", "样本", "分数",
                    "能解析", "规则通过", "能运行", "结果正确",
                    "错误类别(自动猜)", "错误类别(人工校对)", "报错原文(英文)", "备注"])
        for i, (run_i, r) in enumerate(rows, 1):
            key = (run_i, r["level"], r["round"])
            sample[key] += 1
            p = r.get("parts", {})
            cat = categorize(r["feedback"])
            out_rows.append((r["level"], cat))
            w.writerow([i, run_i, r["level"], names.get(str(r["level"]), ""), r["round"], sample[key],
                        f'{r["reward"]:.2f}',
                        *("是" if p.get(k) else "否" for k in ("parses", "rules", "runs", "correct")),
                        cat, "", error_text(r["feedback"])[:500], ""])

    levels = sorted({lvl for lvl, _ in out_rows}, key=int)
    counts = collections.Counter(out_rows)
    with open(a.out + "_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["错误类别"] + [f"level {l}" for l in levels] + ["合计"])
        cats = sorted({c for _, c in out_rows}, key=lambda c: -sum(counts[(l, c)] for l in levels))
        for c in cats:
            n = [counts[(l, c)] for l in levels]
            w.writerow([c] + n + [sum(n)])
        w.writerow(["合计"] + [sum(counts[(l, c)] for c in cats) for l in levels] + [len(out_rows)])

    print(f"{len(out_rows)} attempts, {run} run(s) -> {a.out}_attempts.csv, {a.out}_summary.csv")

if __name__ == "__main__":
    main()
