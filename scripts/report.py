#!/usr/bin/env python3
"""One command from pulled logs to every table: checks, summary, taxonomy, token chart.

    python scripts/report.py OUT_DIR LOG_DIR [LOG_DIR ...] [--match REGEX] [--baseline FILE]

Every *.jsonl under the log dirs is classified by its CONTENT, so any naming works (the baseline's
attempts.jsonl, Ev7_L4.jsonl, final_L1a.jsonl ...):
  attempts      lines with "code" and "round"            (agent.py --log)
  verdicts      lines with "type": "verdict"             (agent.py --verdicts)
  nki_verdicts  lines with "status" and no "type"        (feedback_v7 NKI_VERDICTS)
  usage         lines with "usage" and "code_sha1"       (feedback_v5+ USAGE_LOG)
--match keeps only files whose name matches, e.g. --match '^final_|_final_' for the final run or
--match 'Ev7_|_v7_|nki_verdicts' for round 2. Files are taken in name order, so final_L1a comes
before final_L1b and the runs join in that order.

Then:
1. a check table per level: which of the four kinds are there, runs in the attempts, runs the
   console logs (*.log next to them) started, verdict counts, usage coverage, and any sign that a
   file holds more than one invocation (run numbers restarting; more runs than the console log
   shows). Problems are flagged and the rest still runs.
2. summarize.py (--verdicts --nki-verdicts --usage --baseline), taxonomy.py, token_budget.py
   (--usage), into OUT_DIR.
3. prints OUT_DIR/summary.md.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import taxonomy  # noqa: E402
import usage     # noqa: E402

KINDS = ("attempts", "verdicts", "nki_verdicts", "usage")
RED, GREEN, OFF = ("\033[31m", "\033[32m", "\033[0m") if sys.stdout.isatty() else ("", "", "")
DEFAULT_BASELINE = "runs/seat-116/latest/projects/02-kernel-agent/attempts.jsonl"


def kind_of(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "code" in r and "round" in r:
                    return "attempts"
                if r.get("type") == "verdict":
                    return "verdicts"
                if "usage" in r and "code_sha1" in r:
                    return "usage"
                if "status" in r and "type" not in r:
                    return "nki_verdicts"
                return None
    return None


def find(dirs, match):
    files = collections.defaultdict(list)
    logs = []
    for d in dirs:
        for root, _, names in os.walk(d):
            for n in sorted(names):
                p = os.path.join(root, n)
                if n.endswith(".log"):
                    logs.append(p)
                if not n.endswith(".jsonl") or (match and not re.search(match, n)):
                    continue
                try:
                    k = kind_of(p)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    k = None
                if k:
                    files[k].append(p)
    for k in files:
        files[k].sort(key=lambda p: (os.path.basename(p), p))
    return files, sorted(logs)


def levels_in(path):
    return sorted({json.loads(l)["level"] for l in open(path) if l.strip()})


def console_level_starts(logs):
    """How many times each level was started, per console log (agent.py prints a header)."""
    starts = collections.Counter()
    for p in logs:
        for line in open(p, errors="replace"):
            m = re.match(r"=+ level (\d+):", line)
            if m:
                starts[int(m.group(1))] += 1
    return starts


def checks(files, logs):
    rows, problems = [], []
    per_file_runs = {}
    for p in files["attempts"]:
        eps = taxonomy.load_attempts([p])
        per_file_runs[p] = collections.Counter(lv for _, _, lv in eps)
        runs_seen = [json.loads(l).get("run") for l in open(p) if l.strip()]
        if any(r is not None for r in runs_seen):
            restarts = sum(1 for a, b in zip(runs_seen, runs_seen[1:])
                           if a is not None and b is not None and b < a)
            if restarts:
                problems.append(f"{p}: run numbers restart {restarts}x -- more than one invocation "
                                f"appended to this file")
    att_runs = collections.Counter()
    for c in per_file_runs.values():
        att_runs.update(c)
    started = console_level_starts([l for l in logs
                                    if any(os.path.dirname(l) == os.path.dirname(a)
                                           for a in files["attempts"])])
    count = {k: collections.Counter() for k in ("verdicts", "nki_verdicts")}
    for k in count:
        for p in files[k]:
            for l in open(p):
                if l.strip():
                    count[k][json.loads(l)["level"]] += 1
    q = usage.load(files["usage"]) if files["usage"] else None
    cover = collections.Counter()
    total = collections.Counter()
    if q is not None:
        for p in files["attempts"]:
            rows_ = [json.loads(l) for l in open(p) if l.strip()]
            for r in rows_:
                total[r["level"]] += 1
            usage.apply(rows_, q)
            for r in rows_:
                cover[r["level"]] += bool(r.get("usage_matched"))
    levels = sorted(set(att_runs) | set(count["verdicts"]) | set(count["nki_verdicts"]))
    for lv in levels:
        flags = []
        n = att_runs[lv]
        if not n:
            flags.append("no attempts")
        if started and n > started[lv]:
            flags.append(f"attempts hold {n} runs but the console logs started level {lv} "
                         f"{started[lv]}x: probably an earlier invocation appended")
        for k, label in (("verdicts", "verdicts"), ("nki_verdicts", "v7 verdicts")):
            if not count[k][lv]:
                flags.append(f"no {label}")
            elif count[k][lv] != n:
                flags.append(f"{count[k][lv]} {label} for {n} runs")
        if q is None:
            flags.append("no usage log")
        elif cover[lv] < total[lv]:
            flags.append(f"usage covers {cover[lv]}/{total[lv]} attempts")
        rows.append((lv, n, started.get(lv, "-"), count["verdicts"][lv], count["nki_verdicts"][lv],
                     f"{cover[lv]}/{total[lv]}" if q is not None else "-", flags))
    return rows, problems


def run(cmd, out_note):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-1500:])
    if r.returncode:
        out_note.append(f"`{os.path.basename(cmd[1])}` failed (exit {r.returncode}): "
                        f"{(r.stderr or r.stdout).strip().splitlines()[-1][:200]}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("logdirs", nargs="+")
    ap.add_argument("--match", default="", help="regex on file names to keep")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE if os.path.exists(DEFAULT_BASELINE)
                    else "", help="baseline attempts log for the comparison column")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files, logs = find(a.logdirs, a.match)
    rows, problems = checks(files, logs)
    md = ["# Checks", "", "Files found (in the order used):", ""]
    for k in KINDS:
        md.append(f"- **{k}**: " + (", ".join(f"`{p}`" for p in files[k]) or "**MISSING**"))
    md += ["", "| level | runs in attempts | level starts in console logs | verdicts | v7 verdicts "
               "| usage matched | problems |", "|---|---|---|---|---|---|---|"]
    print("\n=========== checks ===========")
    for lv, n, st, v, nv, cov, flags in rows:
        md.append(f"| {lv} | {n} | {st} | {v} | {nv} | {cov} | "
                  + ("**" + "; ".join(flags) + "**" if flags else "ok") + " |")
        print(f"  level {lv}: runs {n}, console starts {st}, verdicts {v}, v7 verdicts {nv}, "
              f"usage {cov}  " + (f"{RED}{'; '.join(flags)}{OFF}" if flags else f"{GREEN}ok{OFF}"))
    for k in KINDS:
        if not files[k]:
            print(f"  {RED}no {k} files found{OFF}")
    for p in problems:
        md.append(f"- **{p}**")
        print(f"  {RED}{p}{OFF}")

    notes = []
    py = sys.executable
    if files["attempts"]:
        cmd = [py, os.path.join(HERE, "summarize.py"), *files["attempts"],
               "-o", os.path.join(a.out, "summary")]
        if files["verdicts"]:
            cmd += ["--verdicts", *files["verdicts"]]
        if files["nki_verdicts"]:
            cmd += ["--nki-verdicts", *files["nki_verdicts"]]
        if files["usage"]:
            cmd += ["--usage", *files["usage"]]
        if a.baseline:
            cmd += ["--baseline", a.baseline]
        run(cmd, notes)
        cmd = [py, os.path.join(HERE, "taxonomy.py"), *files["attempts"],
               "-o", os.path.join(a.out, "taxonomy")]
        if files["verdicts"]:
            cmd += ["--verdicts", *files["verdicts"]]
        run(cmd, notes)
        cmd = [py, os.path.join(HERE, "token_budget.py"), *files["attempts"],
               "-o", os.path.join(a.out, "token_budget")]
        if files["usage"]:
            cmd += ["--usage", *files["usage"]]
        run(cmd, notes)
    else:
        notes.append("no attempts files: nothing to summarize")
    md += ["", "## Steps that did not complete", ""] + ([f"- {n}" for n in notes] or ["none"])
    with open(os.path.join(a.out, "checks.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nchecks -> {os.path.join(a.out, 'checks.md')}")
    for n in notes:
        print(f"  {RED}{n}{OFF}")
    s = os.path.join(a.out, "summary.md")
    if os.path.exists(s):
        print("\n" + open(s).read())


if __name__ == "__main__":
    main()
