"""Tables, the paired comparison, the failure taxonomy and the chart, from committed run files only.

DESIGN.md 6.8 and section 8. Every number here cites the run file it came from; nothing is averaged
without every rep shown beside it.

    python report.py                 # runs/*.jsonl      -> runs/summary.md, runs/chart.svg
    python report.py --dir runs/dev  # the dev runs, kept apart from the held-out ones
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict

import translator

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ORDER = ["A", "R", "Q", "B", "C", "D", "E"]
RUN_LABEL = {"A": "A one-shot", "R": "R retry, no feedback", "Q": "Q retry, rotating prompts", "B": "B raw feedback",
             "C": "C located feedback", "D": "D 3 seats, breadth", "E": "E gpt-oss-20b"}
PAIRS = [("B", "C"), ("R", "C"), ("A", "C"), ("C", "D"), ("R", "B"),
         ("Q", "C"), ("Q", "B"), ("R", "Q"), ("Q", "D")]   # Q: controls/prompt_rotation.py


def load_runs(directory: str) -> dict:
    """{(run, rep): [records]} for every <RUN>-<N>.jsonl in the directory."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        m = re.match(r"^([A-Z])-(\d+)\.jsonl$", os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        runs[(m.group(1), int(m.group(2)))] = recs
    return runs


def finals(recs: list) -> dict:
    """{problem: final record} -- the record carrying the problem's claim."""
    return {r["problem"]: r for r in recs if r.get("claim")}


def first_pass_attempt(recs: list) -> dict:
    out = {}
    for r in recs:
        if r.get("passed") and r["problem"] not in out:
            out[r["problem"]] = r["attempt"]
    return out


def wall_seconds(recs: list) -> float:
    ts = [datetime.datetime.fromisoformat(r["ts"]) for r in recs if r.get("ts")]
    return (max(ts) - min(ts)).total_seconds() if len(ts) > 1 else 0.0


def audit_counts(directory: str, run: str, rep: int):
    path = os.path.join(directory, f"{run}-{rep}.audit.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["counts"]


def metrics(recs: list, audit) -> dict:
    fin = finals(recs)
    n = len(fin)
    solved = [p for p, r in fin.items() if r["claim"] == "PASS"]
    fpa = first_pass_attempt(recs)
    attempts = [r for r in recs if r.get("attempt", 0) > 0]
    tokens = sum(r.get("completion_tokens") or 0 for r in attempts)
    wall = wall_seconds(recs)
    last_layer = Counter()
    for p, r in fin.items():
        if r["claim"] != "PASS":
            last = max((x for x in recs if x["problem"] == p), key=lambda x: x.get("attempt", 0))
            last_layer[last.get("layer_reached")] += 1
    return {
        "n": n, "solved": len(solved),
        "unverified": sum(r["claim"] == "UNVERIFIED" for r in fin.values()),
        "comb": f"{sum(fin[p]['kind'] == 'comb' for p in solved)}/{sum(r['kind'] == 'comb' for r in fin.values())}",
        "seq": f"{sum(fin[p]['kind'] == 'seq' for p in solved)}/{sum(r['kind'] == 'seq' for r in fin.values())}",
        "median_attempts": statistics.median([fpa[p] for p in solved]) if solved else None,
        "tokens_per_solve": round(tokens / len(solved)) if solved else None,
        "seconds_per_solve": round(wall / len(solved)) if solved else None,
        "wall_s": round(wall),
        "truncation": sum(r.get("finish_reason") == "length" for r in attempts) / len(attempts) if attempts else 0,
        "attempts": len(attempts),
        "failure_layer": dict(sorted(last_layer.items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "escapes": audit["not_equivalent"] if audit else None,
        "copies": copies(recs),
        "code": sorted({r.get("code_version") or "?" for r in attempts}),
        "repairs": sum(r.get("strategy") == "S3" for r in attempts),
        "stops": dict(Counter(r.get("stop_reason") for r in fin.values())),
    }


def copies(recs: list) -> int:
    """Repairs that returned the code they were asked to fix. Runs before DESIGN 1.3.0 have no
    `copied` field, so it is recomputed from the code hashes of consecutive attempts."""
    n = 0
    by_problem = defaultdict(list)
    for r in recs:
        if r.get("attempt", 0) > 0:
            by_problem[(r["problem"], r.get("seat"))].append(r)
    for rs in by_problem.values():
        rs.sort(key=lambda r: r["attempt"])
        for prev, r in zip(rs, rs[1:]):
            if "copied" in r:
                n += bool(r["copied"])
            elif r.get("strategy") == "S3" and r.get("code_sha1") and r["code_sha1"] == prev.get("code_sha1"):
                n += 1
    return n


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs (McNemar's exact test)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(runs: dict, x: str, y: str) -> list:
    rows = []
    reps = sorted({rep for (run, rep) in runs if run == x} & {rep for (run, rep) in runs if run == y})
    for rep in reps:
        fx, fy = finals(runs[(x, rep)]), finals(runs[(y, rep)])
        common = sorted(set(fx) & set(fy))
        px = {p for p in common if fx[p]["claim"] == "PASS"}
        py = {p for p in common if fy[p]["claim"] == "PASS"}
        rows.append(dict(rep=rep, n=len(common), both=len(px & py), only_x=sorted(px - py),
                         only_y=sorted(py - px), neither=len(set(common) - px - py),
                         p=sign_test(len(py - px), len(px - py))))
    return rows


def failure_mode(r: dict) -> str | None:
    """One named mode per failed attempt (EXP-9). Derived from the checker's result, never a model."""
    if r.get("passed") or r.get("attempt", 0) == 0 or r.get("stop_reason") == "seat_down":
        return None
    if r.get("error"):
        return "model request failed"
    l0 = r.get("l0_error") or ""
    if l0 == "no_code":
        return "no code: truncated at max_tokens" if r.get("finish_reason") == "length" else "no code in reply"
    if l0 == "no_topmodule":
        return "no module TopModule"
    if l0.startswith("missing_port"):
        return "missing or renamed port"
    if r.get("layer_reached") == 0:
        if r.get("timed_out"):
            return "compile: timed out"
        err = r.get("compile_error") or ""
        rule = translator.matching_rule(err)
        if rule:
            return f"compile: {rule['id']} {rule['pattern']}"
        if not err.startswith("cand.sv"):
            return "compile: interface does not match testbench"
        if "syntax error" in err:
            return "compile: syntax error" + (" (truncated)" if r.get("finish_reason") == "length" else "")
        return "compile: other"
    if r.get("mismatches") is None:
        return "simulation never finished"
    if any("reset" in h.lower() for h in r.get("tb_hints") or []):
        return "wrong: reset behaviour"
    return f"wrong output ({r.get('kind')})"


def taxonomy(runs: dict) -> dict:
    return {key: Counter(m for m in map(failure_mode, recs) if m) for key, recs in runs.items()}


# ---------------------------------------------------------------- chart

def chart_svg(runs: dict, rows: dict, title: str) -> str:
    """Dot plot: one column per run, one dot per rep, a short tick at the mean of the reps.

    One series, so one hue and no legend; every dot carries a native tooltip. Colours are the dataviz
    reference palette (slot 1 blue, recessive grid and muted axis text), restepped for dark mode.
    """
    present = [r for r in RUN_ORDER if any(k[0] == r for k in runs)]
    W, H, L, R_, T, B = 640, 300, 56, 16, 44, 56
    pw, ph = W - L - R_, H - T - B
    col = pw / max(1, len(present))
    y = lambda v: T + ph * (1 - v)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" aria-label="{title}">',
             "<style>.bg{fill:#fcfcfb}.grid{stroke:#e1e0d9;stroke-width:1}.ax{fill:#898781;font-size:11px}"
             ".lab{fill:#52514e;font-size:12px}.ttl{fill:#0b0b0b;font-size:14px;font-weight:600}"
             ".dot{fill:#2a78d6;stroke:#fcfcfb;stroke-width:2}.mean{stroke:#52514e;stroke-width:2}"
             "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.grid{stroke:#2c2c2a}"
             ".lab{fill:#c3c2b7}.ttl{fill:#ffffff}.dot{fill:#3987e5;stroke:#1a1a19}.mean{stroke:#c3c2b7}}</style>",
             f'<rect class="bg" width="{W}" height="{H}"/>',
             f'<text class="ttl" x="{L}" y="24">{title}</text>']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R_}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="ax" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>')
    for i, run in enumerate(present):
        cx = L + col * (i + 0.5)
        reps = sorted(k for k in runs if k[0] == run)
        rates = [rows[k]["solved"] / rows[k]["n"] if rows[k]["n"] else 0 for k in reps]
        if len(rates) > 1:      # drawn first, so the dots sit on top of it
            mean = sum(rates) / len(rates)
            parts.append(f'<line class="mean" x1="{cx - 18:.1f}" x2="{cx + 18:.1f}" '
                         f'y1="{y(mean):.1f}" y2="{y(mean):.1f}"/>')
        for j, (key, rate) in enumerate(zip(reps, rates)):
            m = rows[key]
            dx = (j - (len(reps) - 1) / 2) * 12
            parts.append(f'<circle class="dot" cx="{cx + dx:.1f}" cy="{y(rate):.1f}" r="5">'
                         f'<title>{run} rep {key[1]}: {m["solved"]}/{m["n"]} ({rate:.0%})</title></circle>')
        name, _, desc = RUN_LABEL.get(run, run).partition(" ")
        parts.append(f'<text class="lab" x="{cx:.1f}" y="{H - B + 20}" text-anchor="middle" '
                     f'font-weight="600">{name}</text>')
        parts.append(f'<text class="ax" x="{cx:.1f}" y="{H - B + 35}" text-anchor="middle">{desc}</text>')
    parts.append(f'<text class="ax" x="{L}" y="{H - 6}">Each dot is one rep; the bar is their mean. '
                 f'Pass = 0 mismatches in the VerilogEval testbench.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- summary

def fmt(v, pct=False):
    if v is None:
        return "-"
    if pct:
        return f"{v:.0%}"
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def summary(directory: str) -> str:
    runs = load_runs(directory)
    if not runs:
        return f"No run files in {directory}."
    keys = sorted(runs, key=lambda k: (RUN_ORDER.index(k[0]) if k[0] in RUN_ORDER else 99, k[1]))
    rows = {k: metrics(runs[k], audit_counts(directory, *k)) for k in keys}
    shown = os.path.relpath(directory, HERE) if os.path.abspath(directory).startswith(HERE) else directory
    out = [f"# Results: `{shown}`", "",
           f"Generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC by `report.py` "
           f"from the run files named in each row. Pass = the VerilogEval testbench reports 0 mismatches "
           f"(simulator). Escapes = combinational passes that yosys proves NOT equivalent (formal).", "",
           "| Run | Rep | Solved | Pass rate | comb | seq | Median attempts to pass | Tokens / solve "
           "| Seconds / solve | Truncated | Unverified | Escapes | Copied repairs | Code | Run file |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        m = rows[k]
        out.append(f"| {k[0]} | {k[1]} | {m['solved']}/{m['n']} | {fmt(m['solved'] / m['n'] if m['n'] else 0, True)} "
                   f"| {m['comb']} | {m['seq']} | {fmt(m['median_attempts'])} | {fmt(m['tokens_per_solve'])} "
                   f"| {fmt(m['seconds_per_solve'])} | {m['truncation']:.0%} of {m['attempts']} | {m['unverified']} "
                   f"| {fmt(m['escapes'])} | {m['copies']} of {m['repairs']} "
                   f"| {', '.join(m['code'])}{' MIXED' if len(m['code']) > 1 else ''} | `{k[0]}-{k[1]}.jsonl` |")
    out += ["", "## Paired, problem by problem", "",
            "Same problems, same rep number. Only the discordant problems carry information; p is the "
            "two-sided exact sign test on them.", ""]
    for x, y in PAIRS:
        for row in paired(runs, x, y):
            out.append(f"- **{y} vs {x}, rep {row['rep']}** ({row['n']} problems): both {row['both']}, "
                       f"only {y} {len(row['only_y'])}, only {x} {len(row['only_x'])}, neither {row['neither']}; "
                       f"p = {row['p']:.3f}")
            if row["only_y"]:
                out.append(f"  - only {y}: {', '.join(row['only_y'])}")
            if row["only_x"]:
                out.append(f"  - only {x}: {', '.join(row['only_x'])}")
    out += ["", "## Where unsolved problems stopped", "",
            "Layer reached by the last attempt (-1 no usable code, 0 compile, 1 simulate, 2 wrong output, "
            "3 wrong output with a formal counterexample).", ""]
    for k in keys:
        out.append(f"- {k[0]}-{k[1]}: {rows[k]['failure_layer'] or 'none'}; stop reasons {rows[k]['stops']}")
    tax = taxonomy({k: runs[k] for k in keys})
    modes = sorted({m for c in tax.values() for m in c}, key=lambda m: -sum(c[m] for c in tax.values()))
    out += ["", "## Failure taxonomy (every failed attempt, one mode each)", "",
            "| Mode | " + " | ".join(f"{k[0]}-{k[1]}" for k in keys) + " |",
            "|---|" + "---|" * len(keys)]
    for m in modes:
        out.append(f"| {m} | " + " | ".join(str(tax[k].get(m, 0)) for k in keys) + " |")
    with open(os.path.join(directory, "chart.svg"), "w") as f:
        f.write(chart_svg(runs, rows, "Pass rate by run"))
    out += ["", "![Pass rate by run](chart.svg)", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()
    text = summary(a.dir)
    with open(os.path.join(a.dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
