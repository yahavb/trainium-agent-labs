"""Failure taxonomy (EXP-9): every failed attempt gets one mode, and "wrong output" is split further.

Both levels are read from the checker's result in a committed run file, never from a model:

  1. The mode, from report.failure_mode(), so these counts use the same labels as runs/summary.md.
  2. For the two "wrong" modes, why the attempt is still wrong. The first matching reason wins:
       copied              a repair that returned the code it was asked to fix (same rule as report.copies)
       output X or Z       the simulator saw X or Z in at least one bit of the first wrong output
                           (not assigned on every path, not initialised, or driven twice)
       reset behaviour     the testbench printed a reset hint
       comb, counterexample    yosys found an input where the design is wrong
       comb, no counterexample formal check unsupported or not run
       seq, wrong from start   first mismatch in clock cycle 0 or 1 (start or reset state)
       seq, wrong later        first mismatch after cycle 1 (the logic that updates state)

    python taxonomy.py runs/dev/C-1.jsonl runs/dev/C-s3rewrite-full-1.jsonl
    python taxonomy.py --dir runs                  # every *.jsonl in the folder
    python taxonomy.py --dir runs --out tax.md     # write the Markdown to a file
    python taxonomy.py runs/dev/C-1.jsonl --show Prob050_kmap1:1   # read one attempt by hand
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report  # noqa: E402  (failure_mode keeps both levels on report.py's labels)

SHORT = {  # second-level reason -> short code for the per-problem trail
    "copied": "copy",
    "output X or Z": "X",
    "reset behaviour": "reset",
    "comb, counterexample": "cex",
    "comb, no counterexample": "comb?",
    "seq, wrong from start": "seq@start",
    "seq, wrong later": "seq@later",
}


def load(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def copied_ids(recs: list) -> set:
    """ids of repairs that returned the code they were asked to fix. Mirrors report.copies()."""
    out, by_lineage = set(), defaultdict(list)
    for r in recs:
        if r.get("attempt", 0) > 0:
            by_lineage[(r["problem"], r.get("seat"))].append(r)
    for rs in by_lineage.values():
        rs.sort(key=lambda r: r["attempt"])
        for prev, r in zip(rs, rs[1:]):
            if "copied" in r:
                hit = bool(r["copied"])
            else:
                hit = (r.get("strategy") == "S3" and bool(r.get("code_sha1"))
                       and r["code_sha1"] == prev.get("code_sha1"))
            if hit:
                out.add(id(r))
    return out


def reason(r: dict, mode: str, copied: bool) -> str | None:
    """Second level, only for the two 'wrong' modes."""
    if not mode.startswith("wrong"):
        return None
    if copied:
        return "copied"
    fm = r.get("first_mismatch") or {}
    if any(ch in str(fm.get("dut", "")).lower() for ch in "xz"):
        return "output X or Z"
    if mode == "wrong: reset behaviour":
        return "reset behaviour"
    if r.get("kind") == "comb":
        return "comb, counterexample" if r.get("counterexample") else "comb, no counterexample"
    cycle = fm.get("cycle")
    return "seq, wrong from start" if cycle is not None and cycle <= 1 else "seq, wrong later"


def short(r: dict, mode: str | None, why: str | None, copied: bool) -> str:
    if r.get("passed"):
        return "PASS"
    if why:
        return SHORT[why]
    if not mode:
        return "-"
    if mode.startswith("compile: R"):
        s = mode.split()[1]                        # R1, R5, ...
    else:
        s = {"compile: syntax error": "syntax", "compile: syntax error (truncated)": "syntax(trunc)",
             "no code in reply": "no code", "no code: truncated at max_tokens": "truncated",
             "model request failed": "request failed"}.get(mode, mode)
    return f"{s} (copy)" if copied else s


def section(path: str) -> list:
    recs = load(path)
    copies = copied_ids(recs)
    modes, copies_by_mode, reasons = Counter(), Counter(), Counter()
    trail = defaultdict(list)
    for r in sorted(recs, key=lambda r: (r["problem"], r.get("seat") or "", r.get("attempt", 0))):
        mode = report.failure_mode(r)
        why = reason(r, mode or "", id(r) in copies) if mode else None
        if mode:
            modes[mode] += 1
            copies_by_mode[mode] += id(r) in copies
        if why:
            reasons[why] += 1
        if r.get("attempt", 0) > 0:
            trail[r["problem"]].append((r, short(r, mode, why, id(r) in copies)))

    failed = sum(modes.values())
    wrong = sum(n for m, n in modes.items() if m.startswith("wrong"))
    rel = os.path.relpath(path, HERE)
    out = [f"### `{rel}`", "",
           f"{len(recs)} attempts, {failed} failed. Every failed attempt has exactly one mode.", "",
           "| Mode | Failed attempts | of which copied repairs |", "|---|---|---|"]
    for m, n in modes.most_common():
        out.append(f"| {m} | {n} | {copies_by_mode[m]} |")
    out += ["", f"**Inside the {wrong} \"wrong\" attempts:**", "", "| Why it is still wrong | Attempts |", "|---|---|"]
    for k in SHORT:
        if reasons[k]:
            out.append(f"| {k} | {reasons[k]} |")
    out += ["", "**Per problem, attempt by attempt** (`copy` = repair returned the same code, `X` = X or Z "
            "in simulation, `cex` = yosys counterexample, `seq@start`/`seq@later` = first wrong cycle, "
            "`R1`... = translator rule, `(copy)` = the same compile error from unchanged code):", ""]
    for prob in sorted(trail):
        rs = trail[prob]
        last = rs[-1][0]
        end = last.get("claim") or "?"
        stop = f", {last['stop_reason']}" if last.get("stop_reason") else ""
        out.append(f"- `{prob}` ({rs[0][0].get('kind')}): " + " → ".join(s for _, s in rs) + f"  **{end}{stop}**")
    return out + [""]


def show(files: list, which: str) -> None:
    """Print one attempt for reading by hand: the checker's facts, the feedback it produced, the code."""
    prob, _, att = which.partition(":")
    for p in files:
        for r in load(p):
            if r["problem"] == prob and str(r.get("attempt")) == att:
                mode = report.failure_mode(r)
                print(f"== {os.path.relpath(p, HERE)}  {prob}  attempt {att}  seat {r.get('seat')}  "
                      f"strategy {r.get('strategy')}")
                print(f"mode:            {mode or ('PASS' if r.get('passed') else '-')}")
                for k in ("layer_reached", "score", "mismatches", "samples", "compile_error", "l0_error",
                          "first_mismatch", "counterexample", "tb_hints", "finish_reason", "stop_reason"):
                    if r.get(k) not in (None, "", [], {}):
                        print(f"{k + ':':16} {r[k]}")
                print(f"feedback sent:   {r.get('feedback_sent') or '(none: last attempt or passed)'}")
                print("code:")
                print(r.get("code") or "(no code)")
                print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="run files (.jsonl)")
    ap.add_argument("--dir", help="use every *.jsonl in this folder")
    ap.add_argument("--out", help="write the Markdown here instead of printing it")
    ap.add_argument("--show", metavar="PROBLEM:ATTEMPT", help="print one attempt in full, for reading by hand")
    a = ap.parse_args()
    files = list(a.files)
    if a.dir:
        files += sorted(p for p in glob.glob(os.path.join(a.dir, "*.jsonl")))
    if not files:
        ap.error("give run files or --dir")
    if a.show:
        show(files, a.show)
        return
    lines = []
    for p in files:
        lines += section(p)
    lines.append(f"_Generated by `taxonomy.py` from {len(files)} run file(s). Counts only; no model was used._")
    text = "\n".join(lines) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print(f"wrote {a.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
