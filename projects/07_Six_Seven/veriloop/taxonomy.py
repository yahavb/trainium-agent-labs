"""M3: the failure taxonomy -- every wrong design the model wrote, grouped into named failure modes, counted.

    python veriloop/taxonomy.py                    # reads results/*_attempts.jsonl, writes results/TAXONOMY.md

Only attempts from COMPLETED runs (run id present in a *_summary.csv) are counted; offline runs, the pilot
and aborted runs are left out. Each distinct failed design is re-graded by the checker and classified:

  no code        the reply had no Verilog in it
  compile: ...   the plain-words compile error (syntax, undeclared signal, wire assigned in always, ...)
  run: ...       the simulation never finished, or the design stopped it itself
  behaviour: ... it ran but was wrong. Classified by the FIRST wrong step:
                 - undefined output (x)          a register never reset / an output never assigned
                 - one cycle early / late        the output is right but shifted by one clock cycle
                 - changes too early / too late  nothing pressed, but a phase is too short / too long
                 - wrong when <condition>        the control inputs active at that step (ped pressed, a or b
                                                 negative, clear, reset, read and write together, ...)
This is a first pass for people to check, not the last word: read the examples it prints.
"""

import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
import checker  # noqa: E402

LEVEL_DIRS = {os.path.basename(d): d for root in (os.path.join(HERE, "levels"), os.path.join(HERE, "tests", "sim_fixtures"))
              for d in glob.glob(os.path.join(root, "*")) if os.path.isfile(os.path.join(d, "reference.py"))}


def completed_runs():
    ids = set()
    for p in glob.glob(os.path.join(RESULTS, "*_summary.csv")):
        with open(p) as f:
            ids |= {r["run_id"] for r in csv.DictReader(f) if r["offline"] != "True"}
    return ids


def compile_kind(messages):
    m = " ".join(messages)
    for pat, name in [("not Verilog", "words around the code"), ("missing `;`", "syntax (missing ;)"),
                      ("`begin`", "syntax (unclosed begin/end)"), ("Syntax error", "syntax (other)"),
                      ("never declared", "undeclared signal"), ("declared as a wire", "wire assigned in always"),
                      ("no port named", "wrong port name"), ("bit(s) wide", "wrong port width"),
                      ("There is no module named", "wrong module name"), ("is not defined", "undefined submodule"),
                      ("declared twice", "declared twice")]:
        if pat in m:
            return name
    return "other"


def condition(ref, inputs):
    """The control inputs active at the first wrong step, in plain words."""
    on = []
    for name, value in inputs.items():
        w = ref.INPUTS.get(name, 1)
        if w == 1 and value:
            on.append(name)
        elif w > 1 and name in ("a", "b") and value < 0:
            on.append(f"{name}<0")
    if "wr_en" in on and "rd_en" in on:
        on = [x for x in on if x not in ("wr_en", "rd_en")] + ["read+write together"]
    return ", ".join(on) if on else "no control input active"


def shifted(rows, signal, first, shift):
    """True if, from the first wrong step on, the output equals the expected value `shift` cycles earlier."""
    lo, hi, hits, n = first, min(len(rows), first + 40), 0, 0
    for i in range(lo, hi):
        j = i - shift
        if 0 <= j < len(rows):
            n += 1
            hits += rows[i][2].get(signal) == rows[j][1][signal]
    return n >= 5 and hits >= 0.9 * n


def classify(ref, sim):
    if not sim.compiled.ok:
        return "compile: " + compile_kind(sim.compiled.messages)
    if not sim.ran:
        return "run: " + ("never finished (endless loop)" if "did not finish" in sim.problem
                          else "stopped the simulation itself")
    first = min(m["step"] for m in sim.mismatches)
    firsts = [m for m in sim.mismatches if m["step"] == first]
    sig = firsts[0]["signal"]
    if any(m["got"] is None for m in firsts):
        return f"behaviour: `{sig}` undefined (x) -- never reset or never assigned"
    if ref.CLOCKED:
        if shifted(sim.rows, sig, first, 1):
            return f"behaviour: `{sig}` one cycle late"
        if shifted(sim.rows, sig, first, -1):
            return f"behaviour: `{sig}` one cycle early"
    cond = condition(ref, sim.rows[first][0])
    if ref.CLOCKED and cond == "no control input active" and first > 0:
        # Nothing was pressed: the design changed state at the wrong moment (or to the wrong state).
        exp_now, exp_before = sim.rows[first][1][sig], sim.rows[first - 1][1][sig]
        got_now, got_before = sim.rows[first][2].get(sig), sim.rows[first - 1][2].get(sig)
        if exp_now == exp_before and got_now != got_before:
            return f"behaviour: `{sig}` changes too early (a phase is too short)"
        if exp_now != exp_before and got_now == got_before:
            return f"behaviour: `{sig}` changes too late (a phase is too long)"
        return f"behaviour: `{sig}` goes to the wrong value"
    return f"behaviour: `{sig}` wrong when {cond}"


def main():
    runs = completed_runs()
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*_attempts.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            if r["run"] in runs and not r["solved"] and not r.get("offline"):
                rows.append(r)
    if not rows:
        sys.exit("No failed attempts from completed runs in results/ yet.")

    cache, counts, examples = {}, defaultdict(Counter), {}
    per_level_fail = Counter()
    for r in rows:
        lv = r["level"]
        per_level_fail[lv] += 1
        if not r["verilog"]:
            kind = "no code"
        else:
            key = (lv, r["verilog"])
            if key not in cache:
                ref = checker.load_level(LEVEL_DIRS[lv])
                cache[key] = classify(ref, checker.simulate(ref, r["verilog"]))
            kind = cache[key]
        counts[lv][kind] += 1
        examples.setdefault((lv, kind), r)

    out = ["# Failure taxonomy (generated by `veriloop/taxonomy.py` — first pass, check the examples)", "",
           f"Every failed attempt from completed real runs: **{len(rows)} attempts**, "
           f"**{len(cache)} distinct failed designs** re-graded by the checker. Counts are attempts.", ""]
    for lv in sorted(counts):
        out += [f"## {lv} — {per_level_fail[lv]} failed attempts", "", "| failure mode | attempts | share |",
                "|---|---|---|"]
        for kind, n in counts[lv].most_common():
            out.append(f"| {kind} | {n} | {100 * n / per_level_fail[lv]:.0f}% |")
        out.append("")
        for kind, _ in counts[lv].most_common(3):
            ex = examples[(lv, kind)]
            out += [f"<details><summary>Example: {kind} (run {ex['run']}, round {ex['round']})</summary>", "",
                    "```verilog", (ex["verilog"] or "(no code)").strip()[:1500], "```", "</details>", ""]
    text = "\n".join(out) + "\n"
    os.makedirs(RESULTS, exist_ok=True)
    open(os.path.join(RESULTS, "TAXONOMY.md"), "w").write(text)
    print("\n".join(l for l in out if not l.startswith(("<details", "```", "</details")) and "module" not in l)[:4000])
    print("\nwrote results/TAXONOMY.md")


if __name__ == "__main__":
    main()
