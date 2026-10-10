#!/usr/bin/env python3
"""The failure taxonomy: every wrong kernel the agent wrote, grouped into named failure modes.

    python scripts/taxonomy.py runs/seat-*/latest/projects/02-kernel-agent/attempts.jsonl \
        --verdicts runs/seat-*/latest/projects/02-kernel-agent/verdicts.jsonl -o analysis/taxonomy

Writes <out>.md (the tables, ready for the README) and <out>.csv (mode x level counts).

Every attempt is classified by the checker's message, first match wins. The patterns match the
ERROR text the checker or the simulator produced, not the advice appended after it, so the same
mode is found whichever feedback version (agent.py, feedback_v2, feedback_v3) wrote the log.
Beyond counts it answers the questions a count cannot:
  stuck       -- once the agent hits a mode, how often the next round hits it again
  transitions -- what it broke while fixing what it was shown ("fixed A, now B")
  walls       -- the mode each unsolved run ended on
  held-out    -- kernels that passed the loop's shapes and failed shapes they never saw
"""
import argparse
import collections
import csv
import json
import os
import re

# (mode, family, description, pattern). First match wins, so specific before general.
MODES = [
    ("solved", "-", "correct on every loop shape", r"^Correct on every shape"),
    # the agent does not know the API
    ("invented_name", "API", "calls an NKI function or attribute that does not exist",
     r"has no attribute|object is not callable"),
    ("invented_module", "API", "imports a module that does not exist", r"no module named"),
    ("wrong_signature", "API", "right function, wrong arguments",
     r"unexpected keyword argument|missing \d+ required|takes \d+ positional|"
     r"got multiple values for argument"),
    # the agent does not know the memory model
    ("wrong_buffer", "memory model", "a tile in the wrong memory (SBUF / PSUM / HBM)",
     r"must be in \["),
    ("python_op_on_tile", "memory model", "Python arithmetic (+=, *) on a tile",
     r"unsupported operand type\(s\).*NkiTensor"),
    # the agent misreads the tiling rules
    ("partition_over_128", "tiling rules", "a tile or contraction larger than 128 partitions",
     r"partition dimension \d+ exceeds|exceeds maximum 128|exceeds the maximum of 128|"
     r"contraction dimension \d+ exceeds pmax"),
    ("tile_1d", "tiling rules", "a 1-D tile (every SBUF/PSUM tile needs 2 dims)",
     r"at least 2 dimensions"),
    ("illegal_allocation", "tiling rules",
     "runs in the simulator, but allocates a tile the chip cannot hold", r"ILLEGAL ON HARDWARE"),
    # the agent gets the index arithmetic wrong
    ("copy_size_mismatch", "index arithmetic", "copies between tiles of different sizes",
     r"same number of elements"),
    ("out_of_bounds", "index arithmetic", "indexes past the end of a tensor",
     r"Out-of-bound access"),
    ("reshape", "index arithmetic", "reshapes instead of slicing", r"cannot reshape"),
    ("broadcast", "index arithmetic", "assigns a value of the wrong shape",
     r"could not be broadcast|shape mismatch|operands could not"),
    ("wrong_output_shape", "index arithmetic", "returns the wrong output shape", r"WRONG SHAPE"),
    ("bad_access_pattern", "index arithmetic", "a strided .ap() view that does not fit the tile",
     r"ap\(\) pattern|invalid partition stride"),
    # it runs, and the numbers are wrong
    ("partial_output", "silent wrong numbers", "some or all output tiles never written",
     r"OUTPUT IS \d+% ZEROS|of the output is zero"),
    ("numeric_mismatch", "silent wrong numbers", "runs, numbers wrong", r"NUMERICAL MISMATCH"),
    ("non_finite", "silent wrong numbers", "NaN or Inf in the output", r"NON-FINITE OUTPUT"),
    ("modified_input", "silent wrong numbers", "writes into its input", r"MODIFIED ITS INPUT"),
    ("hardware_hazard", "silent wrong numbers", "right on CPU, wrong on the device",
     r"WRONG ON HARDWARE"),
    ("wrong_dtype", "silent wrong numbers", "returns a different dtype than the reference",
     r"WRONG DTYPE"),
    ("too_much_traffic", "silent wrong numbers", "correct but over the level's HBM byte bar",
     r"TOO MUCH HBM TRAFFIC"),
    # the answer never became a kernel
    ("rule_violation", "rules / format", "banned call, missing @nki.jit or wrong entry name",
     r"Rule violations|RULE VIOLATIONS|rule violation"),
    ("no_code", "rules / format", "no code in the reply", r"No code came back"),
    ("parse_error", "rules / format", "the code does not parse", r"does not parse"),
    ("load_error", "rules / format", "parses but the kernel cannot be loaded",
     r"could not be loaded"),
]
FAMILY = {m: f for m, f, _, _ in MODES}
DESCRIBE = {m: d for m, _, d, _ in MODES}


def classify(text):
    for mode, _, _, pat in MODES:
        if re.search(pat, text or ""):
            return mode
    return "other"


def error_line(feedback):
    """The checker's error without the advice after it, for the example column."""
    m = re.search(r"raised (\w+: [^\n]*)", feedback or "")
    s = m.group(1) if m else (feedback or "")
    # enrich() appends its advice right after the error; it starts with one of these words
    s = re.split(r"\s(?=(?:The|A|Do|You|Every|Remove|Allocate|Pick|Use|Note|If|Start|Fix|"
                 r"There|Change|This)\b)", s)[0]
    return re.sub(r"\s+", " ", s)[:110]


def _samples_per_round(rows):
    """The usual number of attempts in one (level, round): --samples. Each round logs exactly that
    many, so it marks where a run ends when two runs meet at the same round number."""
    sizes, prev = collections.Counter(), None
    n = 0
    for r in rows:
        key = (r["level"], r["round"])
        if key != prev and prev is not None:
            sizes[n] += 1
            n = 0
        prev, n = key, n + 1
    if n:
        sizes[n] += 1
    return sizes.most_common(1)[0][0] if sizes else 1


def load_attempts(paths):
    """Attempts grouped into episodes: one (file, run, level) is one attempt at one level.

    New logs carry a run field (offset when a file holds several invocations). Older ones are split by order: a new run starts when the level goes
    down (--all starts over), when the round goes down (--level N --repeat starts over), or when a
    round already holds --samples attempts (two runs solved in round 0 back to back, which happens
    once a log is split by level)."""
    episodes = collections.OrderedDict()
    for fi, path in enumerate(paths):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        per_round = _samples_per_round(rows)
        run, prev, filled = 0, None, 0
        offset, last_logged = 0, None
        for r in rows:
            key = (r["level"], r["round"])
            if "run" in r:
                # a second invocation appended to the same file numbers its runs from 0 again
                if last_logged is not None and r["run"] < last_logged:
                    offset = run + 1
                last_logged = r["run"]
                run = offset + r["run"]
            elif prev is not None and (key[0] < prev[0] or
                                       (key[0] == prev[0] and key[1] < prev[1]) or
                                       (key == prev and filled >= per_round)):
                run += 1
                filled = 0
            filled = filled + 1 if key == prev else 1
            prev = key
            episodes.setdefault((fi, run, r["level"]), []).append(r)
    return episodes


def top_per_round(attempts):
    """The attempt the agent carried forward each round: the highest reward, first on ties."""
    rounds = collections.OrderedDict()
    for r in attempts:
        rounds.setdefault(r["round"], []).append(r)
    return [max(rs, key=lambda r: r["reward"]) for rs in rounds.values()]


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempts", nargs="+", help="attempts.jsonl files")
    ap.add_argument("--verdicts", nargs="*", default=[], help="verdicts.jsonl files")
    ap.add_argument("-o", "--out", default="analysis/taxonomy")
    a = ap.parse_args()

    episodes = load_attempts(a.attempts)
    levels = sorted({lv for _, _, lv in episodes})
    counts = collections.Counter()            # (mode, level) -> attempts
    hit = collections.Counter()               # (mode, level) -> episodes that hit it
    example = {}
    stuck, seen_next = collections.Counter(), collections.Counter()
    transitions = collections.Counter()
    walls = collections.Counter()
    solved = collections.Counter()
    truncated = 0
    for (fi, run, lv), atts in episodes.items():
        modes_here = set()
        for r in atts:
            m = classify(r.get("feedback", ""))
            counts[(m, lv)] += 1
            modes_here.add(m)
            example.setdefault(m, error_line(r.get("feedback", "")))
            truncated += r.get("finish") == "length"
        for m in modes_here:
            hit[(m, lv)] += 1
        tops = [classify(r.get("feedback", "")) for r in top_per_round(atts)]
        for x, y in zip(tops, tops[1:]):
            seen_next[x] += 1
            if x == y:
                stuck[x] += 1
            elif y != "solved":
                transitions[(x, y)] += 1
        if tops and tops[-1] == "solved":
            solved[lv] += 1
        elif tops:
            walls[(lv, tops[-1])] += 1
    n_ep = collections.Counter(lv for _, _, lv in episodes)
    total = sum(counts.values())
    failures = total - sum(counts[("solved", lv)] for lv in levels)

    modes = sorted({m for m, _ in counts if m != "solved"},
                   key=lambda m: -sum(counts[(m, lv)] for lv in levels))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["family", "mode", "description"] + [f"level {lv}" for lv in levels]
                   + ["total", "share of failures", "stuck rate"])
        for m in modes:
            n = [counts[(m, lv)] for lv in levels]
            w.writerow([FAMILY.get(m, "other"), m, DESCRIBE.get(m, "")] + n
                       + [sum(n), f"{sum(n) / max(failures, 1):.1%}",
                          f"{stuck[m] / seen_next[m]:.0%}" if seen_next[m] else ""])

    md = [f"# Failure taxonomy",
          "",
          f"{total} attempts in {len(episodes)} level-runs from {len(a.attempts)} log file(s); "
          f"{failures} failed. Each attempt is classified by the checker's error, first match "
          f"wins (`scripts/taxonomy.py`).",
          "",
          "## Solve rate per level",
          "",
          table([[lv, f"{solved[lv]}/{n_ep[lv]}"] for lv in levels], ["level", "solved runs"]),
          "",
          "## Failure modes",
          "",
          "*runs* = level-runs that hit the mode at least once. *stuck* = when the round's carried "
          "attempt had this mode, how often the next round's did too.",
          ""]
    rows = []
    for fam in dict.fromkeys(FAMILY.get(m, "other") for m in modes):
        for m in [m for m in modes if FAMILY.get(m, "other") == fam]:
            n = [counts[(m, lv)] for lv in levels]
            rows.append([fam, f"`{m}`", DESCRIBE.get(m, "")]
                        + [x or "" for x in n]
                        + [sum(n), f"{sum(n) / max(failures, 1):.0%}",
                           sum(hit[(m, lv)] for lv in levels),
                           f"{stuck[m] / seen_next[m]:.0%}" if seen_next[m] else "-",
                           example.get(m, "").replace("|", "\\|")])
    md.append(table(rows, ["family", "mode", "what it means"] + [f"L{lv}" for lv in levels]
                    + ["total", "% of failures", "runs", "stuck", "example"]))
    fam_tot = collections.Counter()
    for m in modes:
        fam_tot[FAMILY.get(m, "other")] += sum(counts[(m, lv)] for lv in levels)
    md += ["", "By family: " + ", ".join(f"**{f}** {n} ({n / max(failures, 1):.0%})"
                                          for f, n in fam_tot.most_common()), ""]

    md += ["## Fixed one thing, broke another", "",
           "Round-to-round changes in the carried attempt's failure mode (not counting a solve).",
           "", table([[f"`{x}`", f"`{y}`", n] for (x, y), n in transitions.most_common(12)],
                     ["was", "became", "times"]) if transitions else "(none)", ""]
    md += ["## Where unsolved runs ended", "",
           table([[lv, f"`{m}`", n] for (lv, m), n in sorted(walls.items(),
                                                              key=lambda kv: (kv[0][0], -kv[1]))],
                 ["level", "last failure", "runs"]) if walls else "(every run solved)", ""]
    if truncated:
        md += [f"{truncated} attempt(s) were cut off by the token budget (finish=length).", ""]

    verdicts = [json.loads(l) for p in a.verdicts for l in open(p) if l.strip()]
    if verdicts:
        status = collections.Counter(
            (v["level"], v.get("claim") or v.get("status", "").split(":")[0].split(" (")[0])
            for v in verdicts)
        held = collections.Counter()
        for v in verdicts:
            if v.get("solved"):
                for fcase in v.get("heldout_failures") or []:
                    held[(v["level"], fcase["kind"], classify(fcase["why"]))] += 1
        md += ["## Held-out check", "",
               "After the loop, each level's best kernel ran once on shapes and values it never saw "
               "(`nkibench.py --eval`). Confidence was stated before that check.", "",
               table([[lv, s, n] for (lv, s), n in sorted(status.items())],
                     ["level", "verdict", "runs"]), ""]
        if held:
            md += ["Solved on the loop's shapes, failed held-out (verdicts keep the first 8 failing "
                   "cases per kernel):", "",
                   table([[lv, k, f"`{m}`", n] for (lv, k, m), n in sorted(held.items())],
                         ["level", "value kind", "mode", "cases"]), ""]
        scored = [v for v in verdicts if v.get("heldout_total")]
        if scored:
            ok = [1.0 if v["heldout_passed"] == v["heldout_total"] else 0.0 for v in scored]
            brier = sum((v["confidence"] - y) ** 2 for v, y in zip(scored, ok)) / len(scored)
            over = sum(1 for v, y in zip(scored, ok) if v["confidence"] >= 0.5 and not y)
            md += [f"Calibration over {len(scored)} verdicts: Brier score {brier:.3f}; "
                   f"confident (>= 0.5) but wrong {over} time(s).", ""]

    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    other = sum(counts[("other", lv)] for lv in levels)
    print(f"{total} attempts, {len(episodes)} level-runs, {failures} failures in {len(modes)} modes"
          f" ({other} unclassified) -> {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
