#!/usr/bin/env python3
"""
analyze_log.py -- read an attempt log and say what the model actually did.

agent.py prints the best score per round. That is enough to see THAT a level is stuck and not
enough to see WHY. The log holds every kernel the model wrote and every message it was sent, so
the questions that decide what to fix can be answered by counting rather than by guessing:

  * which failures occur, how often, and which of them fed the next prompt
  * after the model was sent an instruction, did it resend the same code, change the code and
    hit the same failure, move to a different failure, or improve?
  * does the failing code contain the thing the instruction talks about?

    python analyze_log.py baseline.jsonl                    # every level
    python analyze_log.py baseline.jsonl --level 3 --show 2   # and print 2 example kernels per failure
    python analyze_log.py baseline.jsonl locate.jsonl       # one after another, then side by side

Needs nothing but Python. It reads logs; it does not grade anything.
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict

FULL = 1.0 - 1e-9

TAGS = [("WRONG SHAPE", "wrong output shape"),
        ("NON-FINITE OUTPUT", "NaN or Inf in the output"),
        ("OUTPUT IS", "output almost all zeros"),
        ("of the output is zero", "part of the output is zero"),
        ("NUMERICAL MISMATCH", "numerical mismatch"),
        ("THE KERNEL MODIFIED ITS INPUT", "wrote into its input"),
        ("CORRECT, BUT TOO MUCH HBM TRAFFIC", "correct but over the traffic bar"),
        ("CORRECT ON CPU BUT WRONG ON HARDWARE", "correct on CPU, hazard on hardware"),
        ("CANNOT SIMULATE", "cannot simulate (environment)")]


def classify(feedback):
    """A short name for a failure, the same for every attempt that failed the same way.

    Numbers are replaced by # so `size 32768` and `size 8192` are one failure, and the located
    line (--features locate) is ignored so logs with and without it classify identically.
    """
    fb = feedback or ""
    if fb.startswith("Correct on every shape"):
        return "SOLVED"
    if fb.startswith("No code came back"):
        return "no code came back"
    if fb.startswith("The code does not parse"):
        return "does not parse"
    if fb.startswith("There is no module named"):
        return "imports a module that does not exist"
    if fb.startswith("The file imports but"):
        return "could not be loaded"
    if fb.startswith("Rule violations"):
        first = fb.split("Fix exactly these:", 1)[-1].strip()
        first = re.sub(r"line \d+: ", "", first)
        return "rule: " + re.sub(r"\d+", "#", first)[:60]
    m = re.search(r"raised (\w+): (.*)", fb, re.S)
    if m:
        # The instruction is appended straight after the error text, so keep only the start,
        # cut at a word, which is the part the simulator wrote.
        text = re.sub(r"\d+", "#", " ".join(m.group(2).split()))
        if len(text) > 50:
            text = text[:50].rsplit(" ", 1)[0]
        return f"{m.group(1)}: {text}"
    for needle, name in TAGS:
        if needle in fb:
            return name
    return "other: " + " ".join(fb.split())[:50]


def shape_of_failure(feedback):
    """Which test shape the reported failure was on, and how many shapes passed before it."""
    m = re.match(r"(\d+) of (\d+) shapes passed\. On (.*?): ", feedback or "")
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def features_of(code):
    """Facts about a kernel that can be read off its text. Each is a yes/no or a short string."""
    f = {}
    f["calls reshape"] = bool(re.search(r"\breshape\s*\(", code))
    f["has a loop"] = bool(re.search(r"^\s*for\s", code, re.M))
    f["uses .ap()"] = ".ap(" in code
    f["tile shaped like a whole input"] = bool(
        re.search(r"ndarray\(\s*\w+\.shape\s*,[^)]*buffer\s*=\s*nl\.(sbuf|psum)", code, re.S))
    m = re.search(r"(\w+)\s*,\s*(\w+)\s*=\s*lhsT\.shape", code)
    f["unpacks lhsT.shape as"] = f"{m.group(1)}, {m.group(2)}" if m else "-"
    one_d = 0
    try:
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "ndarray" \
                    and node.args and isinstance(node.args[0], ast.Tuple) \
                    and len(node.args[0].elts) == 1:
                one_d += 1
    except SyntaxError:
        pass
    f["allocates a 1-D tile"] = one_d > 0
    return f


def load(path):
    """Group the flat log into runs -> rounds -> samples.

    Newer logs carry the run number (`rep`) and it is used. Older ones do not, so there a run of a
    level is recovered from the order: the log is written level by level, and a level's round
    counter restarting means the next run began. Order alone cannot split two runs that each
    ended on round 0 -- a level solved first time, twice running -- which is why `rep` exists.
    """
    episodes, current, last = [], None, (None, None, None)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            level, rnd, rep = r["level"], r["round"], r.get("rep")
            if current is None or level != last[0] or rnd < last[1] or rep != last[2]:
                current = dict(level=level, rounds=defaultdict(list))
                episodes.append(current)
            current["rounds"][rnd].append(r)
            last = (level, rnd, rep)
    for ep in episodes:
        ep["rounds"] = [ep["rounds"][k] for k in sorted(ep["rounds"])]
        ep["best"] = max(s["reward"] for rnd in ep["rounds"] for s in rnd)
    return episodes


def top_of(samples):
    """The sample the loop repaired next: highest reward, first on a tie -- as agent.py sorts."""
    return max(samples, key=lambda s: s["reward"])


def squash(code):
    return "".join((code or "").split())


def report_level(level, eps, show):
    print(f"\n{'=' * 78}\nLEVEL {level}   {len(eps)} run(s)\n{'=' * 78}")
    solved = sum(1 for e in eps if e["best"] >= FULL)
    print(f"  solved {solved}/{len(eps)}   best per run = {[round(e['best'], 2) for e in eps]}   "
          f"rounds used = {[len(e['rounds']) for e in eps]}")

    # ---- which failures, and which of them the loop acted on
    all_n, top_n, example = Counter(), Counter(), {}
    for e in eps:
        for rnd in e["rounds"]:
            for s in rnd:
                c = classify(s["feedback"])
                all_n[c] += 1
                example.setdefault(c, []).append(s)
            top_n[classify(top_of(rnd)["feedback"])] += 1
    total = sum(all_n.values())
    print(f"\n  Failures, over {total} attempts. 'fed next prompt' counts rounds where this was the "
          f"best\n  attempt, so its message is the one the model was sent.\n")
    print(f"  {'attempts':>8}  {'fed next prompt':>15}   failure")
    for c, n in all_n.most_common():
        print(f"  {n:>8}  {top_n.get(c, 0):>15}   {c}")

    # ---- the path each run took
    print("\n  What each run's best attempt hit, round by round:")
    order = {c: chr(ord('A') + i) for i, (c, _) in enumerate(all_n.most_common(26))}
    for c, letter in order.items():
        print(f"    {letter} = {c}")
    for i, e in enumerate(eps):
        path = " ".join(order.get(classify(top_of(r)["feedback"]), "?") for r in e["rounds"])
        print(f"    run {i + 1}: {path}   (best {e['best']:.2f})")

    # ---- after an instruction was sent, what came back?
    outcome = defaultdict(Counter)
    for e in eps:
        for prev, nxt in zip(e["rounds"], e["rounds"][1:]):
            sent = top_of(prev)
            c = classify(sent["feedback"])
            if c == "SOLVED":
                continue
            for s in nxt:
                if squash(s["code"]) == squash(sent["code"]):
                    outcome[c]["resent the same code"] += 1
                elif s["reward"] > sent["reward"] + 1e-9:
                    outcome[c]["scored higher"] += 1
                elif classify(s["feedback"]) == c:
                    outcome[c]["changed the code, same failure"] += 1
                else:
                    outcome[c]["changed the code, different failure"] += 1
    if outcome:
        cols = ["resent the same code", "changed the code, same failure",
                "changed the code, different failure", "scored higher"]
        print("\n  After the model was sent the message for a failure, what its next attempts did:\n")
        print("  " + "  ".join(f"{h.split(',')[0][:18]:>18}" for h in
                               ["same code", "same failure", "different failure", "scored higher"])
              + "   message sent for")
        for c, _ in all_n.most_common():
            if c in outcome:
                print("  " + "  ".join(f"{outcome[c][k]:>18}" for k in cols) + f"   {c}")

    # ---- does the failing code contain what the message talks about?
    print("\n  What the failing kernels contain, per failure (share of attempts):")
    for c, n in all_n.most_common(6):
        if c in ("SOLVED", "no code came back", "does not parse"):
            continue
        feats = [features_of(s["code"]) for s in example[c]]
        parts = []
        for k in ("calls reshape", "has a loop", "allocates a 1-D tile",
                  "tile shaped like a whole input", "uses .ap()"):
            share = sum(1 for f in feats if f[k]) / len(feats)
            parts.append(f"{k} {share:.0%}")
        unpack = Counter(f["unpacks lhsT.shape as"] for f in feats)
        if set(unpack) != {"-"}:
            parts.append("unpacks lhsT.shape as " + ", ".join(
                f"`{k}` x{v}" for k, v in unpack.most_common(3)))
        on = Counter(sf[2] for sf in (shape_of_failure(s["feedback"]) for s in example[c]) if sf)
        print(f"\n    {c}   ({n} attempts)")
        print("      " + "; ".join(parts))
        if on:
            print("      reported on shape: " + ", ".join(f"{k} x{v}" for k, v in on.most_common(4)))

    # ---- examples to read
    for c, _ in all_n.most_common(4 if show else 0):
        if c == "SOLVED":
            continue
        seen = set()
        for s in example[c]:
            if squash(s["code"]) in seen or not (s["code"] or "").strip():
                continue
            seen.add(squash(s["code"]))
            print(f"\n  ---- level {level}, example of: {c}  (round {s['round']}, "
                  f"reward {s['reward']:.2f}) ----")
            print("  message sent:\n    " + (s["feedback"] or "").replace("\n", "\n    "))
            print("  kernel:")
            for n, line in enumerate((s["code"] or "").splitlines(), 1):
                print(f"    {n:3d}  {line}")
            if len(seen) >= show:
                break


def summarise(episodes):
    by = defaultdict(list)
    for e in episodes:
        by[e["level"]].append(e)
    return by



def report_diversity(episodes):
    """Show whether parallel samples produce distinct repair evidence."""
    print("\n  Sample diversity per round (distinct feedback / samples):")
    for run_no, episode in enumerate(episodes, 1):
        ratios = []
        for rnd in episode["rounds"]:
            if not rnd:
                continue
            unique = len({sample.get("feedback", "") for sample in rnd})
            ratios.append((unique, len(rnd)))
        if not ratios:
            continue
        path = " -> ".join(f"{unique}/{count}" for unique, count in ratios)
        mean = sum(unique / count for unique, count in ratios) / len(ratios)
        print(f"    run {run_no}: {path}  (mean {mean:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="one or more attempt logs (.jsonl)")
    ap.add_argument("--level", type=int)
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="print N distinct example kernels for each of the commonest failures")
    ap.add_argument("--summary", action="store_true", help="only the side-by-side table")
    ap.add_argument("--diversity", action="store_true",
                    help="report distinct feedback outcomes per sampling round")
    a = ap.parse_args()

    loaded = {}
    for path in a.logs:
        try:
            loaded[path] = summarise(load(path))
        except (OSError, json.JSONDecodeError, KeyError) as e:
            sys.exit(f"cannot read {path}: {type(e).__name__}: {e}")

    if not a.summary:
        for path, by in loaded.items():
            print(f"\n{'#' * 78}\n# {path}\n{'#' * 78}")
            for level in sorted(by):
                if a.level in (None, level):
                    report_level(level, by[level], a.show)
                if a.diversity:
                    report_diversity(by[level])

    print(f"\n{'=' * 78}\nSIDE BY SIDE   solved / runs, then the best score of every run\n{'=' * 78}")
    levels = sorted({lv for by in loaded.values() for lv in by})
    for level in levels:
        if a.level not in (None, level):
            continue
        print(f"\n  level {level}")
        for path, by in loaded.items():
            eps = by.get(level, [])
            if not eps:
                continue
            best = [round(e["best"], 2) for e in eps]
            print(f"    {sum(1 for b in best if b >= FULL)}/{len(best)}  mean {sum(best) / len(best):.2f}"
                  f"  all={best}   {path}")
    print("\n  Report the rate and the spread. One run is not a result.")


if __name__ == "__main__":
    main()
