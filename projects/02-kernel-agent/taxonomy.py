#!/usr/bin/env python3
"""
taxonomy.py — turn the attempt log into the failure taxonomy and the token report.

Reads attempts.jsonl (what agent.py appends to) and prints, as markdown you can paste into the
write-up:

    1. every attempt classified into a named failure mode, counted per level
    2. how the failure MOVED from one round to the next (fixed one thing, broke another)
    3. where the prompt's characters went, per kind of prompt
    4. how often the checker could quote the failing line

    python taxonomy.py                         # attempts.jsonl in this folder
    python taxonomy.py --log attempts.jsonl --session 20261010-104500
    python taxonomy.py --examples              # one real checker message per failure mode
    python taxonomy.py > TAXONOMY.md

Needs nothing but the standard library, so it runs on a laptop with no chip and no model.
"""

import argparse
import json
import re
from collections import Counter, defaultdict

# Order matters: the first pattern that matches names the failure. Each entry is
# (name, what it means in one line, regex over the checker's feedback).
MODES = [
    ("solved", "correct on every shape", r"^Correct on every shape"),
    ("no code", "nothing usable came back (empty or all reasoning)", r"^No code came back"),
    ("does not parse", "syntax error, usually a truncated answer", r"does not parse"),
    ("wrong entry point", "the function is missing or misnamed", r"no function named"),
    ("missing @nki.jit", "entry point is not decorated", r"not decorated with `@nki\.jit`"),
    ("framework shortcut", "called numpy/torch to do the whole job, or used @ / .T",
     r"hands the whole operation|`@` matmul operator|transposes the\s+input on the host"),
    ("tile too big (static)", "a literal partition dimension over 128 in the source",
     r"exceeds the maximum of \d+\. Tile it"),
    ("invented module", "imported a module that does not exist", r"There is no module named"),
    ("invented API name", "called an nl/nisa function that does not exist",
     r"has no attribute"),
    ("called a memory region", "wrote nl.sbuf(...) instead of buffer=nl.sbuf",
     r"'MemoryRegion' object is not callable"),
    ("invented argument", "passed a keyword the function does not take",
     r"unexpected keyword argument|got multiple values for argument"),
    ("1-D tile", "allocated an SBUF/PSUM tile with one dimension",
     r"must have at least 2 dimensions"),
    ("reshaped instead of slicing", "tried to reshape a tensor into a tile",
     r"cannot reshape array"),
    ("no tiling: partition over 128", "one tile for the whole tensor",
     r"partition dimension \d+ exceeds maximum"),
    ("no tiling: K over 128", "contracted more than one tile's worth in one nc_matmul",
     r"Matmul contraction dimension \d+ exceeds"),
    ("tile/slice size mismatch", "source and destination hold different element counts",
     r"same number of elements|could not be broadcast"),
    ("read past the end", "indexed beyond the tensor, usually a padded final tile",
     r"Out-of-bound access"),
    ("wrong buffer", "operand or result in the wrong memory (sbuf / psum / hbm)",
     r"must be in \['"),
    ("Python operator on a tile", "used += or * on a tile instead of a nisa op",
     r"unsupported operand type"),
    ("modified its input", "wrote into the tensor it was given", r"MODIFIED ITS INPUT"),
    ("wrong output shape", "ran, but the output size is wrong", r"WRONG SHAPE"),
    ("NaN or Inf", "read an uninitialised tile", r"NON-FINITE OUTPUT"),
    ("result never written out", "ran, output is zeros", r"OUTPUT IS \d+% ZEROS"),
    ("partial coverage", "some tiles written, others not", r"% of the output is zero"),
    ("wrong numbers: ragged edge", "right in the interior, wrong in the last partial tile",
     r"NUMERICAL MISMATCH[\s\S]*ragged edge"),
    ("wrong numbers: core arithmetic", "most elements wrong",
     r"NUMERICAL MISMATCH[\s\S]*core arithmetic"),
    ("wrong numbers: accumulation", "wrong inside a full tile",
     r"NUMERICAL MISMATCH"),
    ("wrong on hardware only", "matches on CPU, the simulator warns the device would differ",
     r"WRONG ON HARDWARE"),
    ("correct but too much traffic", "passes numerics, fails the level's byte bar",
     r"TOO MUCH HBM TRAFFIC"),
    ("cannot simulate", "environment problem, not the model's", r"CANNOT SIMULATE"),
    ("could not load", "the file raised while being imported", r"could not be loaded"),
]
COMPILED = [(name, why, re.compile(pat)) for name, why, pat in MODES]
OTHER = re.compile(r"raised (\w+)")


def classify(rec):
    fb = rec.get("feedback") or ""
    if rec.get("reward", 0) >= 1.0 - 1e-9:
        return "solved"
    for name, _, pat in COMPILED:
        if pat.search(fb):
            return name
    m = OTHER.search(fb)
    return f"other: {m.group(1)}" if m else "unclassified"


def load(path, session=None, level=None):
    rows, last_round, segment = [], {}, Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a half-written last line while a run is live
            if session and r.get("session") != session:
                continue
            if level is not None and r.get("level") != level:
                continue
            # A round number that goes backwards for the same level means a new run started. Logs
            # written before session ids existed can only be split this way.
            key = (r.get("session"), r.get("run", 0), r.get("level"))
            if r.get("round", 0) < last_round.get(key, -1):
                segment[key] += 1
            last_round[key] = r.get("round", 0)
            r["segment"] = segment[key]
            r["mode"] = classify(r)
            rows.append(r)
    return rows


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--session", help="only this session id (printed at the start of a run)")
    ap.add_argument("--level", type=int)
    ap.add_argument("--examples", action="store_true",
                    help="also print one real checker message per failure mode")
    a = ap.parse_args()

    rows = load(a.log, a.session, a.level)
    if not rows:
        raise SystemExit(f"no attempts found in {a.log} for that filter")
    levels = sorted({r["level"] for r in rows})
    why = {name: w for name, w, _ in MODES}

    # ---- 1. the taxonomy
    counts = defaultdict(Counter)
    for r in rows:
        counts[r["mode"]][r["level"]] += 1
    order = sorted(counts, key=lambda m: -sum(counts[m].values()))
    sessions = sorted({r.get("session") or "(before session ids)" for r in rows})
    print(f"# Failure taxonomy\n\n{len(rows)} attempts, levels {levels}, "
          f"{len(sessions)} session(s): {', '.join(sessions)}\n")
    body = []
    for m in order:
        total = sum(counts[m].values())
        body.append([m, why.get(m, "an exception no rule above names yet")]
                    + [counts[m].get(lv, "") for lv in levels]
                    + [total, f"{100 * total / len(rows):.0f}%"])
    print(table(["failure mode", "what it means"] + [f"L{lv}" for lv in levels]
                + ["total", "share"], body))

    # ---- 2. how the failure moved between rounds
    # Group by (session, run, level); within a round take the best-scoring sample, which is the
    # one the agent repaired from.
    groups = defaultdict(dict)
    for r in rows:
        key = (r.get("session"), r.get("run", 0), r["level"], r["segment"])
        cur = groups[key].get(r["round"])
        if cur is None or r["reward"] > cur["reward"]:
            groups[key][r["round"]] = r
    moves, drops = Counter(), 0
    for rounds in groups.values():
        seq = [rounds[k] for k in sorted(rounds)]
        for prev, nxt in zip(seq, seq[1:]):
            if nxt["round"] != prev["round"] + 1:
                continue
            moves[(prev["mode"], nxt["mode"])] += 1
            if nxt["reward"] < prev["reward"] - 1e-9:
                drops += 1
    if moves:
        total = sum(moves.values())
        stuck = sum(n for (x, y), n in moves.items() if x == y)
        print(f"\n## How the failure moved, round to round\n\n{total} repair steps: "
              f"{stuck} repeated the same failure ({100 * stuck / total:.0f}%), "
              f"{total - stuck} changed it, and {drops} made the score go DOWN.\n")
        body = [[x, y, n] for (x, y), n in moves.most_common(15) if x != y]
        if body:
            print(table(["after this failure", "the next round hit", "times"], body))

    # ---- 3. where the prompt went
    shaped = [r for r in rows if r.get("prompt_shape") and r.get("sample", 0) == 0]
    if shaped:
        parts = ["reference", "docs", "code", "feedback", "ledger", "instructions"]
        by_kind = defaultdict(list)
        for r in shaped:
            by_kind[r["prompt_shape"]["kind"]].append(r)
        print("\n## Where the prompt went (characters, mean per prompt)\n")
        body = []
        for kind, rs in sorted(by_kind.items()):
            n = len(rs)
            toks = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens")]
            out = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
            body.append([kind, n]
                        + [round(sum(r["prompt_shape"].get(p, 0) for r in rs) / n) for p in parts]
                        + [round(sum(toks) / len(toks)) if toks else "n/a",
                           round(sum(out) / len(out)) if out else "n/a"])
        print(table(["prompt kind", "prompts"] + parts + ["prompt tokens", "answer tokens"], body))
        cut = sum(1 for r in rows if r.get("finish") == "length")
        if cut:
            print(f"\n{cut} of {len(rows)} answers were cut off at the token limit "
                  f"(finish_reason=length).")

    # ---- 4. did the checker manage to quote the line
    raised = [r for r in rows if "raised " in (r.get("feedback") or "") and "located" in r]
    if raised:
        got = sum(1 for r in raised if r["located"])
        print(f"\n## Line-located feedback\n\nThe checker quoted the failing line on {got} of "
              f"{len(raised)} attempts that raised ({100 * got / len(raised):.0f}%).")

    if a.examples:
        print("\n## One real checker message per failure mode\n")
        shown = set()
        for r in rows:
            if r["mode"] in shown or r["mode"] == "solved":
                continue
            shown.add(r["mode"])
            print(f"**{r['mode']}** (level {r['level']}, round {r['round']}, "
                  f"reward {r['reward']:.2f})\n\n> {(r.get('feedback') or '').strip()[:500]}\n")


if __name__ == "__main__":
    main()
