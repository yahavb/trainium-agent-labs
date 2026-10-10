#!/usr/bin/env python3
"""
taxonomy.py -- every failed attempt from every seat, sorted into a handful of named failure modes, with
counts per arm and one real example each. Owner: P3.

    python taxonomy.py                                   # every logs/seat-*/attempts*.jsonl (and sim/)
    python taxonomy.py more.jsonl other.jsonl            # plus these files
    python taxonomy.py --json taxonomy.json              # also write the table for the dashboard

Each record lands in exactly ONE mode: the first rule below that matches, on its verdict, its message and
the instruction it was given. Correct attempts are counted too, as their own modes, because "correct but
not faster" is part of the story. Anything no rule names goes to "other wrong" and is printed in full, so
the rules get better rather than the leftovers getting hidden.

LABEL: "sim" when the record came from the simulator-only fallback referee (source == "sim"), else
"chip" (P1's speedcheck, which also leaves source empty on a rejection). Never mix them in one claim.
"""

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHIPBOOST = os.path.dirname(HERE)

# (mode, test). First match wins; order matters: the most specific diagnosis first.
def _msg(rec):
    return " ".join(str(rec.get(k) or "") for k in ("referee_message", "instruction_given"))


def _has(pattern):
    rx = re.compile(pattern, re.I)
    return lambda rec: rec.get("verdict") in ("wrong", None) and bool(rx.search(_msg(rec)))


MODES = [
    ("faster (correct, beat the noise)", lambda r: r.get("verdict") == "faster"),
    ("no gain (correct, inside noise)", lambda r: r.get("verdict") == "no_gain"),
    ("slower (correct)", lambda r: r.get("verdict") == "slower"),
    ("correct, untimed (sim)", lambda r: r.get("verdict") is None),
    ("no usable code", lambda r: not (r.get("code") or "").strip()
        or bool(re.search(r"does not parse|No code came back", _msg(r)))),
    ("rule violation", lambda r: r.get("verdict") == "rules"),
    ("wrong only on held-out shapes", lambda r: r.get("verdict") == "heldout_fail"),
    ("invented API", _has(r"has no attribute|No module named|unexpected keyword argument|is not defined"
                         r"|object is not callable")),
    ("dropped part of K", _has(r"ONLY THE FIRST 128 ROWS OF K|ONLY THE LAST 128 ROWS OF K")),
    ("kept only one tile", _has(r"KEEPS ONLY ONE (rhs|lhsT) TILE")),
    ("shared accumulator across tiles", _has(r"RUNNING SUMS OF EACH OTHER")),
    ("tile too big / wrong size", _has(r"same number of elements|exceeds? (the )?maximum|exceed dimension size"
                                       r"|Out-of-bound|partition dimension \d+ exceeds")),
    ("wrong layout / swapped axes", _has(r"contraction dimension mismatch|WRONG SHAPE")),
    ("wrong memory buffer or dtype", _has(r"must be in \[|dtype must be")),
    ("precision loss", _has(r"PRECISION LOSS")),
    ("timed out / killed", _has(r"timed out|SIGXCPU|SIGKILL")),
    ("output never written (zeros/NaN)", _has(r"IS \d+% ZEROS|NON-FINITE OUTPUT")),
    ("correct in simulator, wrong on chip", lambda r: r.get("verdict") == "wrong" and r.get("sim_ok") is True),
    ("wrong numbers, cause not named", _has(r"NUMERICAL MISMATCH|mismatch")),
    ("other wrong", lambda r: True),
]
CORRECT = {m for m, _ in MODES[:4]}


def classify(rec):
    return next(m for m, test in MODES if test(rec))


def label(rec):
    return "sim" if rec.get("source") == "sim" else "chip"


def _short(p):
    try:
        return os.path.relpath(p, CHIPBOOST)
    except ValueError:   # another drive (Windows)
        return p


def load(paths):
    """Records from every file, de-duplicated (P4 may copy the same file twice)."""
    seen, out = set(), []
    for p in paths:
        try:
            lines = open(p).read().splitlines()
        except OSError as e:
            print(f"  skipping {p}: {e}", file=sys.stderr)
            continue
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skipping {p}:{n}: not JSON", file=sys.stderr)
                continue
            key = (rec.get("seat"), rec.get("run_id"), rec.get("attempt_no"), rec.get("code_hash"))
            if key not in seen:
                seen.add(key)
                rec["_file"] = _short(p)
                out.append(rec)
    return out


def first_line(s, n=110):
    s = (s or "").strip().splitlines()[0] if (s or "").strip() else ""
    return s if len(s) <= n else s[:n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--json", metavar="OUT", help="write the table as JSON (for the dashboard)")
    a = ap.parse_args()

    found = sorted(glob.glob(os.path.join(CHIPBOOST, "logs", "seat-*", "attempts*.jsonl"))
                   + glob.glob(os.path.join(CHIPBOOST, "logs", "seat-*", "sim", "*.jsonl")))
    recs = load(found + list(a.files))
    if not recs:
        sys.exit("no attempt records found (looked in logs/seat-*/attempts*.jsonl and logs/seat-*/sim/)")

    cols = sorted({(r.get("arm") or "?", label(r)) for r in recs})
    table = {m: {c: 0 for c in cols} for m, _ in MODES}
    example = {}
    for r in recs:
        m = classify(r)
        table[m][(r.get("arm") or "?", label(r))] += 1
        example.setdefault(m, r)
    assert sum(sum(v.values()) for v in table.values()) == len(recs), "every record in exactly one mode"

    heads = [f"{arm}/{lbl}" for arm, lbl in cols]
    w = max(len(m) for m, _ in MODES)
    print(f"{len(recs)} attempts from {len({r['_file'] for r in recs})} file(s); "
          f"columns are arm/referee (sim = simulator-only fallback, chip = P1's speedcheck)\n")
    print(f"{'failure mode':<{w}} | " + " | ".join(f"{h:>18}" for h in heads) + " | total")
    print("-" * (w + 3 + 21 * len(heads) + 8))
    for part, modes in (("CORRECT", [m for m, _ in MODES if m in CORRECT]),
                        ("FAILED", [m for m, _ in MODES if m not in CORRECT])):
        print(f"{part}")
        for m in modes:
            row = table[m]
            if not sum(row.values()):
                continue
            print(f"{m:<{w}} | " + " | ".join(f"{row[c]:>18}" for c in cols) + f" | {sum(row.values()):>5}")
    totals = {c: sum(table[m][c] for m in table) for c in cols}
    print(f"{'all attempts':<{w}} | " + " | ".join(f"{totals[c]:>18}" for c in cols) + f" | {len(recs):>5}")

    print("\none example per mode (seat, arm, run, attempt: first line of what the referee said)")
    for m, _ in MODES:
        r = example.get(m)
        if r is None:
            continue
        said = r.get("referee_message") or r.get("instruction_given") or ""
        print(f"  {m}\n      seat {r.get('seat')} {r.get('arm')} {r.get('run_id')} #{r.get('attempt_no')} "
              f"[{label(r)}]: {first_line(said)}")
    others = [r for r in recs if classify(r) == "other wrong"]
    if others:
        print(f"\n{len(others)} attempt(s) no rule names -- read these and add a rule:")
        for r in others[:5]:
            print(f"  {r['_file']} #{r.get('attempt_no')}: {first_line(r.get('referee_message'), 160)}")

    if a.json:
        rows = [dict(mode=m, correct=m in CORRECT,
                     counts={f"{arm}/{lbl}": table[m][(arm, lbl)] for arm, lbl in cols},
                     total=sum(table[m].values()),
                     example=None if m not in example else dict(
                         seat=example[m].get("seat"), arm=example[m].get("arm"),
                         run_id=example[m].get("run_id"), attempt_no=example[m].get("attempt_no"),
                         label=label(example[m]),
                         said=first_line(example[m].get("referee_message")
                                         or example[m].get("instruction_given"), 300)))
                for m, _ in MODES if sum(table[m].values())]
        with open(a.json, "w") as f:
            json.dump(dict(n_attempts=len(recs), files=sorted({r["_file"] for r in recs}), modes=rows), f,
                      indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
