"""C3 test: scores land where they should, and feedback A < B < C in information -- with C never
handing over the whole answer.

    python veriloop/tests/test_feedback.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import checker  # noqa: E402

FIX = os.path.join(HERE, "sim_fixtures")
COMPILE = os.path.join(HERE, "compile_cases")

CASES = [
    # fixture   design file                          stage        score range     C must mention
    ("adder4",   "good.v",                           "solved",    (1.0, 1.0),     "PASS"),
    ("adder4",   os.path.join(COMPILE, "semi.v"),    "compile",   (0.0, 0.0),     "missing `;`"),
    ("adder4",   "bad_finish.v",                     "run",       (0.2, 0.2),     "$finish"),
    ("adder4",   "bad_nocarry.v",                    "behaviour", (0.3, 0.999),   "`cout` = 0, expected 1"),
    ("adder4",   "bad_xor.v",                        "behaviour", (0.3, 0.999),   "`cout`"),
    ("counter8", "good.v",                           "solved",    (1.0, 1.0),     "PASS"),
    ("counter8", "bad_noenable.v",                   "behaviour", (0.3, 0.999),   "first wrong cycle"),
    ("counter8", "bad_saturate.v",                   "behaviour", (0.9, 0.999),   "cycle 259"),
    ("counter8", "bad_noreset.v",                    "behaviour", (0.3, 0.3),     "undefined"),
]


def check_levels(fb, stage):
    """A says only pass/fail; B adds amounts but no values; C adds where -- at most a few rows."""
    problems = []
    if re.search(r"\d", fb["A"]) or "`" in fb["A"]:
        problems.append("A leaks detail")
    if stage == "behaviour" and ("expected" in fb["B"] or "`" in fb["B"]):
        problems.append("B leaks values")
    if stage == "behaviour" and fb["C"].count("expected") > checker.MAX_EXAMPLES:
        problems.append("C shows too many rows")
    if stage != "solved" and not (len(fb["A"]) < len(fb["B"]) < len(fb["C"])):
        problems.append("A < B < C does not hold")
    return problems


def main():
    failed = 0
    for fixture, design, stage, (lo, hi), phrase in CASES:
        ref = checker.load_level(os.path.join(FIX, fixture))
        path = design if os.path.isabs(design) else os.path.join(FIX, fixture, design)
        g = checker.grade(ref, open(path).read())
        problems = check_levels(g.feedback, g.stage)
        if g.stage != stage:
            problems.append(f"stage {g.stage}, wanted {stage}")
        if not lo <= g.score <= hi:
            problems.append(f"score {g.score} outside {lo}-{hi}")
        if phrase not in g.feedback["C"]:
            problems.append(f"C does not mention {phrase!r}")
        if g.solved != (stage == "solved"):
            problems.append("solved flag wrong")
        failed += bool(problems)
        print(f"  {'ok  ' if not problems else 'FAIL'} {fixture}/{os.path.basename(design):<18} "
              f"score {g.score:.2f}  {g.stage:<9} {'; '.join(problems)}")
    print(f"\nC3 feedback test: {len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
