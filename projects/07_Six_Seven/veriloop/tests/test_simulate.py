"""C2 test: the checker must pass correct designs and catch wrong behaviour in simulation.

    python veriloop/tests/test_simulate.py

Fixtures in sim_fixtures/ (an adder and a counter -- test material, not real levels). Every bad design
compiles; only the simulation can tell it is wrong. Run after any change to checker.py.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import checker  # noqa: E402

FIX = os.path.join(HERE, "sim_fixtures")

CASES = [
    # fixture   design                   expected outcome                 detail that must appear
    ("adder4",   "good.v",                "pass",    ""),
    ("adder4",   "bad_nocarry.v",         "fail",    "cout"),
    ("adder4",   "bad_xor.v",             "fail",    "sum"),
    ("adder4",   "bad_undriven.v",        "fail",    "undefined"),
    ("adder4",   "bad_finish.v",          "no-run",  "$finish"),
    ("counter8", "good.v",                "pass",    ""),
    ("counter8", "bad_noenable.v",        "fail",    "count"),
    ("counter8", "bad_saturate.v",        "fail",    "count"),
    ("counter8", "bad_noreset.v",         "fail",    "undefined"),
    ("counter8", "bad_enable_priority.v", "fail",    "count"),
    ("counter8", "bad_loop.v",            "no-run",  "never"),
]


def outcome(r):
    if not r.compiled.ok:
        return "no-compile", " ".join(r.compiled.messages)
    if not r.ran:
        return "no-run", r.problem
    if r.passed:
        return "pass", ""
    detail = " ".join(f"{m['signal']} {'undefined' if m['got'] is None else m['got']}" for m in r.mismatches[:5])
    return "fail", detail


def main():
    failed = 0
    for fixture, design, want, phrase in CASES:
        ref = checker.load_level(os.path.join(FIX, fixture))
        r = checker.simulate(ref, open(os.path.join(FIX, fixture, design)).read())
        got, detail = outcome(r)
        ok = got == want and phrase in detail
        failed += not ok
        wrong = len({m["step"] for m in r.mismatches})
        note = f"{wrong}/{len(r.rows)} steps wrong" if got == "fail" else detail[:70]
        print(f"  {'ok  ' if ok else 'FAIL'} {fixture}/{design:<24} {got:<10} {note}")
        if not ok:
            print(f"         expected {want} mentioning {phrase!r}")
    print(f"\nC2 simulate test: {len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
