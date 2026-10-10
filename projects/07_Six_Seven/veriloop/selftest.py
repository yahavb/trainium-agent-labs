"""C4: prove every level before the model sees it.

    python veriloop/selftest.py                        # every level in veriloop/levels/
    python veriloop/selftest.py veriloop/levels/04_traffic_fsm   # just one (use while writing a level)
    python veriloop/selftest.py --fixtures             # also the checker's own test fixtures

For each level folder (contract: veriloop/levels/README.md) it checks:
  - spec.txt, reference.py, good.v and at least three bad_*.v exist
  - spec.txt names the module and every port, so the model is told exactly what to build
  - reference() gives one answer per test input, with exactly the declared outputs
  - good.v scores 1.0
  - every bad_*.v is caught, and at least one is caught by SIMULATION (not only by the compiler) --
    otherwise the level's tests are never proven to catch wrong behaviour
A level that fails here would grade the model wrongly, so fix it before running experiments on it.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import checker  # noqa: E402

LEVELS = os.path.join(HERE, "levels")
FIXTURES = os.path.join(HERE, "tests", "sim_fixtures")


def level_dirs(root):
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "reference.py")))


def check_level(d):
    """Returns (problems, notes). Any problem means the level is not ready."""
    problems, notes = [], []
    need = ["spec.txt", "reference.py", "good.v"]
    missing = [f for f in need if not os.path.isfile(os.path.join(d, f))]
    bads = sorted(f for f in os.listdir(d) if f.startswith("bad_") and f.endswith(".v"))
    if missing:
        problems.append("missing " + ", ".join(missing))
    if len(bads) < 3:
        problems.append(f"only {len(bads)} bad_*.v file(s); the contract asks for at least 3")
    if missing:
        return problems, notes

    try:
        ref = checker.load_level(d)
        vecs = ref.vectors()
        want = ref.reference(vecs)
    except Exception as e:
        return problems + [f"reference.py fails to load or run: {type(e).__name__}: {e}"], notes

    # reference.py must agree with itself
    if len(want) != len(vecs):
        problems.append(f"reference() returned {len(want)} answers for {len(vecs)} test inputs")
    bad_keys = [i for i, w in enumerate(want) if set(w) != set(ref.OUTPUTS)]
    if bad_keys:
        problems.append(f"reference() answer {bad_keys[0]} has outputs {sorted(want[bad_keys[0]])}, "
                        f"expected exactly {sorted(ref.OUTPUTS)}")
    missing_in = [i for i, v in enumerate(vecs) if set(v) != set(ref.INPUTS)]
    if missing_in:
        problems.append(f"vectors() step {missing_in[0]} sets {sorted(vecs[missing_in[0]])}, "
                        f"expected exactly {sorted(ref.INPUTS)}")
    if not bad_keys:
        too_wide = [(n, w[n]) for w in want for n, width in ref.OUTPUTS.items()
                    if not 0 <= int(w[n]) < (1 << width)]
        if too_wide:
            notes.append(f"reference() gives `{too_wide[0][0]}` = {too_wide[0][1]}, wider than its port; "
                         f"the checker keeps only the low bits -- make sure that is intended")
    if problems:
        return problems, notes

    # the spec must tell the model the exact interface
    spec = open(os.path.join(d, "spec.txt")).read()
    names = [ref.MODULE] + (["clk"] if ref.CLOCKED else []) + list(ref.INPUTS) + list(ref.OUTPUTS)
    unnamed = [n for n in names if not re.search(rf"\b{re.escape(n)}\b", spec)]
    if unnamed:
        problems.append("spec.txt never mentions " + ", ".join(f"`{n}`" for n in unnamed))

    # good must pass, every bad must be caught
    g = checker.grade(ref, open(os.path.join(d, "good.v")).read())
    if not g.solved:
        problems.append(f"good.v is NOT accepted (score {g.score:.2f}):\n      "
                        + g.feedback["C"].replace("\n", "\n      "))
    caught_by_sim = 0
    for b in bads:
        src = open(os.path.join(d, b)).read()
        gb = checker.grade(ref, src)
        if gb.solved:
            problems.append(f"{b} is ACCEPTED -- the tests do not catch it; add test inputs that expose it")
            continue
        caught_by_sim += gb.stage in ("behaviour", "run")
        if not src.lstrip().startswith("//"):
            notes.append(f"{b} has no top comment saying what is wrong")
        notes.append(f"{b}: caught at {gb.stage}, score {gb.score:.2f}")
    if bads and caught_by_sim == 0:
        problems.append("every bad_*.v fails only at compile; add one that compiles but behaves wrongly")
    notes.insert(0, f"{len(vecs)} tests, {'clocked' if ref.CLOCKED else 'combinational'}")
    return problems, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("levels", nargs="*", help="level folders (default: every folder in veriloop/levels/)")
    ap.add_argument("--fixtures", action="store_true", help="also check the checker's test fixtures")
    ap.add_argument("-q", "--quiet", action="store_true", help="only the verdict per level")
    a = ap.parse_args()

    dirs = a.levels or level_dirs(LEVELS)
    if a.fixtures:
        dirs += level_dirs(FIXTURES)
    if not dirs:
        print("No level folders yet (each needs a reference.py). See veriloop/levels/README.md.")
        return 0

    failed = 0
    for d in dirs:
        problems, notes = check_level(d)
        name = os.path.relpath(d, HERE) if os.path.abspath(d).startswith(HERE) else d
        print(f"{'PASS' if not problems else 'FAIL'}  {name}")
        for p in problems:
            print(f"   x {p}")
        if not a.quiet:
            for n in notes:
                print(f"     {n}")
        failed += bool(problems)
    print(f"\nselftest: {len(dirs) - failed}/{len(dirs)} level(s) ready")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
