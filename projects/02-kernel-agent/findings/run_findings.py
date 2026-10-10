#!/usr/bin/env python3
"""
run_findings.py -- run every finding through the organisers' own check, and say what it shows.

For each kernel in this folder: what `nkibench.py --level N --check` prints, which is the
organisers' grader with nothing of ours in it, next to what is actually wrong with the kernel.
Needs the Neuron SDK, so run it in a seat pod:

    python findings/run_findings.py
"""
import contextlib
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import nkibench  # noqa: E402

CASES = [
    ("tall_tile_level4", 4, "passes", "allocates a 256- and 512-row SBUF tile"),
    ("psum_output_level3", 3, "passes", "returns its PSUM tile as the output"),
    ("misleading_reshape_level3", 3, "cannot reshape", "has no reshape; its result tile is (1, 64)"),
    ("misleading_moving_level4", 4, "moving free dimension", "both operands are within limits; the result tile is 1024 wide"),
]


def main():
    import nki
    print(f"simulator: nki {getattr(nki, '__version__', '?')}\n")
    summary = []
    for name, level, expect, truth in CASES:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = nkibench.verify(os.path.join(HERE, f"{name}.py"), level)
        text = out.getvalue()
        print(f"{'=' * 78}\n{name}.py  --  nkibench.py --level {level} --check  (exit {rc})\n{'=' * 78}")
        print(text.strip())
        if expect == "passes":
            seen = rc == 0
        else:
            seen = expect in text
        summary.append((name, rc, seen, truth))
        print()
    # the rule scan sees a literal height and not a symbolic one
    src = open(os.path.join(HERE, "tall_tile_level4.py")).read()
    literal = src.replace("nl.ndarray((K, M),", "nl.ndarray((256, M),")
    lit, sym = nkibench.check_rules(literal, 4), nkibench.check_rules(src, 4)
    print(f"rule scan on the tall tile written as (256, M): {lit or 'clean'}")
    print(f"rule scan on the same tile written as (K, M):   {sym or 'clean'}\n")

    print("SUMMARY")
    for name, rc, seen, truth in summary:
        verdict = "PASSES the organisers' check" if rc == 0 else f"check exits {rc}"
        print(f"  {'REPRODUCED' if seen else 'NOT REPRODUCED':15s} {name:28s} {verdict}; actually: {truth}")
    print(f"  {'REPRODUCED' if lit and not sym else 'NOT REPRODUCED':15s} {'rule scan blind spot':28s} "
          f"catches (256, M), not (K, M)")
    stale = subprocess.run([sys.executable, os.path.join(HERE, "stale_bytecode.py")],
                           capture_output=True, text=True).stdout.strip().splitlines()
    print(f"  {'REPRODUCED' if stale and stale[-1].startswith('REPRODUCED') else 'NOT REPRODUCED':15s} "
          f"{'stale_bytecode':28s} {stale[0] if stale else 'did not run'}")


if __name__ == "__main__":
    main()
