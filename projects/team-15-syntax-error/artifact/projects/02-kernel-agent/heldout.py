#!/usr/bin/env python3
"""
heldout.py -- hostile cases the agent NEVER sees, and calibration of the agent's confidence.

The agent is graded on the public shapes in nkibench.LEVELS. A kernel that passes them can still be
wrong: hard-coded tile counts that only work when a dimension divides by 128, a 1-row edge case, an
overflow that only shows on large magnitudes. This file holds, per level, extra cases drawn from the
same distribution the judges describe (prime/ragged dims, a dim of 1, large magnitudes, constant
rows), and two functions:

  confidence(src, level)  -> (c in [0,1], reasons). Computed from the kernel text and the PUBLIC
                             result only, before any held-out case runs. This is what the agent
                             prints next to "SOLVED".
  evaluate(src, level)    -> per held-out case pass/fail, through the same simulator and mismatch
                             check as the public grade.

    python heldout.py --check reference_level4.py --level 4
    python heldout.py --logs 'runs/*.jsonl' > ../../CALIBRATION.md

Calibration is scored on every kernel the agent CLAIMED solved (public reward 1.0): Brier score of
confidence vs "passes every held-out case", plus a reliability table. Kernels the agent did not
claim are reported with confidence 0 ("could not verify"), which is honest by construction.
"""

import argparse
import ast
import glob
import json
import os
import re
import sys

import numpy as np

import nkibench

# value transforms applied to the generated float inputs
def _large(args):
    return tuple(a * 1e4 if isinstance(a, np.ndarray) else a for a in args)


def _const_row(args):
    out = []
    for a in args:
        if isinstance(a, np.ndarray):
            a = a.copy()
            a[0] = a.flat[0]          # first partition row all identical
        out.append(a)
    return tuple(out)


_SEQ = __import__("itertools").count()

HELDOUT = {
    1: [(dict(shape=(1, 8, 8), pool_size=2), None, "C=1"),
        (dict(shape=(128, 7, 7), pool_size=2), None, "H,W not divisible by pool"),
        (dict(shape=(16, 9, 9), pool_size=3), _large, "x1e4 magnitudes"),
        (dict(shape=(5, 6, 6), pool_size=2), _const_row, "constant row")],
    2: [(dict(shape=(1, 6), shape2D=(2, 3)), None, "P=1"),
        (dict(shape=(128, 35), shape2D=(5, 7)), None, "P=128, prime factors"),
        (dict(shape=(7, 13), shape2D=(13, 1)), None, "F2=1"),
        (dict(shape=(16, 12), shape2D=(4, 3)), _large, "x1e4 magnitudes")],
    # level 3 is DEFINED as one fixed tile (K=128 M=64 N=512): hostile values only
    3: [(dict(K=128, M=64, N=512), _large, "x1e4 magnitudes"),
        (dict(K=128, M=64, N=512), _const_row, "constant row")],
    4: [(dict(K=96, M=200, N=700), None, "ragged in K, M and N"),
        (dict(K=384, M=1, N=513), None, "M=1, N=512+1"),
        (dict(K=131, M=128, N=512), None, "prime K > 128"),
        (dict(K=128, M=128, N=512), _large, "x1e4 magnitudes")],
}
for _lv in (5, 6, 7):
    HELDOUT[_lv] = HELDOUT[4]


def _x30(args):
    return tuple(a * 30.0 if isinstance(a, np.ndarray) else a for a in args)


# level 8's trap: large scores overflow a naive exp(); a kernel without max-subtraction returns NaN
HELDOUT[8] = [(dict(seq=128, dim=128), _x30, "x30 inputs: scores ~1e3, naive exp overflows"),
              (dict(seq=1, dim=8), None, "seq=1"),
              (dict(seq=127, dim=1), None, "dim=1, seq prime"),
              (dict(seq=32, dim=16), _const_row, "constant row")]


def _run_case(kernel, level, spec, transform):
    args, _ = nkibench.make_inputs(spec, level, seed=1000)
    if transform:
        args = transform(args)
    want = nkibench.LEVELS[level]["ref"](*args)
    try:
        got, counted = nkibench.simulate_and_count(kernel, args)
    except Exception as e:  # noqa: BLE001 -- any raise is a held-out failure
        return False, f"raised {type(e).__name__}: {str(e)[:160]}"
    m = nkibench.describe_mismatch(got, want)
    return (m is None), (m or "ok").split("\n")[0][:160]


def evaluate(src, level):
    path = f"/tmp/_heldout_{level}_{os.getpid()}_{next(_SEQ)}.py"   # fresh path: see agent.grade
    with open(path, "w") as f:
        f.write(src)
    try:
        kernel = nkibench.load_kernel(path, nkibench.LEVELS[level]["entry"])
    except Exception as e:  # noqa: BLE001
        return [(lbl, False, f"load failed: {e}") for _, _, lbl in HELDOUT.get(level, [])]
    return [(lbl,) + _run_case(kernel, level, spec, tr) for spec, tr, lbl in HELDOUT.get(level, [])]


# ---------------------------------------------------------------- confidence

def _public_has_ragged(level):
    """Does any PUBLIC shape exercise a partial tile? If not, ragged handling is unverified."""
    for sp in nkibench.LEVELS[level]["shapes"]:
        if "K" in sp:
            if sp["K"] % 128 or sp["M"] % 128 or sp["N"] % 512:
                return True
        elif "shape" in sp and sp["shape"][0] % 128:
            return True
    return False


def confidence(src, level, hazards=0):
    """Confidence that a kernel that passed every PUBLIC case is correct in general.

    Starts at 0.95 and is cut by evidence visible in the code or the public test set -- never by
    the held-out result. The cuts are judgement, stated so that the calibration table can prove or
    disprove them:
      x0.4  shape asserts (assert ... % ... / == ...): the kernel itself refuses other shapes
      x0.5  no remainder handling (no min(), no %, no ceil-div) while the public shapes never had a
            partial tile -- ragged inputs were simply never tested
      x0.7  hard-coded literal equal to a public dimension other than the tile limits
      x0.3  the simulator raised a hardware-hazard warning
    """
    c, why = 0.95, []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0.0, ["does not parse"]
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    if any(re.search(r"%|==", ast.unparse(a.test)) for a in asserts):
        c *= 0.4
        why.append("asserts on input shape (refuses shapes outside the public set)")
    # the stable form needs BOTH a row max and a subtraction of it (a computed-but-unused max is
    # exactly the naive kernel in handwritten/sk8_attention_NAIVE_overflows.py)
    if level == 8 and not (re.search(r"nl.maximum", src) and re.search(r"nl.subtract", src)):
        c *= 0.3
        why.append("no row-max subtraction before exp: large scores overflow to NaN")
    remainder = re.search(r"\bmin\(|%|\+ *\w+ *- *1\) *//|-\(-", src)
    # levels 3 and 8 are single-tile by definition, so ragged handling is not a hazard there
    if level not in (3, 8) and not remainder and not _public_has_ragged(level):
        c *= 0.5
        why.append("no remainder handling, and no public shape had a partial tile")
    public_dims = set()
    for sp in nkibench.LEVELS[level]["shapes"]:
        for v in sp.values():
            for d in (v if isinstance(v, tuple) else (v,)):
                if isinstance(d, int) and d not in (1, 2, 128, 512):
                    public_dims.add(d)
    lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, int)}
    hard = sorted(lits & public_dims)
    if hard and level not in (3, 8):
        c *= 0.7
        why.append(f"hard-coded public dimensions {hard}")
    if hazards:
        c *= 0.3
        why.append("simulator hazard warning")
    return round(c, 3), why or ["all public shapes pass; remainder handling present"]


# ---------------------------------------------------------------- calibration over logs

def calibrate(paths):
    claims = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("reward", 0) >= 1.0 - 1e-9 and r.get("code"):
                claims.setdefault((r["level"], r["code"]), r.get("tag"))
    rows = []
    for (lv, code), tag in sorted(claims.items(), key=lambda kv: (kv[0][0], str(kv[1]))):
        conf, why = confidence(code, lv)
        res = evaluate(code, lv)
        ok = all(passed for _, passed, _ in res)
        rows.append((lv, tag, conf, ok, why, res))
    return rows


def report(rows, extra=()):
    sys.stdout.reconfigure(encoding="utf-8")
    allrows = list(rows) + list(extra)
    print("# CALIBRATION — confidence vs held-out truth\n")
    print("Every kernel the agent claimed SOLVED (public reward 1.0), re-run on held-out hostile "
          "cases it never saw (`projects/02-kernel-agent/heldout.py`). Confidence is computed from "
          "the code and the public result only. All **[sim]**.\n")
    if not allrows:
        print("_No solved claims yet._")
        return
    brier = sum((c - (1.0 if ok else 0.0)) ** 2 for _, _, c, ok, _, _ in allrows) / len(allrows)
    print(f"**Brier score {brier:.3f}** over n={len(allrows)} claims (0 = perfect, 0.25 = always "
          f"saying 0.5).\n")
    print("| level | source | confidence | held-out | why that confidence |")
    print("|---|---|---|---|---|")
    for lv, tag, c, ok, why, res in allrows:
        fails = [f"{l}: {m}" for l, p, m in res if not p]
        print(f"| L{lv} | {tag} | {c:.2f} | {'PASS' if ok else 'FAIL — ' + '; '.join(fails)[:200]} "
              f"| {'; '.join(why)} |")
    bins = [(0, .5), (.5, .8), (.8, 1.01)]
    print("\n| confidence bin | n | predicted | observed pass rate |")
    print("|---|---|---|---|")
    for lo, hi in bins:
        b = [(c, ok) for _, _, c, ok, _, _ in allrows if lo <= c < hi]
        if b:
            print(f"| [{lo:.1f}, {min(hi, 1):.1f}) | {len(b)} | {sum(c for c, _ in b) / len(b):.2f} | "
                  f"{sum(ok for _, ok in b) / len(b):.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    ap.add_argument("--level", type=int)
    ap.add_argument("--logs", nargs="*")
    ap.add_argument("--refs", action="store_true",
                    help="also score the shipped reference kernels as labelled calibration points")
    a = ap.parse_args()
    if a.check:
        src = open(a.check).read()
        conf, why = confidence(src, a.level)
        print(f"confidence {conf:.2f}: {'; '.join(why)}")
        for lbl, ok, msg in evaluate(src, a.level):
            print(f"  {'PASS' if ok else 'FAIL'}  {lbl:28s} {'' if ok else msg}")
        return
    rows = calibrate([p for g in (a.logs or []) for p in glob.glob(g)])
    extra = []
    if a.refs:
        for lv in (1, 2, 3, 4, 5, 6, 7):
            try:
                src = open(f"reference_level{lv}.py").read()
            except FileNotFoundError:
                continue
            conf, why = confidence(src, lv)
            res = evaluate(src, lv)
            extra.append((lv, f"reference_level{lv}.py", conf, all(p for _, p, _ in res), why, res))
    report(rows, extra)


if __name__ == "__main__":
    main()
