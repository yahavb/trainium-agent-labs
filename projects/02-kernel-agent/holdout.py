#!/usr/bin/env python3
"""
holdout.py -- does a "verified" kernel still work on shapes the checker never showed it?

nkibench.py grades each level on 3-4 fixed shapes. A model that sees those shapes in the error
messages can pass them with code that is wrong in general (a special case for the square shape, an
off-by-one that happens to cancel). This runs a kernel on EXTRA shapes, never used for grading or
feedback, and reports which pass.

Fairness: a held-out shape the shipped reference kernel itself fails (for instance a matmul tutorial
kernel that assumes tile-multiple sizes) says nothing about the candidate. `check_holdout` therefore
also runs reference_level{n}.py when it exists and marks those shapes 'ref_fails'; they are
excluded from the verdict.

    python holdout.py --level 2 --check my_kernel.py
    python holdout.py --level 2 --check reference_level2.py

Needs the Neuron SDK (it simulates), like nkibench.py.
"""

import argparse
import os
import sys
import tempfile

import numpy as np

import nkibench

HERE = os.path.dirname(os.path.abspath(__file__))

# Shapes deliberately different from nkibench.LEVELS[n]["shapes"]: odd sizes, a single partition row,
# a free extent of 1 or 2, and more than one tile where the level allows it.
HOLDOUT = {
    1: [dict(shape=(16, 12, 12), pool_size=2), dict(shape=(100, 9, 9), pool_size=3),
        dict(shape=(1, 8, 8), pool_size=4), dict(shape=(48, 20, 20), pool_size=5)],
    2: [dict(shape=(16, 6), shape2D=(2, 3)), dict(shape=(100, 30), shape2D=(5, 6)),
        dict(shape=(1, 7), shape2D=(1, 7)), dict(shape=(128, 20), shape2D=(4, 5)),
        dict(shape=(5, 16), shape2D=(4, 4)), dict(shape=(64, 2), shape2D=(2, 1))],
    3: [dict(K=96, M=48, N=300), dict(K=128, M=100, N=200)],
    4: [dict(K=384, M=256, N=512), dict(K=128, M=384, N=1536)],
    8: [dict(seq=32, dim=16), dict(seq=100, dim=48)],
}


def _write(code):
    fd, path = tempfile.mkstemp(suffix=".py", prefix="_holdout_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _run_cases(level, code, cases):
    """-> list of (label, 'ok'|'fail'|'error', message). Never raises for a kernel problem."""
    spec = nkibench.LEVELS[level]
    path = _write(code)
    out = []
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
        for case in cases:
            lab = nkibench.label(case, level)
            try:
                args, _ = nkibench.make_inputs(case, level, seed=101)     # not the grading seed
                want = spec["ref"](*args)
                got, _ = nkibench.simulate_and_count(kernel, args)
            except nkibench.NkiMissing:
                raise
            except Exception as e:
                out.append((lab, "error", f"{type(e).__name__}: {str(e)[:160]}"))
                continue
            miss = nkibench.describe_mismatch(got, want)
            out.append((lab, "fail", miss.splitlines()[0][:160]) if miss else (lab, "ok", ""))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return out


def check_holdout(level, code, reference_code=None):
    """Run `code` on the held-out shapes of `level`.

    Returns {"available": bool, "passed": n, "total": m, "cases": [...], "ref_fails": [...],
             "ok": bool}. `ok` is True when every shape the reference passes is also passed.
    """
    cases = HOLDOUT.get(level)
    if not cases:
        return {"available": False, "passed": 0, "total": 0, "cases": [], "ref_fails": [], "ok": True}
    try:
        got = _run_cases(level, code, cases)
    except nkibench.NkiMissing:
        return {"available": False, "passed": 0, "total": 0, "cases": [], "ref_fails": [], "ok": True}
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}", "passed": 0, "total": 0,
                "cases": [], "ref_fails": [], "ok": True}
    if reference_code is None:
        ref_path = os.path.join(HERE, f"reference_level{level}.py")
        if os.path.exists(ref_path):
            with open(ref_path, encoding="utf-8") as f:
                reference_code = f.read()
    ref_fails = []
    if reference_code:
        try:
            ref = _run_cases(level, reference_code, cases)
            ref_fails = [lab for lab, status, _ in ref if status != "ok"]
        except Exception:
            ref_fails = []
    scored = [(lab, s, m) for lab, s, m in got if lab not in ref_fails]
    passed = sum(1 for _, s, _ in scored if s == "ok")
    return {"available": True, "passed": passed, "total": len(scored),
            "cases": [{"label": lab, "status": s, "message": m} for lab, s, m in got],
            "ref_fails": ref_fails, "ok": passed == len(scored)}


def format_holdout(result):
    """A sentence usable as verifier feedback when a kernel passes the grader but not the hold-out."""
    if not result.get("available") or result.get("ok"):
        return ""
    bad = [c for c in result["cases"] if c["status"] != "ok" and c["label"] not in result["ref_fails"]]
    shown = "; ".join(f"{c['label']}: {c['message']}" for c in bad[:2])
    return (f"HELD-OUT SHAPES: the kernel passes the checker's shapes but fails "
            f"{len(bad)} of {result['total']} other shapes ({shown}). It is correct only for the "
            f"shapes it was shown. Derive every size from the input tensor instead of special-casing.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, required=True, choices=sorted(HOLDOUT))
    p.add_argument("--check", required=True, help="kernel file to test")
    a = p.parse_args()
    with open(a.check, encoding="utf-8") as f:
        res = check_holdout(a.level, f.read())
    if not res["available"]:
        sys.exit(f"hold-out unavailable (no Neuron SDK, or no shapes for level {a.level}). "
                 f"{res.get('error', '')}")
    print(f"level {a.level}: {res['passed']}/{res['total']} held-out shapes passed"
          + (f"  (excluded, the reference fails them too: {', '.join(res['ref_fails'])})"
             if res["ref_fails"] else ""))
    for c in res["cases"]:
        print(f"  {c['status']:5s} {c['label']}  {c['message']}")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()