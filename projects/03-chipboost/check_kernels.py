#!/usr/bin/env python3
"""
check_kernels.py -- every P2 kernel against every shape it should handle, in the CPU simulator. Owner: P2.

The fast gate before anything goes to the chip: rules, correctness against shapes.py's float32
reference (nkibench.describe_mismatch at the op's tolerance), inputs untouched, no simulator warning of a
hardware hazard, and the HBM bytes each kernel moved against the floor. The referee's stricter bf16-ulp
check and the timing happen on the chip (speedcheck.py, heldout_grid.py), not here.

    python check_kernels.py                          # dev + held-out shapes
    python check_kernels.py --which dev --only matmul_expert,rmsnorm_start
    python check_kernels.py --timing                 # also the timing shapes: big, minutes
    python check_kernels.py --json check.json        # one object per row
    python check_kernels.py --dry-run                # build every input and reference; no nki needed
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import shapes  # noqa: E402  also puts ../02-kernel-agent on sys.path
import nkibench  # noqa: E402

KERNELS = [("matmul", "kernels/matmul_start.py"), ("matmul", "kernels/matmul_expert.py"),
           ("matmul", "kernels/matmul_expert_aws.py"), ("rmsnorm", "kernels/rmsnorm_start.py"),
           ("copy", "kernels/copy_tiled.py"), ("copy", "kernels/copy_floor.py"),
           ("swiglu", "kernels/swiglu_start.py")]


def check_case(kernel, op, case, seed):
    args = shapes.make_inputs(op, case, seed)
    before = [a.copy() for a in args]
    want = shapes.reference(op, args)
    t0 = time.time()
    got, counted = nkibench.simulate_and_count(kernel, args)
    dt = time.time() - t0
    msg = (nkibench.check_inputs_untouched(before, args)
           or nkibench.describe_mismatch(got, want, shapes.tolerance(op)))
    hazards = [w for w in counted.get("warnings", []) if "incorrect results on hardware" in w]
    if hazards and not msg:
        msg = "CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
    return dict(passed=not msg, message=(msg or "").splitlines()[0] if msg else "", sim_s=round(dt, 2),
                bytes=counted["bytes"], floor_bytes=shapes.work(op, case)[1],
                transfers=counted["transfers"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="dev,heldout", help="comma list of dev, timing, heldout")
    ap.add_argument("--timing", action="store_true", help="also the timing shapes (big)")
    ap.add_argument("--only", default="", help="kernel file stems, e.g. matmul_expert,copy_floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--dry-run", action="store_true", help="inputs and references only; no nki")
    a = ap.parse_args()

    which = [w.strip() for w in a.which.split(",") if w.strip()]
    if a.timing and "timing" not in which:
        which.append("timing")
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if "timing" in which:
        print("note: the timing shapes are Qwen3's real sizes; the simulator takes minutes on them")

    if not a.dry_run:
        try:
            import nki  # noqa: F401
        except ImportError:
            print("nki is not installed here: run this in the seat pod (or use --dry-run).")
            return 2

    rows = []
    for op, path in KERNELS:
        stem = os.path.splitext(os.path.basename(path))[0]
        if only and stem not in only:
            continue
        if not os.path.exists(os.path.join(HERE, path)):
            print(f"{stem:<18} not written yet")
            continue
        kernel = None if a.dry_run else nkibench.load_kernel(os.path.join(HERE, path), shapes.entry(op))
        for w in which:
            for case in shapes.cases(op, w):
                label = shapes.label(op, case)
                if a.dry_run:
                    args = shapes.make_inputs(op, case, a.seed)
                    want = shapes.reference(op, args)
                    print(f"{stem:<18} {w:<8} {label:<62} inputs {[x.shape for x in args]} -> {want.shape}")
                    continue
                try:
                    r = check_case(kernel, op, case, a.seed)
                except Exception as e:
                    r = dict(passed=False, message=f"RAISED {type(e).__name__}: {str(e)[:200]}", sim_s=None,
                             bytes=None, floor_bytes=shapes.work(op, case)[1], transfers=None)
                r.update(kernel=stem, op=op, which=w, shape=label)
                rows.append(r)
                ratio = f"{r['bytes'] / r['floor_bytes']:.2f}x floor" if r["bytes"] else ""
                sim = f"{r['sim_s']:.1f}s" if r["sim_s"] is not None else "-"
                print(f"{stem:<18} {w:<8} {label:<62} {'PASS' if r['passed'] else 'FAIL'} {sim:>7} "
                      f"{ratio:>12} {str(r['transfers'] or ''):>6}  {r['message'][:120]}", flush=True)

    if a.dry_run:
        print("dry run: inputs and references build for every case")
        return 0
    failed = sum(1 for r in rows if not r["passed"])
    print(f"\n{len(rows) - failed} passed, {failed} failed")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
