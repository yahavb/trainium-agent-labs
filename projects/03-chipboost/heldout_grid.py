#!/usr/bin/env python3
"""
heldout_grid.py -- the end-of-run held-out check, as a before-and-after grid for the dashboard's panel 5.
Owner: P2. Run it once, after the loops finish (about 17:30), in a seat pod.

The referee checks held-out shapes only until the first failure and never times them, and the loops run
with heldout=False to save compiles. So no one has asked the question panel 5 answers: does each arm's
best kernel stay correct, and stay faster, at shapes it never saw? This asks it.

Rows: the start kernel, the expert (matmul), and the best verified kernel of each arm, found in every
attempts*.jsonl under this folder (verdict "faster", highest speedup; its code is in the log). Columns:
every held-out shape in shapes.py. Each cell is run on the chip with hostile inputs, judged with the
referee's own check (speedcheck._mismatch: RMS and bf16 ulps), and, if correct, timed against the start
kernel at that shape, interleaved A/B on the device clock (P1's timing.py).

    python heldout_grid.py --op matmul --dry-run        # which kernels and shapes; no chip
    python heldout_grid.py --op matmul                  # writes results_heldout_matmul.json
    python heldout_grid.py --op rmsnorm --kernel mine=path/to/kernel.py

Output, the heldout section of a results file (format at the top of dashboard/build.py):
    {"heldout": [{"kernel": "matmul", "which": "start", "shape": "256x6144x4096",
                  "passed": true, "speedup": 1.0, "time_us": 512.3, "message": ""}, ...]}
Shapes are written MxKxN for matmul and rowsxdim for the rest, as the dashboard's examples are.
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "02-kernel-agent"))
sys.path.insert(0, HERE)

import shapes  # noqa: E402

SEED = 20261010   # fixed, so two runs of the grid see the same hostile inputs


def shape_text(op, shape):
    if op == "matmul":
        K, M, N = shape
        return f"{M}x{K}x{N}"
    return "x".join(str(v) for v in shape)


def best_per_arm(op, root=HERE):
    """Each arm's best verified attempt across every attempts*.jsonl under root: verdict 'faster', highest
    speedup. Returns {arm: record}. Lines that do not parse are skipped, and counted."""
    best, skipped = {}, 0
    for path in sorted(glob.glob(os.path.join(root, "**", "attempts*.jsonl"), recursive=True)):
        for line in open(path, encoding="utf-8", errors="replace"):
            try:
                rec = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if (rec.get("kernel") != op or rec.get("verdict") != "faster" or not rec.get("code")
                    or not isinstance(rec.get("speedup"), (int, float))):
                continue
            arm = rec.get("arm") or "?"
            if arm not in best or rec["speedup"] > best[arm]["speedup"]:
                best[arm] = dict(rec, log=os.path.relpath(path, root))
    if skipped:
        print(f"  ({skipped} log line(s) did not parse and were skipped)")
    return best


def kernel_rows(op, extra):
    """[(which, path, note)] in display order: start, expert, every arm's best, then --kernel extras."""
    rows = [("start", os.path.join(HERE, "kernels", f"{op}_start.py"), "the baseline")]
    if op == "matmul":
        rows.append(("expert", os.path.join(HERE, "kernels", "matmul_expert.py"),
                     "AWS's tutorial kernel, fp32 accumulation"))
        rows.append(("aws as published", os.path.join(HERE, "kernels", "matmul_expert_aws.py"),
                     "AWS's tutorial kernel, bf16 accumulation"))
    out_dir = os.path.join(HERE, "heldout_runs")
    for arm, rec in sorted(best_per_arm(op).items()):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{op}_best_{arm}.py")
        with open(path, "w") as f:
            f.write(rec["code"])
        rows.append((f"best {arm}", path, f"{rec['speedup']:.3f}x on the timing shapes; run {rec.get('run_id')} "
                     f"attempt {rec.get('attempt_no')}, {rec['log']}"))
    for spec in extra:
        which, _, path = spec.partition("=")
        rows.append((which, os.path.abspath(path), "given on the command line"))
    return [r for r in rows if os.path.exists(r[1])]


def run_grid(op, rows, rounds):
    import numpy as np
    import nkibench
    import speedcheck
    import timing

    spec = shapes.OPS[op]
    cells = []
    start_loaded = {}
    for which, path, note in rows:
        print(f"\n{which}: {os.path.relpath(path, HERE)}  ({note})")
        try:
            kernel = nkibench.load_kernel(path, spec["entry"])
        except Exception as e:
            print(f"  cannot load: {type(e).__name__}: {e}")
            for shape in spec["heldout_shapes"]:
                cells.append(dict(kernel=op, which=which, shape=shape_text(op, shape), passed=False,
                                  speedup=None, time_us=None, message=f"cannot load: {e}"[:300]))
            continue
        for shape in spec["heldout_shapes"]:
            label = shape_text(op, shape)
            inp = spec["make_inputs"](shape, SEED, hostile=True)
            want = spec["ref"](inp)
            cell = dict(kernel=op, which=which, shape=label, passed=False, speedup=None, time_us=None,
                        message="")
            try:
                L = timing.Loaded(timing.compile_kernel(kernel, inp), inp)
                got = L.run()[0]
                after = [t.numpy() for t in L.inputs.values()]
                bad = (nkibench.check_inputs_untouched(list(inp.values()), after)
                       or speedcheck._mismatch(got, want, spec["tol"], f"held-out {label}"))
                if bad:
                    cell["message"] = bad.splitlines()[0][:300]
                else:
                    cell["passed"] = True
                    if which == "start":
                        start_loaded[label] = L
                        cell["time_us"] = round(L.time()["median_us"], 1)
                        cell["speedup"] = 1.0
                    elif label in start_loaded:
                        ab = timing.time_ab(start_loaded[label], L, rounds=rounds)
                        cell["time_us"] = round(ab["b"]["median_us"], 1)
                        cell["speedup"] = round(ab["speedup"], 3)
                    else:
                        cell["message"] = "correct; not timed, because the start kernel failed here"
            except Exception as e:
                cell["message"] = f"failed on the chip: {type(e).__name__}: {str(e)[:250]}"
            mark = "pass" if cell["passed"] else "FAIL"
            sp = f"{cell['speedup']:.3f}x" if cell["speedup"] is not None else ""
            tu = f"{cell['time_us']:.1f} us" if cell["time_us"] is not None else ""
            print(f"  {label:<16} {mark}  {tu:>11} {sp:>8}  {cell['message'][:110]}")
            cells.append(cell)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="matmul", choices=sorted(shapes.OPS))
    ap.add_argument("--kernel", action="append", default=[], metavar="WHICH=PATH",
                    help="an extra row, e.g. mine=kernels/my_try.py")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved A/B rounds per timed cell")
    ap.add_argument("--out", help="default: results_heldout_<op>.json next to this file")
    ap.add_argument("--dry-run", action="store_true", help="list rows and shapes; touch no chip")
    a = ap.parse_args()

    rows = kernel_rows(a.op, a.kernel)
    cols = [shape_text(a.op, s) for s in shapes.OPS[a.op]["heldout_shapes"]]
    print(f"{a.op}: {len(rows)} kernels x {len(cols)} held-out shapes, hostile inputs, seed {SEED}")
    print(f"  shapes: {', '.join(cols)}")
    for which, path, note in rows:
        print(f"  row {which:<18} {os.path.relpath(path, HERE)}  ({note})")
    if a.dry_run:
        return 0
    try:
        import nki  # noqa: F401
    except ImportError:
        print("nki is not installed here: run this in the seat pod (or use --dry-run).")
        return 2

    t0 = time.time()
    cells = run_grid(a.op, rows, a.rounds)
    out = a.out or os.path.join(HERE, f"results_heldout_{a.op}.json")
    doc = dict(owner="P2 heldout_grid.py", source="chip", seed=SEED,
               measured=time.strftime("%Y-%m-%d %H:%M:%S"), seat=os.environ.get("CHIPBOOST_SEAT"),
               heldout=cells)
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    fails = sum(1 for c in cells if not c["passed"])
    print(f"\n{len(cells)} cells, {fails} failing, {time.time() - t0:.0f}s. Wrote {os.path.relpath(out, HERE)}"
          f" (the dashboard reads every *results*.json).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
