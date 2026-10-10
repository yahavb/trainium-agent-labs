#!/usr/bin/env python3
"""
holdout_check.py -- HELD-OUT check of the solved/ kernels.

The harness (nkibench.py) tests every kernel on a few friendly shapes: all tile-divisible,
standard_normal inputs, seed 0. This script asks whether the kernels GENERALISE. For each
solved kernel it generates NEW cases, in four groups:

  seeds    the original harness shapes with unseen random seeds
  shapes   unseen but valid shapes (other multiples, other factorisations, other (seq, dim))
  ragged   non-divisible / partial-tile shapes, only where the operation itself allows them
  hostile  values chosen to break numerics (x50/x100 scaling, large means, constant rows,
           large-magnitude matmul operands)

Each case runs through nkibench.simulate_and_count (nki.simulate, the same mechanism the harness
uses) and is compared with the NumPy reference by nkibench.describe_mismatch at the harness
tolerance (2e-2 of the output RMS). Levels 5-7 also report HBM traffic against the byte floor and
against the level's traffic bar on every new shape.

    python holdout_check.py                 # all levels, one subprocess per level, markdown out
    python holdout_check.py --level 10      # one level, printed
    python holdout_check.py --level 10 --json out.json

Nothing in the repo is modified. It needs the Neuron SDK (nki) to do anything useful; run it on
the pod.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nkibench as nb  # noqa: E402

TOL = 2e-2
SOLVED_DIR = os.path.join(HERE, "solved")

# ---------------------------------------------------------------- case generators
#
# Every case is (group, label, args). Args are built exactly the way the harness builds them
# (same dtype, same argument order), only the shapes, seeds and values differ.


def _rng(seed, level_n):
    # Same convention as nkibench.make_inputs so "seed s" means the same thing here.
    return np.random.default_rng(seed + level_n)


HOLDOUT_SEEDS = (1, 7, 123)


def seed_cases(level_n):
    spec = nb.LEVELS[level_n]
    out = []
    for s in HOLDOUT_SEEDS:
        for case in spec["shapes"]:
            args, _ = nb.make_inputs(case, level_n, seed=s)
            out.append(("seeds", f"seed={s} {nb.label(case, level_n)}", args))
    return out


def cases_level1():
    n = 1
    out = seed_cases(n)
    r = _rng(5, n)
    shapes = [((16, 64, 64), 2), ((128, 32, 32), 8), ((3, 48, 48), 4), ((100, 20, 20), 5),
              ((64, 12, 12), 6), ((1, 128, 128), 2), ((128, 64, 64), 2), ((8, 24, 24), 24)]
    for shape, p in shapes:
        out.append(("shapes", f"C,H,W={shape} pool={p}",
                    (r.standard_normal(shape).astype(np.float32), p)))
    # Non-divisible H/W: the reference truncates to the largest multiple of p, so these are valid.
    ragged = [((8, 25, 25), 2), ((32, 30, 30), 4), ((16, 17, 13), 3), ((4, 9, 10), 4)]
    for shape, p in ragged:
        out.append(("ragged", f"C,H,W={shape} pool={p} (H,W not multiples of p)",
                    (r.standard_normal(shape).astype(np.float32), p)))
    # Over the partition limit: C > 128 is a legal input to the reference. A single-tile kernel is
    # not expected to handle it, but the judge's question is exactly whether it does.
    out.append(("ragged", "C,H,W=(256, 16, 16) pool=2 (C > PMAX=128)",
                (r.standard_normal((256, 16, 16)).astype(np.float32), 2)))
    base = r.standard_normal((32, 32, 32)).astype(np.float32)
    out.append(("hostile", "x1000 magnitude", (base * 1000.0, 2)))
    out.append(("hostile", "large mean (+1e4)", (base + 1e4, 2)))
    out.append(("hostile", "constant tensor (all 7.0)", (np.full((32, 32, 32), 7.0, np.float32), 2)))
    out.append(("hostile", "tiny magnitude (x1e-6)", (base * 1e-6, 2)))
    return out


def cases_level2():
    n = 2
    out = seed_cases(n)
    r = _rng(5, n)
    shapes = [((128, 128), (16, 8)), ((64, 256), (16, 16)), ((16, 60), (6, 10)),
              ((100, 30), (2, 15)), ((128, 64), (2, 32)), ((128, 64), (64, 1)),
              ((32, 12), (1, 12)), ((8, 210), (14, 15)), ((128, 512), (32, 16))]
    for shape, s2 in shapes:
        out.append(("shapes", f"shape={shape} as {s2[0]}x{s2[1]}",
                    (r.standard_normal(shape).astype(np.float32), s2)))
    # Ragged does not exist for this operation: the reference asserts F1*F2 == F, so there is no
    # partial tile to speak of. The only axis with a limit is the partition axis.
    out.append(("ragged", "shape=(200, 12) as 3x4 (P > PMAX=128)",
                (r.standard_normal((200, 12)).astype(np.float32), (3, 4))))
    base = r.standard_normal((128, 64)).astype(np.float32)
    out.append(("hostile", "x1e6 magnitude", (base * 1e6, (8, 8))))
    out.append(("hostile", "large mean (+1e4)", (base + 1e4, (8, 8))))
    out.append(("hostile", "constant (all -3.5)", (np.full((128, 64), -3.5, np.float32), (8, 8))))
    return out


def _mm(r, K, M, N, scale=1.0, mean=0.0):
    return (r.standard_normal((K, M)).astype(np.float32) * scale + mean,
            r.standard_normal((K, N)).astype(np.float32) * scale + mean)


def cases_level3():
    n = 3
    out = seed_cases(n)
    r = _rng(5, n)
    # One tile: K <= 128 (partition), M <= 128 (stationary free), N <= 512 (moving free).
    shapes = [(64, 128, 512), (128, 128, 256), (32, 16, 64), (128, 128, 512), (16, 128, 512),
              (128, 8, 8)]
    for K, M, N in shapes:
        out.append(("shapes", f"K={K} M={M} N={N}", _mm(r, K, M, N)))
    # Non-multiples INSIDE a tile are legal for the operation. Multi-tile shapes are not what this
    # level is (it is "single tile" by definition), so they are not tested here; see level 4.
    for K, M, N in [(96, 48, 300), (100, 100, 100), (7, 13, 29), (128, 127, 511)]:
        out.append(("ragged", f"K={K} M={M} N={N} (non-multiples inside one tile)",
                    _mm(r, K, M, N)))
    out.append(("hostile", "x100 magnitude", _mm(r, 128, 64, 512, scale=100.0)))
    out.append(("hostile", "large mean (+50)", _mm(r, 128, 64, 512, mean=50.0)))
    a, b = _mm(r, 128, 64, 512)
    out.append(("hostile", "mixed scale lhsT*1e3 rhs*1e-3", (a * 1e3, b * 1e-3)))
    out.append(("hostile", "constant lhsT (all 2.0)", (np.full((128, 64), 2.0, np.float32), b)))
    return out


def cases_matmul_tiled(n):
    out = seed_cases(n)
    r = _rng(5, n)
    shapes = [(384, 384, 1536), (512, 256, 512), (128, 256, 512), (1024, 128, 512),
              (256, 1024, 1024), (128, 128, 1024)]
    for K, M, N in shapes:
        out.append(("shapes", f"K={K} M={M} N={N}", _mm(r, K, M, N)))
    # The operation allows any K, M, N; the harness only ever uses multiples of 128/512.
    for K, M, N in [(192, 200, 600), (256, 256, 1000), (130, 128, 512), (128, 64, 512)]:
        out.append(("ragged", f"K={K} M={M} N={N} (not multiples of 128/512)", _mm(r, K, M, N)))
    out.append(("hostile", "x100 magnitude K=256 M=256 N=1024", _mm(r, 256, 256, 1024, scale=100.0)))
    out.append(("hostile", "large mean (+50) K=256 M=256 N=1024", _mm(r, 256, 256, 1024, mean=50.0)))
    a, b = _mm(r, 256, 256, 1024)
    out.append(("hostile", "mixed scale lhsT*1e3 rhs*1e-3", (a * 1e3, b * 1e-3)))
    out.append(("hostile", "constant lhsT (all 2.0)", (np.full((256, 256), 2.0, np.float32), b)))
    return out


def _tile(r, seq, dim, scale=3.0, mean=0.0):
    # _args_one_tile scales by 3.0; keep that as the baseline.
    return (r.standard_normal((seq, dim)).astype(np.float32) * scale + mean,)


def cases_level9():
    n = 9
    out = seed_cases(n)
    r = _rng(5, n)
    for seq, dim in [(32, 16), (80, 96), (128, 128), (1, 128), (128, 1), (16, 16), (2, 2)]:
        out.append(("shapes", f"seq={seq} dim={dim}", _tile(r, seq, dim)))
    # seq and dim both ride the partition axis (before / after), so neither may exceed 128; the
    # operation itself allows any size within that, including odd ones.
    for seq, dim in [(17, 33), (100, 7), (127, 127), (3, 101)]:
        out.append(("ragged", f"seq={seq} dim={dim} (odd sizes)", _tile(r, seq, dim)))
    out.append(("hostile", "x100 magnitude", _tile(r, 128, 64, scale=100.0)))
    out.append(("hostile", "large mean (+1e4)", _tile(r, 128, 64, mean=1e4)))
    out.append(("hostile", "constant (all 1.0)", (np.ones((128, 64), np.float32),)))
    return out


def cases_level10():
    n = 10
    out = seed_cases(n)
    r = _rng(5, n)
    for seq, dim in [(32, 16), (80, 96), (128, 128), (128, 512), (16, 2048), (1, 64), (8, 8)]:
        out.append(("shapes", f"seq={seq} dim={dim}", _tile(r, seq, dim)))
    # seq rides the partition axis so seq <= 128; dim is free. Odd sizes are legal.
    for seq, dim in [(17, 33), (100, 7), (128, 1), (127, 129), (5, 1001)]:
        out.append(("ragged", f"seq={seq} dim={dim} (odd sizes)", _tile(r, seq, dim)))
    base = r.standard_normal((128, 64)).astype(np.float32)
    out.append(("hostile", "x50 (values ~ +-150: exp overflows without max-subtraction)",
                (base * 50.0,)))
    out.append(("hostile", "x100 (values ~ +-300)", (base * 100.0,)))
    out.append(("hostile", "x1000 (values ~ +-3000)", (base * 1000.0,)))
    out.append(("hostile", "large mean (+1000)", (base * 3.0 + 1000.0,)))
    out.append(("hostile", "large negative mean (-1000)", (base * 3.0 - 1000.0,)))
    out.append(("hostile", "constant rows (all 5.0): softmax must be uniform",
                (np.full((128, 64), 5.0, np.float32),)))
    out.append(("hostile", "constant rows (all 0.0)", (np.zeros((96, 32), np.float32),)))
    spike = base * 3.0
    spike[:, 0] = 1e4
    out.append(("hostile", "one spike of 1e4 per row (one-hot expected)", (spike,)))
    wide = np.zeros((64, 128), np.float32)
    wide[:, ::2] = -1e4
    out.append(("hostile", "half the row at -1e4 (exp underflow to 0)", (wide,)))
    return out


CASES = {1: cases_level1, 2: cases_level2, 3: cases_level3,
         4: lambda: cases_matmul_tiled(4), 5: lambda: cases_matmul_tiled(5),
         6: lambda: cases_matmul_tiled(6), 7: lambda: cases_matmul_tiled(7),
         9: cases_level9, 10: cases_level10}

RAGGED_NOTE = {
    2: "no ragged shapes exist: the reference asserts F1*F2 == F; only P > PMAX tested",
    3: "single-tile level by definition, so multi-tile shapes are level 4's problem; "
       "non-multiples INSIDE one tile tested",
    9: "seq and dim both ride the partition axis, so both <= 128; odd sizes tested",
    10: "seq rides the partition axis, so seq <= 128; odd sizes tested",
}


# ---------------------------------------------------------------- running

def solved_file(level_n):
    hits = sorted(glob.glob(os.path.join(SOLVED_DIR, f"level{level_n:02d}_*.py")))
    return hits[0] if hits else None


def run_case(kernel, spec, level_n, args):
    """One case -> dict(ok, msg, bytes, floor, waste, seconds)."""
    want = spec["ref"](*args)
    before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    t0 = time.time()
    rec = dict(ok=False, msg=None, bytes=None, floor=None, waste=None, warnings=[])
    try:
        got, counted = nb.simulate_and_count(kernel, args)
    except Exception as e:  # noqa: BLE001 -- a raise IS the finding
        rec["msg"] = f"RAISED during simulation: {type(e).__name__}: {str(e).strip()[:300]}"
        rec["seconds"] = round(time.time() - t0, 1)
        return rec
    rec["seconds"] = round(time.time() - t0, 1)
    rec["warnings"] = counted.get("warnings", [])
    m = nb.describe_mismatch(got, want, TOL)
    if m is None:
        m = nb.check_inputs_untouched(before, args)
    rec["ok"] = m is None
    rec["msg"] = m
    if counted.get("bytes"):
        floor = nb.minimum_hbm_bytes(args, want)
        rec["bytes"] = int(counted["bytes"])
        rec["floor"] = int(floor)
        rec["waste"] = round(counted["bytes"] / floor, 2) if floor else None
        rec["transfers"] = int(counted["transfers"])
    return rec


def run_level(level_n):
    spec = nb.LEVELS[level_n]
    path = solved_file(level_n)
    result = dict(level=level_n, op=spec["op"], file=path, max_waste=spec.get("max_waste"),
                  ragged_note=RAGGED_NOTE.get(level_n), cases=[])
    if path is None:
        result["error"] = "no solved file"
        return result
    try:
        kernel = nb.load_kernel(path, spec["entry"])
    except Exception as e:  # noqa: BLE001
        result["error"] = f"import failed: {type(e).__name__}: {e}"
        return result
    for group, lbl, args in CASES[level_n]():
        rec = run_case(kernel, spec, level_n, args)
        rec.update(group=group, label=lbl)
        result["cases"].append(rec)
        flag = "ok  " if rec["ok"] else "FAIL"
        extra = f"  {rec['waste']:.2f}x floor" if rec.get("waste") else ""
        print(f"  [{flag}] L{level_n:<2} {group:<7} {lbl}{extra}  ({rec['seconds']}s)", flush=True)
        if not rec["ok"]:
            print("         " + (rec["msg"] or "").replace("\n", "\n         "), flush=True)
    return result


# ---------------------------------------------------------------- reporting

GROUPS = ("seeds", "shapes", "ragged", "hostile")


def summarize(results):
    lines = ["# Held-out check of solved/ kernels", "",
             f"Tolerance {TOL:g} of output RMS (nkibench.describe_mismatch), simulated with "
             f"nki.simulate via nkibench.simulate_and_count. Seeds {HOLDOUT_SEEDS} on the harness "
             f"shapes; new shapes, ragged shapes and hostile values as listed per case below.", "",
             "## Pass counts (passed/total)", "",
             "| level | op | " + " | ".join(GROUPS) + " | total |",
             "|---|---|" + "---|" * (len(GROUPS) + 1)]
    for res in results:
        if res.get("error"):
            lines.append(f"| {res['level']} | {res['op']} | " + " | ".join("-" for _ in GROUPS)
                         + f" | ERROR: {res['error']} |")
            continue
        cells, tp, tn = [], 0, 0
        for g in GROUPS:
            cs = [c for c in res["cases"] if c["group"] == g]
            p = sum(c["ok"] for c in cs)
            tp, tn = tp + p, tn + len(cs)
            cells.append(f"{p}/{len(cs)}" if cs else "n/a")
        lines.append(f"| {res['level']} | {res['op']} | " + " | ".join(cells) + f" | {tp}/{tn} |")
    lines += ["", "## Ragged-shape applicability", ""]
    for res in results:
        if res.get("ragged_note"):
            lines.append(f"- level {res['level']}: {res['ragged_note']}")
    lines += ["", "## HBM traffic on the new shapes (levels 4-7)", "",
              "| level | case | bytes | floor | ratio | bar | verdict |", "|---|---|---|---|---|---|---|"]
    for res in results:
        if res["level"] not in (4, 5, 6, 7) or res.get("error"):
            continue
        for c in res["cases"]:
            if c["group"] != "shapes" or not c.get("bytes"):
                continue
            bar = res.get("max_waste")
            verdict = ("n/a (correctness only)" if not bar else
                       ("meets bar" if c["waste"] <= bar else "OVER the bar"))
            if not c["ok"]:
                verdict = "numerics FAILED"
            lines.append(f"| {res['level']} | {c['label']} | {c['bytes']:,} | {c['floor']:,} | "
                         f"{c['waste']:.2f}x | {bar if bar else '-'} | {verdict} |")
    lines += ["", "## Every failure", ""]
    any_fail = False
    for res in results:
        for c in res.get("cases", []):
            if c["ok"]:
                continue
            any_fail = True
            first = (c["msg"] or "").split("\n")
            lines.append(f"- **level {res['level']}** [{c['group']}] {c['label']}:")
            for l in first:
                lines.append(f"    {l}")
    if not any_fail:
        lines.append("(none)")
    lines += ["", "## All cases", ""]
    for res in results:
        lines.append(f"### level {res['level']}: {res['op']} ({os.path.basename(res['file'] or '?')})")
        lines.append("")
        for c in res.get("cases", []):
            tr = f", {c['waste']:.2f}x floor" if c.get("waste") else ""
            lines.append(f"- {'PASS' if c['ok'] else 'FAIL'} [{c['group']}] {c['label']} "
                         f"({c['seconds']}s{tr})")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int)
    ap.add_argument("--json", help="write this level's results as JSON here")
    ap.add_argument("--out", default="holdout_results.md", help="markdown output (all-levels mode)")
    ap.add_argument("--workdir", default=os.path.join(HERE, "holdout_out"))
    ap.add_argument("--timeout", type=int, default=900, help="seconds per level subprocess")
    a = ap.parse_args()

    if a.level:
        res = run_level(a.level)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1, default=str)
        return 0

    os.makedirs(a.workdir, exist_ok=True)
    results = []
    for n in sorted(CASES):
        jpath = os.path.join(a.workdir, f"level{n:02d}.json")
        print(f"=== level {n}", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--level", str(n), "--json", jpath]
        try:
            subprocess.run(cmd, timeout=a.timeout, check=False)
        except subprocess.TimeoutExpired:
            print(f"  level {n}: TIMEOUT after {a.timeout}s", flush=True)
        if os.path.exists(jpath):
            with open(jpath) as f:
                results.append(json.load(f))
        else:
            results.append(dict(level=n, op=nb.LEVELS[n]["op"], file=solved_file(n),
                                error="subprocess produced no result (crash or timeout)",
                                ragged_note=RAGGED_NOTE.get(n), cases=[]))
    md = summarize(results)
    with open(a.out, "w") as f:
        f.write(md)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
