#!/usr/bin/env python3
"""After-the-fact HBM traffic check for level 5-7 solves, on shapes bigger than any the loop or the held-out
set uses. Analysis only: nothing here feeds the agent.

    python scripts/traffic_check.py LOG.jsonl [LOG ...] [--kernel LEVEL:FILE ...] [-o OUT]

For every distinct kernel that scored 1.0 on levels 5-7 in the logs (and every --kernel given directly),
simulate it on each EXTRA shape and report: correct or not, HBM bytes moved as a multiple of the byte floor,
and that multiple against the bars of levels 5, 6 and 7 (1.60x, 1.25x, 1.05x). The held-out set of levels
5-7 is level 4's, whose shapes stay under level 5's bar even for level 4's reference kernel, so it cannot
tell the levels apart; these shapes have several blocks in both M and N, where re-reading shows.
Needs the NKI simulator; set NEURON_PLATFORM_TARGET_OVERRIDE=trn2.
"""
import argparse
import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "projects", "02-kernel-agent"))
import nkibench  # noqa: E402

EXTRA = [dict(K=512, M=512, N=2048), dict(K=256, M=1024, N=1024)]
BARS = {lv: nkibench.LEVELS[lv]["max_waste"] for lv in (5, 6, 7)}


def run(code, level):
    path = nkibench.candidate_path(f"_traffic_L{level}")
    open(path, "w").write(code)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        kernel = nkibench.load_kernel(path, nkibench.LEVELS[level]["entry"])
    out = []
    for case in EXTRA:
        args, _ = nkibench.make_inputs(case, level, seed=7)
        want = nkibench.ref_matmul(*args)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                got, counted = nkibench.simulate_and_count(kernel, args)
        except Exception as e:
            out.append(dict(case=nkibench.label(case, level), error=f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        wrong = nkibench.describe_illegal(counted) or nkibench.describe_mismatch(got, want)
        ratio = counted["bytes"] / nkibench.minimum_hbm_bytes(args, want)
        out.append(dict(case=nkibench.label(case, level), correct=not wrong, why=(wrong or "")[:120],
                        ratio=round(ratio, 3), passes={lv: ratio <= bar for lv, bar in BARS.items()}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--kernel", nargs="*", default=[], help="LEVEL:FILE, checked directly (e.g. controls)")
    ap.add_argument("-o", "--out", default=None, help="write OUT.md and OUT.json")
    a = ap.parse_args()
    items = []                                   # (label, level, code)
    seen = set()
    for p in a.logs:
        for n, line in enumerate(open(p), 1):
            r = json.loads(line)
            if r.get("level") in (5, 6, 7) and r.get("reward", 0) >= 1 - 1e-9:
                sha = hashlib.sha1(r["code"].encode()).hexdigest()[:10]
                if (r["level"], sha) not in seen:
                    seen.add((r["level"], sha))
                    items.append((f"{os.path.basename(p)}:{n} ({sha})", r["level"], r["code"]))
    for spec in a.kernel:
        lv, _, f = spec.partition(":")
        items.append((os.path.basename(f), int(lv), open(f).read()))
    rows = []
    for label, lv, code in items:
        res = run(code, lv)
        rows.append(dict(kernel=label, level=lv, results=res))
        for r in res:
            if "error" in r:
                print(f"L{lv} {label}: {r['case']}: {r['error']}")
                continue
            verdict = "  ".join(f"L{k} {'ok' if v else 'OVER'}" for k, v in r["passes"].items())
            print(f"L{lv} {label}: {r['case']}: {'correct' if r['correct'] else 'WRONG'}, "
                  f"{r['ratio']:.2f}x the floor | {verdict} | its own bar (L{lv} {BARS[lv]:.2f}x): "
                  f"{'met' if r['passes'][lv] else 'NOT met'}")
    if a.out:
        json.dump(rows, open(a.out + ".json", "w"), indent=1)
        md = ["# Traffic on bigger shapes (after the fact, not in the loop)", "",
              f"Extra shapes: {', '.join(nkibench.label(c, 5) for c in EXTRA)}. Bars: L5 {BARS[5]}x, L6 {BARS[6]}x, "
              f"L7 {BARS[7]}x the byte floor.", "",
              "| level | kernel | shape | correct | bytes / floor | L5 | L6 | L7 | its own bar |", "|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            for r in row["results"]:
                if "error" in r:
                    md.append(f"| {row['level']} | {row['kernel']} | {r['case']} | raises | | | | | |")
                    continue
                md.append(f"| {row['level']} | {row['kernel']} | {r['case']} | {'yes' if r['correct'] else '**no**'} | "
                          f"{r['ratio']:.2f}x | " + " | ".join('ok' if r['passes'][k] else 'over' for k in (5, 6, 7))
                          + f" | {'met' if r['passes'][row['level']] else '**not met**'} |")
        open(a.out + ".md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
