#!/usr/bin/env python3
"""pick_nki.py: the hand-in NKI kernels, one per level, chosen from every solve in the attempt logs.

    cd <their projects/02-kernel-agent> && NEURON_PLATFORM_TARGET_OVERRIDE=trn2 PYTHONPATH=$PWD:<kit>/agent/nki \\
        python3 pick_nki.py --out <out>/nki_kernels [--compile <out>/compile.txt] <out>/attempts_nki_*.jsonl

Each distinct solve (at most --cap per level) is re-checked, then ranked:
  1. re-graded in the simulator on the level's own shapes (normal values): must pass;
  2. the chip's compiler, from `seat.sh compile`'s compile.txt if given: lowers for trn2 and matches
     in birsim > lowers (birsim off) > not compiled > rejected;
  3. extra cases (hidden_eval.py): the level's shapes with hostile values (x1e3, +1e3 offset, all
     negative, zeros, a constant) and, for levels 1-4, shapes the checker never shows. Only cases
     the reference kernel itself passes count ("fair"): theirs for 1-4, ours (answers/) for 8-14.
Status: VERIFIED (simulator, extra cases and compiler), SIMULATOR ONLY (compiler not run),
DOUBTFUL (fails a fair extra case), REJECTED BY COMPILER. Writes <out>/l<N>.py and kernels.json.
"""
import argparse
import collections
import hashlib
import json
import os
import re

import feedback_v7  # noqa: F401  registers levels 9-14 (ops07, ops08)
import hidden_eval as H
import nkibench

HERE = os.path.dirname(os.path.abspath(__file__))


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return max(0.0, (c - r) / d)


def compile_status(path):
    """sha1[:8] of a kernel's code -> 'match' | 'lowers' | 'wrong numbers' | 'does not lower' | 'does not load'."""
    out, cur, verdicts = {}, None, []

    def close():
        if cur:
            out[cur] = ("does not load" if "load" in verdicts else "does not lower" if "fail" in verdicts else
                        "wrong numbers" if "wrong" in verdicts else "match" if "match" in verdicts else
                        "lowers" if "lowers" in verdicts else "unknown")
    for line in open(path, errors="replace"):
        m = re.match(r"== l\d+_.*_([0-9a-f]{8})\.py", line.strip())
        if m:
            close()
            cur, verdicts = m.group(1), []
        elif cur:
            verdicts += (["load"] if "does not load" in line else ["fail"] if " FAILS:" in line else
                         ["wrong"] if "WRONG NUMBERS" in line else ["match"] if "MATCHES" in line else
                         ["lowers"] if " lowers;" in line else [])
    close()
    return out


def reference(level, theirs):
    p = os.path.join(theirs, f"reference_level{level}.py")
    if not os.path.exists(p):
        p = os.path.join(HERE, "answers", f"ans_level{level}.py")
    return H.load(open(p).read(), level) if os.path.exists(p) else None


def cases(level):
    shown = nkibench.LEVELS[level]["shapes"]
    if level in H.SHAPES:
        return [(s, "normal") for s in shown], H.cases_for(level)
    return [(s, "normal") for s in shown], [(s, v) for s in shown for v in H.VALUES if v != "normal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compile", default="")
    ap.add_argument("--cap", type=int, default=8, help="distinct solves re-checked per level (most frequent first)")
    a = ap.parse_args()
    theirs = os.getcwd()
    comp = compile_status(a.compile) if a.compile and os.path.exists(a.compile) else {}
    solves = collections.defaultdict(collections.Counter)
    where = {}
    for p in a.logs:
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["reward"] >= 1 - 1e-9 and r["code"].strip():
                solves[r["level"]][r["code"]] += 1
                where.setdefault(r["code"], os.path.basename(p))
    os.makedirs(a.out, exist_ok=True)
    rank_c = {"match": 3, "lowers": 2, "unknown": 1, "": 1, "wrong numbers": 0, "does not lower": 0, "does not load": 0}
    picked = {}
    print(f"{'level':>5}  {'solves':>6}  pick")
    for lv in sorted(solves):
        own, extra = cases(lv)
        ref = reference(lv, theirs)
        fair = [c for c in extra if ref is None or not H.run_case(ref, lv, *c)]
        ranked = []
        for code, _ in solves[lv].most_common(a.cap):
            try:
                k = H.load(code, lv)
            except Exception:  # noqa: BLE001
                continue
            if any(H.run_case(k, lv, *c) for c in own):
                continue                                     # does not re-grade: not a solve
            bad = [H.case_label(lv, *c) for c in fair if H.run_case(k, lv, *c)]
            cs = comp.get(hashlib.sha1(code.encode()).hexdigest()[:8], "") if comp else ""
            ranked.append(((rank_c.get(cs, 1), -len(bad), -len(code)), code, cs, bad))
        if not ranked:
            print(f"{lv:>5}  {len(solves[lv]):>6}  none re-grades")
            continue
        ranked.sort(key=lambda t: t[0], reverse=True)
        _, code, cs, bad = ranked[0]
        if cs in ("does not lower", "wrong numbers", "does not load"):
            status = "REJECTED BY COMPILER"
        elif bad:
            status = "DOUBTFUL"
        elif cs in ("match", "lowers"):
            status = "VERIFIED"
        else:
            status = "SIMULATOR ONLY"
        conf = 0.0 if status == "REJECTED BY COMPILER" else round(wilson_lower(len(fair) - len(bad), len(fair)), 3)
        with open(os.path.join(a.out, f"l{lv}.py"), "w") as f:
            f.write(code if code.endswith("\n") else code + "\n")
        checks = f"compiler: {cs or 'not run'}"
        picked[lv] = dict(level=lv, name=nkibench.LEVELS[lv]["op"], status=status, confidence=conf,
                          extra_failed=len(bad), extra_total=len(fair), near_zero_only=0, tile=checks, strict=[],
                          from_log=where[code], candidates=len(solves[lv]), usable=len(ranked),
                          first_extra_failure=bad[0] if bad else None)
        print(f"{lv:>5}  {len(solves[lv]):>6}  {status} conf {conf:.2f}, extra failed {len(bad)}/{len(fair)}, "
              f"{checks}  <- {where[code]}")
    with open(os.path.join(a.out, "kernels.json"), "w") as f:
        json.dump(picked, f, indent=1)
    print(f"\n{len(picked)} levels with a kernel -> {a.out}")


if __name__ == "__main__":
    main()
