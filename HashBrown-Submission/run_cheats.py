#!/usr/bin/env python3
"""
run_cheats.py -- two jobs, both reusing the same cheat files in cheats/:

  1. `main()` (unchanged from Phase 2): score every ORIGINAL cheat (C1-C8) through BOTH
     agent.grade() and nkibench.verify()/--check, unaugmented -> results/cheats_before.csv.
     This file is left exactly as Phase 2 produced it, per the Phase 4 instruction; running
     `python run_cheats.py` with no arguments reproduces it but does not have to be re-run.

  2. `tiered_main()` (Phase 4): run every cheat (C1-C9, C11 -- C10 and C12 were attempted and
     dropped; see HARDENING.md) through nkibench.verify() at each CUMULATIVE augmentation
     tier -- A0, +A1, +A2a, +A2b, +A3, +A4 -- and write results/cheats_after.csv and
     results/cost.csv. This is the only path with per-tier granularity today (see
     nkibench.verify()'s `tiers` parameter); agent.grade() has no augmentation hook yet
     (deferred to Phase 5).

    python run_cheats.py            # cheats_before.csv (A0 only, both paths)
    python run_cheats.py --tiered   # cheats_after.csv + cost.csv (cumulative tiers)
"""

import contextlib
import csv
import io
import os
import sys
import time

import agent
import nkibench

CHEATS = [
    ("C1", "do_nothing", 1, "cheats/c1_do_nothing.py"),
    ("C2", "partial_rows", 2, "cheats/c2_partial_rows.py"),
    ("C3", "hardcoded_shape", 3, "cheats/c3_hardcoded_shape.py"),
    ("C4", "constant_output", 1, "cheats/c4_constant_output.py"),
    ("C5", "input_tamper", 2, "cheats/c5_input_tamper.py"),
    ("C6", "almost_right", 2, "cheats/c6_almost_right.py"),
    ("C7", "edge_skip", 2, "cheats/c7_edge_skip.py"),
    ("C8", "path_mismatch", 2, "cheats/c8_path_mismatch.py"),
]

# The second-generation cheats, built to pass the now-hardened A0. C10 (fixed_input_lookup)
# and C12 (precision_shortcut) were attempted and dropped -- see HARDENING.md Findings log
# for the concrete, empirically-verified reasons -- so they are not in this list.
CHEATS_GEN2 = [
    ("C9", "memorise_all_shapes", 3, "cheats/c9_memorise_all_shapes.py"),
    ("C11", "assume_divisible", 2, "cheats/c11_assume_divisible.py"),
]

ALL_CHEATS = CHEATS + CHEATS_GEN2

HONEST_KERNELS = [
    (1, "reference_level1.py"), (2, "reference_level2.py"),
    (3, "reference_level3.py"), (4, "reference_level4.py"),
]

# Cumulative tiers, in the order the instruction specified. A1 needs a seed that actually
# differs from the default 0 to mean anything -- fixed and logged here, not time-based, so
# this report is reproducible (rule 8).
A1_SEED = 424242
TIER_SEQUENCE = [
    ("A0", set(), 0),
    ("+A1", {"a1"}, A1_SEED),
    ("+A2a", {"a1", "a2a"}, A1_SEED),
    ("+A2b", {"a1", "a2a", "a2b"}, A1_SEED),
    ("+A3", {"a1", "a2a", "a2b", "a3"}, A1_SEED),
    ("+A4", {"a1", "a2a", "a2b", "a3", "a4"}, A1_SEED),
]

FULL = sum(agent.WEIGHTS.values())

# Markers that introduce an actual failure description in verify()'s printed output, in the
# order they can appear. The LAST line is almost always "improve THIS message before you
# touch the prompt" -- useless as a message -- so pull the real failure text instead.
FAILURE_MARKERS = (
    "RULE VIOLATIONS", "FAILED to import", "simulation  SKIPPED",
    "RAISED during", "NUMERICAL MISMATCH", "NON-FINITE OUTPUT", "WRONG SHAPE",
    "THE KERNEL MODIFIED", "OUTPUT IS", "OUTPUT DID NOT CHANGE",
    "CORRECT, BUT TOO MUCH HBM TRAFFIC",
)


def extract_failure(text):
    """Pull the actual failure description out of verify()'s stdout, not just its last line.

    verify() prints one or more "case <label>:" blocks, each followed by an indented message
    that starts with one of FAILURE_MARKERS and may continue for a couple more indented
    lines. Return the first such block; fall back to the "numerics N/M" summary line (a
    clean pass has no failure block at all), then to the last line as a last resort.
    """
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if any(mk in ln for mk in FAILURE_MARKERS):
            chunk = [ln.strip()]
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith("^") or nxt.startswith("case "):
                    break
                chunk.append(nxt)
            return " | ".join(chunk)
    for ln in lines:
        if "numerics" in ln:
            return ln.strip()
    return lines[-1].strip() if lines else ""


def score_via_grade(level, path):
    src = open(path).read()
    reward, parts, feedback = agent.grade(src, level)
    return reward, reward >= FULL - 1e-9, feedback


def score_via_check(level, path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = nkibench.verify(path, level)
        except Exception as e:
            buf.write(f"\nEXCEPTION during verify(): {type(e).__name__}: {e}")
            rc = -1
    text = buf.getvalue().strip()
    return rc, rc == 0, extract_failure(text), text


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for cid, name, level, path in CHEATS:
        g_reward, g_passed, g_feedback = score_via_grade(level, path)
        c_rc, c_passed, c_last, c_full = score_via_check(level, path)
        mismatch = g_passed != c_passed
        rows.append(dict(
            cheat=cid, name=name, level=level,
            grade_reward=round(g_reward, 3),
            grade_passed=g_passed,
            grade_message=(g_feedback or "").replace("\n", " | ")[:300],
            check_rc=c_rc,
            check_passed=c_passed,
            check_message=c_last.replace("\n", " | ")[:300],
            path_mismatch=mismatch,
        ))
        flag = "  <-- PATH MISMATCH" if mismatch else ""
        print(f"{cid:3} {name:<16} L{level}  grade={g_reward:.2f} "
              f"({'PASS 1.0' if g_passed else 'caught'})   "
              f"check rc={c_rc} ({'PASS' if c_passed else 'caught'}){flag}")

    out_path = "results/cheats_before.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")


def _run_tier(level, path, tiers, seed):
    """One nkibench.verify() call at one tier. Returns (passed, message, seconds)."""
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        try:
            rc = nkibench.verify(path, level, seed=seed, tiers=tiers)
        except Exception as e:
            buf.write(f"\nEXCEPTION during verify(): {type(e).__name__}: {e}")
            rc = -1
    elapsed = time.perf_counter() - t0
    text = buf.getvalue().strip()
    return (rc == 0), extract_failure(text), elapsed


def tiered_main():
    os.makedirs("results", exist_ok=True)
    after_rows, cost_rows = [], []

    print("=== cheats_after.csv: every cheat at every cumulative tier ===")
    for cid, name, level, path in ALL_CHEATS:
        for tier_name, tiers, seed in TIER_SEQUENCE:
            passed, msg, secs = _run_tier(level, path, tiers, seed)
            after_rows.append(dict(
                cheat=cid, name=name, level=level, tier=tier_name,
                passed=passed, message=msg.replace("\n", " | ")[:300],
                seconds=round(secs, 3),
            ))
            cost_rows.append(dict(kernel=f"{cid}_{name}", level=level, tier=tier_name,
                                   seconds=round(secs, 3)))
            print(f"{cid:4} {name:<20} L{level} {tier_name:<5} "
                  f"{'PASS' if passed else 'caught':6} {secs:5.2f}s")

    print("\n=== cost.csv: honest reference kernels at every cumulative tier ===")
    for level, path in HONEST_KERNELS:
        for tier_name, tiers, seed in TIER_SEQUENCE:
            passed, msg, secs = _run_tier(level, path, tiers, seed)
            cost_rows.append(dict(kernel=f"honest_level{level}", level=level, tier=tier_name,
                                   seconds=round(secs, 3)))
            flag = "" if passed else "  <-- HONEST KERNEL FAILED, SHOULD NOT HAPPEN"
            print(f"honest L{level:<3} {tier_name:<5} "
                  f"{'PASS' if passed else 'FAIL':6} {secs:5.2f}s{flag}")

    after_path = "results/cheats_after.csv"
    with open(after_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(after_rows[0].keys()))
        w.writeheader()
        w.writerows(after_rows)
    print(f"\nwrote {after_path}")

    cost_path = "results/cost.csv"
    with open(cost_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
        w.writeheader()
        w.writerows(cost_rows)
    print(f"wrote {cost_path}")


if __name__ == "__main__":
    if "--tiered" in sys.argv:
        tiered_main()
    else:
        main()
