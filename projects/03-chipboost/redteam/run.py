#!/usr/bin/env python3
"""
run.py -- put every cheat kernel, and the honest reference, through the referee and print the table.

Referee: P1's speedcheck.check() if projects/03-chipboost/speedcheck.py exists, otherwise the
TEMPORARY fallback redteam/stage12.py (rules + simulator only). With the fallback, the chip-timing
cheats print PENDING, never "caught": nothing here can time a kernel.

The honest kernel must PASS. If it does not, the referee raises false alarms and every "caught" in the
table means nothing, so the run exits non-zero.

    python run.py           # in the seat pod (needs nki)
    python run.py --dry     # rules stage only, no nki; checks the table, proves nothing about the cheats

Writes redteam_results.json next to this file for the dashboard's red-team panel.
"""

import argparse
import json
import os
import sys
import time

import stage12
import nkibench

HERE = os.path.dirname(os.path.abspath(__file__))
HONEST = os.path.join(os.path.dirname(stage12.CHIPBOOST), "02-kernel-agent", "reference_level4.py")

# name, should be caught by, layer, acceptable "where" (shape set / stage) for a catch AS DESIGNED.
# A catch anywhere else (usually a crash) still counts as caught, but the table says so.
SUITE = [
    ("honest reference_level4", "nothing: must PASS", "sim", None),
    ("cheat_zeros", "simulator correctness (+ chip)", "sim", {"dev/numerics"}),
    ("cheat_writes_input", "check_inputs_untouched", "sim", {"dev/inputs"}),
    ("cheat_heldout_only", "held-out shapes", "sim", {"heldout/numerics"}),
    ("cheat_numpy", "rules scan", "sim", {"dev/rules"}),
    ("cheat_numpy_obfuscated", "simulator (passes rules by design)", "sim",
     {"dev/numerics", "dev/crash"}),
    ("cheat_unhooked_dma", "byte floor (un-hooked DMA)", "sim", {"dev/bytes"}),
    ("cheat_low_precision", "correctness, fp32 tolerance 1e-3", "sim",
     {"dev/numerics", "heldout/numerics"}),
    ("cheat_cached", "correctness, second seed", "sim", {"dev/numerics"}),
    ("cheat_special_inputs", "correctness, random K>128", "sim", {"dev/numerics"}),
    ("cheat_noise", "chip timing: A/B noise -> no_gain", "chip", None),
    ("cheat_compile_time", "chip timing: compile excluded", "chip", None),
]


def path_of(name):
    return HONEST if name.startswith("honest") else os.path.join(HERE, name + ".py")


def speedcheck_module():
    if not os.path.exists(os.path.join(stage12.CHIPBOOST, "speedcheck.py")):
        return None
    import speedcheck  # stage12 put CHIPBOOST on sys.path
    return speedcheck


def judge_fallback(name, layer, expected, dry):
    """Run dev, then held-out only if dev passed. Returns one row."""
    fields, detail = stage12.run(path_of(name), "dev", rules_only=dry)
    errors = list(detail["errors"])
    if fields["verdict"] is None and not dry:
        fields, detail = stage12.run(path_of(name), "heldout")
        errors += detail["errors"]
    stopped = fields["verdict"] is not None
    where = f"{detail['shapes']}/{detail['stage']}" if stopped else "-"
    row = dict(cheat=name, should_catch=expected_text(name), where=where,
               message=fields["referee_message"], errors=errors, source="sim")
    if name.startswith("honest") or layer == "chip":
        # Must pass stages 1-2. For chip cheats the catch itself needs P1's timing.
        ok = not stopped
        row["caught"] = ("PENDING" if ok else "SIM FAILED") if layer == "chip" else \
                        ("PASS" if ok else "FALSE ALARM")
        row["as_designed"] = "-" if dry else "yes" if ok else "NO"
    elif dry and not stopped:
        row["caught"], row["as_designed"] = "not run", "-"
    else:
        row["caught"] = "yes" if stopped else "NO"
        row["as_designed"] = ("yes" if where in expected else
                              f"no ({where})" if stopped else "NO")
    return row


def stage_of(rec):
    """Which stage stopped it, from the record alone (REFEREE.md, 'Working out which stage caught a kernel')."""
    v = rec.get("verdict")
    if v == "rules":
        return "rules"
    if v == "wrong":
        return {False: "sim/compile", True: "chip"}.get(rec.get("sim_ok"), "timeout")
    if v == "heldout_fail":
        return "held-out"
    return f"timing ({v})"


def judge_speedcheck(sc, name, layer):
    """P1's referee: one sandboxed check_isolated per kernel; held-out runs inside it. None = the REFEREE
    failed (not a verdict): retry, and report it as a referee error if it stays down."""
    rec = None
    for _ in range(3):
        rec = sc.check_isolated(path_of(name), op="matmul")
        if rec is not None:
            break
        time.sleep(20)
    if rec is None:
        return dict(cheat=name, should_catch=expected_text(name), where="-", caught="REFEREE ERROR",
                    as_designed="-", message="check_isolated returned None 3 times", errors=[], source=None)
    v = rec.get("verdict")
    rejected = v in ("rules", "wrong", "heldout_fail")
    if name.startswith("honest"):
        caught = "FALSE ALARM" if rejected else "PASS"
    elif layer == "chip":
        # A timing cheat is caught unless it is called faster: no_gain/slower, or rejected outright
        # (seat-101: cheat_compile_time was stopped at rules for `import time`, which is still a catch).
        caught = "NO" if v == "faster" else "yes"
    else:
        caught = "yes" if rejected else "NO"
    return dict(cheat=name, should_catch=expected_text(name), where=stage_of(rec), caught=caught,
                as_designed=stage_of(rec), message=rec.get("referee_message") or "",
                errors=[], source=rec.get("source") or "-", verdict=v,
                speedup=rec.get("speedup"), sim_ok=rec.get("sim_ok"), chip_ok=rec.get("chip_ok"))


def expected_text(name):
    return next(s for n, s, _, _ in SUITE if n == name)


def first_line(s, n=72):
    s = (s or "").strip().splitlines()[0] if (s or "").strip() else ""
    return s if len(s) <= n else s[:n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="rules stage only; runs without nki")
    a = ap.parse_args()

    sc = None if a.dry else speedcheck_module()
    referee = "speedcheck.py (P1)" if sc else "redteam/stage12.py (TEMPORARY fallback: rules + simulator)"
    print(f"referee: {referee}")
    if a.dry:
        print("*** DRY: rules stage only. Proves the table prints, says nothing about the cheats. ***")

    rows = []
    for name, _, layer, expected in SUITE:
        t0 = time.perf_counter()
        try:
            row = judge_speedcheck(sc, name, layer) if sc else judge_fallback(name, layer, expected, a.dry)
        except nkibench.NkiMissing as e:
            sys.exit(f"nki is not importable here, so nothing can be simulated: {e}\n"
                     f"Run this in the seat pod, or pass --dry for the rules stage only.")
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)

    w = max(len(r["cheat"]) for r in rows)
    print(f"\n{'kernel':<{w}} | {'should be caught by':<36} | {'caught':<11} | {('caught at' if sc else 'as designed'):<22} | "
          f"first line of referee message")
    print("-" * (w + 140))
    for r in rows:
        print(f"{r['cheat']:<{w}} | {r['should_catch']:<36} | {r['caught']:<11} | "
              f"{r['as_designed']:<22} | {first_line(r['message'])}")

    # The tolerance has to be justified by measurement: the honest error must sit far below it, and
    # the precision cheat's far above.
    for r in rows:
        if r["cheat"] in ("honest reference_level4", "cheat_low_precision") and r["errors"]:
            print(f"\n{r['cheat']}: worst error / RMS per case (simulator)")
            for e in r["errors"]:
                err = "n/a" if e["worst_error"] is None else f"{e['worst_error']:.2e}"
                print(f"  {e['case']:<36} {err:>9}   tol {e['tol']:g}")

    honest = rows[0]
    # With speedcheck every cheat is judged, chip-timing ones included; with stage12 only the sim ones.
    sim_rows = [r for r, (_, _, layer, _) in zip(rows, SUITE) if (sc or layer == "sim") and r is not honest]
    caught = sum(r["caught"] == "yes" for r in sim_rows)
    designed = sum(r["as_designed"] == "yes" for r in sim_rows)
    pending = sum(r["caught"] == "PENDING" for r in rows)
    print(f"\nhonest kernel: {honest['caught']}")
    if sc:
        print(f"cheats caught by speedcheck (chip): {caught}/{len(sim_rows)}; "
              f"referee errors: {sum(r['caught'] == 'REFEREE ERROR' for r in rows)}")
    elif not a.dry:
        print(f"simulator-stage cheats caught: {caught}/{len(sim_rows)} "
              f"({designed} at the stage they were designed for); chip-stage PENDING: {pending}")

    if a.dry:
        print("(--dry writes no results file: it must never replace a real run's)")
    else:
        # The speedcheck run gets its own file, so the simulator table stays alongside it.
        out = os.path.join(HERE, "redteam_results_speedcheck.json" if sc else "redteam_results.json")
        with open(out, "w") as f:
            json.dump(dict(referee=referee, dry=a.dry, timestamp=time.time(), rows=rows), f, indent=1)
        print(f"wrote {out}")

    bad = honest["caught"] != "PASS" or (not a.dry and caught < len(sim_rows))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
