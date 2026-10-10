#!/usr/bin/env python3
"""
stage12.py -- TEMPORARY FALLBACK referee, stages 1-2 only (rules + simulator). Owner: P3.

P1's speedcheck.py is the real referee. This file exists so the red-team table and the agent loop can
run before it merges, and it is retired the moment speedcheck.py exists (run.py switches on its own).
It has the same signature as the Referee API in TEAM.md and returns only schema.ATTEMPT_FIELDS keys,
so callers do not change when it is swapped out. It never times anything: every result is
source="sim", chip_ok=None, and no verdict is ever slower/no_gain/faster.

Built only from nkibench functions. In order, stopping at the first failure:
  1. rules       check_rules
  2. load once   one module per check, so state kept across calls (a cache) is visible to step 3
  3. per shape, per seed (2 random seeds):
       simulate_and_count       a raise is a failure, labelled RAISED
       check_inputs_untouched
       describe_mismatch        with the dtype tolerance below
       byte floor               counted HBM bytes BELOW minimum_hbm_bytes cannot happen for a correct
                                kernel that moves every byte through nisa.dma_copy, so it means a
                                transfer the counter did not see
       hardware hazard          a simulator warning that the pattern is wrong on hardware

TOLERANCE (worst |error| / RMS of the reference, as in describe_mismatch):
  float32 inputs  1e-3.  An honest fp32 kernel should land around 1e-6 (run.py prints the measured
                  value). Accumulating in bf16 loses up to 2^-9 of the running sum at every K tile:
                  at K=128 that is ~9e-3 on the worst element (~4.5x RMS), which nkibench's default
                  2e-2 would MISS, and at K=4096 (32 roundings) ~2e-2. Estimates, not measurements;
                  the large-K held-out shape is there to make the gap unmistakable.
  bfloat16 inputs 2e-2, nkibench's default, unchanged.
  This is the SIMULATOR tolerance only. On the chip, fp32 matmul may run in a reduced-precision
  mode; P1 sets the chip tolerance.

SHAPES are a temporary fallback until P2's shapes.py: dev = nkibench level 4's shapes, held-out = other
tile multiples (the reference asserts K, M % 128 == 0 and N % 512 == 0), plus K=4096 (Qwen3 hidden).

    python stage12.py path/to/kernel.py [--shapes dev|heldout] [--rules-only]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHIPBOOST = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(CHIPBOOST), "02-kernel-agent"))
sys.path.insert(0, CHIPBOOST)

import nkibench  # noqa: E402
import schema    # noqa: E402

LEVEL = 4         # matmul, tiled: entry nki_matmul_tiled_(lhsT, rhs)
SEEDS = (0, 1)    # two seeds, so a kernel that caches its first answer is wrong on the second

SHAPES = {
    "dev": [dict(s) for s in nkibench.LEVELS[LEVEL]["shapes"]],
    "heldout": [dict(K=384, M=384, N=1536),
                dict(K=640, M=256, N=512),
                dict(K=4096, M=128, N=512)],
}

TOL = {"float32": 1e-3, "bfloat16": 2e-2}
DEFAULT_TOL = 2e-2

# The keys of schema.ATTEMPT_FIELDS the referee owns; the caller fills in the rest.
REFEREE_FIELDS = ("verdict", "referee_message", "instruction_given", "sim_ok", "chip_ok",
                  "time_us_median", "time_us_iqr", "baseline_us_same_session", "speedup", "source")
assert set(REFEREE_FIELDS) <= set(schema.ATTEMPT_FIELDS)


def tolerance(args):
    return TOL.get(str(getattr(args[0], "dtype", "")), DEFAULT_TOL)


def worst_error(got, want):
    """The number describe_mismatch compares with its tolerance, or None if it is not comparable."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if got.shape != want.shape or not np.all(np.isfinite(got)):
        return None
    scale = float(np.sqrt((want ** 2).mean())) or 1.0
    return float(np.abs(got - want).max()) / scale


def _fields(verdict, message):
    return dict(verdict=verdict, referee_message=message,
                # nkibench's messages already end in what to change; the agent sends this back as is.
                instruction_given=message, sim_ok=None if verdict == "rules" else verdict is None,
                chip_ok=None, time_us_median=None, time_us_iqr=None,
                baseline_us_same_session=None, speedup=None, source="sim")


def run(path, shapes="dev", rules_only=False):
    """Returns (fields, detail). fields: schema keys. detail: stage, errors, shape of the failure.

    verdict is None when every stage passed: stages 1-2 cannot say faster or slower.
    """
    detail = dict(stage="pass", shapes=shapes, case=None, errors=[])
    fail_verdict = "heldout_fail" if shapes == "heldout" else "wrong"

    def fail(stage, verdict, message, case=None):
        detail.update(stage=stage, case=case)
        return _fields(verdict, message), detail

    src = open(path).read()
    violations = nkibench.check_rules(src, LEVEL)
    if violations:
        return fail("rules", "rules", "RULE VIOLATIONS: " + " ".join(violations))
    if rules_only:
        detail["stage"] = "rules-only"
        return _fields(None, "rules clean (simulation not run)"), detail

    entry = nkibench.LEVELS[LEVEL]["entry"]
    try:
        kernel = nkibench.load_kernel(path, entry)
    except ModuleNotFoundError as e:
        # A missing SDK is an environment problem, not the kernel's: never turn it into a verdict.
        try:
            import nki  # noqa: F401
        except ImportError:
            raise nkibench.NkiMissing("nki is not importable here; run in the seat pod") from e
        return fail("import", fail_verdict, f"RAISED on import: {type(e).__name__}: {e}. The only "
                    f"imports that exist are nki, nki.language as nl and nki.isa as nisa.")
    except Exception as e:
        return fail("import", fail_verdict, f"RAISED on import: {type(e).__name__}: {e}")

    ref = nkibench.LEVELS[LEVEL]["ref"]
    for case in SHAPES[shapes]:
        lbl = nkibench.label(case, LEVEL)
        for seed in SEEDS:
            where = f"{lbl} seed={seed}"
            args, _ = nkibench.make_inputs(case, LEVEL, seed)
            before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
            want = ref(*args)
            tol = tolerance(args)
            try:
                got, counted = nkibench.simulate_and_count(kernel, args)
            except nkibench.NkiMissing:
                raise
            except Exception as e:
                return fail("crash", fail_verdict,
                            f"RAISED during simulation on {where}: {type(e).__name__}: {e}", where)

            m = nkibench.check_inputs_untouched(before, args)
            if m:
                return fail("inputs", fail_verdict, f"On {where}: {m}", where)

            err = worst_error(got, want)
            detail["errors"].append(dict(case=where, worst_error=err, tol=tol))
            m = nkibench.describe_mismatch(got, want, tol)
            if m:
                return fail("numerics", fail_verdict, f"On {where}: {m}", where)

            floor = nkibench.minimum_hbm_bytes(args, want)
            if counted.get("unmeasured"):
                detail.setdefault("notes", []).append(
                    f"{where}: {counted['unmeasured']} transfers could not be sized; byte floor "
                    f"not checked")
            elif counted["bytes"] < floor:
                return fail("bytes", fail_verdict,
                            f"On {where}: COUNTED HBM TRAFFIC IS BELOW THE FLOOR: "
                            f"{counted['bytes']:,} bytes in {counted['transfers']} transfers, but "
                            f"reading each input once and writing the output once is {floor:,}. "
                            f"Some data moved through a path the counter does not see. Move every "
                            f"HBM transfer with nisa.dma_copy, called by that name.", where)

            hazards = [w for w in counted.get("warnings", []) if "incorrect results on hardware" in w]
            if hazards:
                return fail("hazard", fail_verdict,
                            f"On {where}: CORRECT ON CPU BUT WRONG ON HARDWARE: {hazards[0]}", where)

    return _fields(None, f"stages 1-2 passed on {len(SHAPES[shapes])} {shapes} shapes x "
                         f"{len(SEEDS)} seeds (simulator only)"), detail


def check(path, op="matmul", shapes="dev", baseline=None):
    """The Referee API from TEAM.md. `baseline` is accepted and ignored: nothing is timed here."""
    if op != "matmul":
        raise NotImplementedError("stage12 only knows matmul (level 4); RMSNorm arrives with P2")
    return run(path, shapes)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--shapes", default="dev", choices=sorted(SHAPES))
    ap.add_argument("--rules-only", action="store_true")
    a = ap.parse_args()
    fields, detail = run(a.path, a.shapes, a.rules_only)
    print(f"verdict {fields['verdict'] or 'PASS'}  (stage {detail['stage']}, source sim)")
    print(fields["referee_message"])


if __name__ == "__main__":
    main()
