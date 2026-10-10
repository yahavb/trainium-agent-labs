#!/usr/bin/env python3
"""
selftest.py — proves the harness before anything trusts a score.

The challenge says to build this FIRST and to break a correct kernel ON PURPOSE: "If the
harness can't catch your own deliberate bug, it will not catch the model's." Every check
below plants a specific bug and requires (a) that it is caught and (b) that the
INSTRUCTION produced actually names a change -- an instruction that does not contain a
verb phrase is a verdict, and a verdict reproduces the violation (measured twice in
project 02).

    python selftest.py
"""

import context as ctx_mod
import errors
import ladder
import numpy as np
import tools
import verifier

PASS, FAIL = "ok", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {name:<46} -> {PASS if ok else FAIL}{'  ' + detail if detail and not ok else ''}")


GOOD_SUM = """
import numpy as np
def kernel(x):
    R, C = x.shape
    out = np.zeros(R, dtype=np.float64)
    for r0 in range(0, R, 128):
        for c0 in range(0, C, 512):
            t = x[r0:r0+128, c0:c0+512]
            out[r0:r0+128] += t.sum(axis=1)
    return out
"""


def instruction_ok(instr):
    return bool(instr) and 20 <= len(instr) <= 500


def main():
    print("proving the harness catches planted bugs (the challenge's first build step)\n")

    # --- a correct kernel passes --------------------------------------------
    r = verifier.check(GOOD_SUM, 2)
    check("correct level-2 kernel passes", r["solved"], str(r))

    # --- planted bug: ragged edge -------------------------------------------
    bad = GOOD_SUM.replace("out[r0:r0+128] += t.sum(axis=1)",
                           "out[r0:r0+128] += t.sum(axis=1) if t.shape[1] == 512 else 0")
    r = verifier.check(bad, 2)
    check("ragged-edge bug caught", not r["solved"] and r["taxonomy"] == "ragged-edge",
          r["taxonomy"] or "missed")
    check("  its instruction names a change",
          instruction_ok(r["failures"][0]["instruction"]),
          r["failures"][0]["instruction"][:80])

    # --- planted bug: whole-array cheat --------------------------------------
    r = verifier.check("import numpy as np\ndef kernel(x):\n    return x.sum(axis=1)\n", 2)
    check("whole-input .sum() cheat caught", r["taxonomy"] == "banned-call", str(r["taxonomy"]))
    r = verifier.check("import numpy as np\ndef kernel(x):\n    return np.sum(x, axis=1)\n", 2)
    check("whole-input np.sum cheat caught", r["taxonomy"] == "banned-call", str(r["taxonomy"]))
    # ... while the same call on a TILE SLICE stays legal (rejecting a correct kernel is
    # worse than missing a cheat -- project 02's measured rule)
    ok = verifier.check_rules(GOOD_SUM, 2)[0] == []
    check("tile-slice .sum() stays legal", ok)

    # --- planted bug: boolean-mask indexing ----------------------------------
    r = verifier.check("import numpy as np\ndef kernel(x):\n    return x[x > 0].sum()\n", 2)
    check("boolean-mask indexing caught", r["taxonomy"] == "fancy-indexing", str(r["taxonomy"]))

    # --- planted bug: whole-array arithmetic ---------------------------------
    r = verifier.check("import numpy as np\ndef kernel(x, a, b):\n"
                       "    return np.maximum(a * x + b, 0.0)\n", 1)
    check("whole-array arithmetic caught", r["taxonomy"] == "whole-array-op", str(r["taxonomy"]))

    # --- planted bug: no tile loop where one is required ----------------------
    r = verifier.check("import numpy as np\ndef kernel(x):\n    out = np.zeros(x.shape[0])\n"
                       "    for c in range(x.shape[1]):\n        out += x[:, c]\n    return out\n", 2)
    v, tax = verifier.check_rules(
        "import numpy as np\ndef kernel(x):\n    return np.array([np.sum(x[i]) for i in range(len(x))])\n", 2)
    check("loopless kernel caught statically", any("no explicit loop" in s for s in v))

    # --- planted bug: softmax without max subtraction -------------------------
    nosub = """
import numpy as np
def kernel(x):
    R, C = x.shape
    out = np.zeros((R, C), dtype=np.float64)
    for r0 in range(0, R, 128):
        t = x[r0:r0+128]
        e = np.exp(t)
        out[r0:r0+128] = e / e.sum(axis=1, keepdims=True)
    return out
"""
    r = verifier.check(nosub, 5)
    check("overflowing softmax caught (NaN)", r["taxonomy"] == "non-finite", str(r["taxonomy"]))
    check("  and the good softmax passes", verifier.check(
        nosub.replace("e = np.exp(t)",
                      "e = np.exp(t.astype(np.float64) - t.astype(np.float64).max(axis=1, keepdims=True))"),
        5)["solved"])

    # --- planted bug: input mutation ------------------------------------------
    mut = GOOD_SUM.replace("out[r0:r0+128] += t.sum(axis=1)",
                           "out[r0:r0+128] += t.sum(axis=1); x[r0, 0] = -999.0")
    r = verifier.check(mut, 2)
    check("input mutation caught", r["taxonomy"] == "modified-input", str(r["taxonomy"]))

    # --- planted bug: per-tile statistic (the stat-scope trap, measured on level 4) --
    # RMS normalised per column tile: wherever the reference's whole-row RMS differs
    # from the tile-local one, and visibly wherever a tile is 1 column wide (x/|x| = 1).
    perscope = """
import numpy as np
def kernel(x, eps=1e-6):
    R, C = x.shape
    out = np.zeros((R, C), dtype=np.float64)
    for r0 in range(0, R, 128):
        for c0 in range(0, C, 512):
            t = x[r0:r0+128, c0:c0+512].astype(np.float64)
            rms = np.sqrt((t * t).mean(axis=1, keepdims=True)) + eps
            out[r0:r0+128, c0:c0+512] = t / rms
    return out
"""
    r = verifier.check(perscope, 4)
    check("per-tile RMS scope caught", r["taxonomy"] == "stat-scope", str(r["taxonomy"]))
    check("  its instruction names the two-pass restructure",
          "two passes" in r["failures"][0]["instruction"],
          r["failures"][0]["instruction"][:80])

    # --- planted bug: per-tile softmax (the L5 face of stat-scope, from the live run) --
    # a one-wide final column tile softmaxes to exactly 1.0 -- plausible numbers, wrong
    # row denominators.
    pertile_sm = """
import numpy as np
def kernel(x):
    R, C = x.shape
    out = np.zeros((R, C), dtype=np.float64)
    for i in range(0, R, 128):
        for j in range(0, C, 512):
            tile = x[i:i+128, j:j+512]
            tile_max = tile.max(axis=-1, keepdims=True)
            tile_exp = np.exp((tile - tile_max).astype(np.float64))
            out[i:i+128, j:j+512] = tile_exp / tile_exp.sum(axis=-1, keepdims=True)
    return out
"""
    r = verifier.check(pertile_sm, 5)
    check("per-tile softmax scope caught", r["taxonomy"] == "stat-scope",
          str(r["taxonomy"]))
    check("  softmax instruction names two passes",
          "two passes" in r["failures"][0]["instruction"],
          r["failures"][0]["instruction"][:90])

    # --- planted bug: per-tile ASSIGN overwriting a per-row reduction (L2, live run 2) --
    # each column tile's partial sum overwrites the last: the output holds only the
    # final tile's contribution.
    overwrite = """
import numpy as np
def kernel(x):
    n_rows, n_cols = x.shape
    result = np.empty((n_rows,), dtype=x.dtype)
    for start_row in range(0, n_rows, 128):
        end_row = min(start_row + 128, n_rows)
        for start_col in range(0, n_cols, 512):
            end_col = min(start_col + 512, n_cols)
            tile = x[start_row:end_row, start_col:end_col]
            result[start_row:end_row] = tile.sum(axis=1)
    return result
"""
    r = verifier.check(overwrite, 2)
    check("per-tile assign overwrite caught", r["taxonomy"] == "stat-scope",
          str(r["taxonomy"]))
    check("  overwrite instruction says accumulate",
          "combine" in r["failures"][0]["instruction"],
          r["failures"][0]["instruction"][:90])

    # --- planted bug: one-pass variance cancellation (the level-8/12 trap) ----
    onepass = """
import numpy as np
def kernel(x):
    R, C = x.shape
    out = np.zeros(R, dtype=np.float64)
    for r0 in range(0, R, 128):
        t = x[r0:r0+128].astype(np.float32)
        out[r0:r0+128] = (t * t).mean(axis=1) - (t.mean(axis=1) ** 2)
    return out
"""
    r = verifier.check(onepass, 12)
    check("E[x^2]-E[x]^2 cancellation caught", not r["solved"], str(r["taxonomy"]))

    # --- planted bug: cumsum carry lost at a column-tile boundary (holdout 11) ----
    # A row-wise cumsum is independent per ROW; the state that must cross a boundary is
    # the row's running total between COLUMN tiles. Restarting the cumsum per column
    # tile loses it.
    nocarry = """
import numpy as np
def kernel(x):
    R, C = x.shape
    out = np.zeros((R, C), dtype=np.float64)
    for r0 in range(0, R, 128):
        for c0 in range(0, C, 512):
            t = x[r0:r0+128, c0:c0+512]
            out[r0:r0+128, c0:c0+512] = np.cumsum(t, axis=1)
    return out
"""
    r = verifier.check(nocarry, 11)
    check("lost cumsum carry caught (holdout)", not r["solved"], str(r["taxonomy"]))
    carry = """
import numpy as np
def kernel(x):
    R, C = x.shape
    out = np.zeros((R, C), dtype=np.float64)
    for r0 in range(0, R, 128):
        prev = np.zeros((min(128, R - r0), 1))
        for c0 in range(0, C, 512):
            t = x[r0:r0+128, c0:c0+512].astype(np.float64)
            out[r0:r0+128, c0:c0+512] = np.cumsum(t, axis=1) + prev
            prev = out[r0:r0+128, c0:c0+512][:, -1:]
    return out
"""
    r = verifier.check(carry, 11)
    check("correct carry version passes (holdout)", r["solved"], str(r["failures"][:1]))

    # --- planted bug: wrong signature ------------------------------------------
    r = verifier.check("import numpy as np\ndef kernel(x):\n    return x\n", 1)
    check("wrong signature caught before running", r["taxonomy"] == "parse-error",
          str(r["taxonomy"]))

    # --- nondeterminism ---------------------------------------------------------
    nd = ("import numpy as np\n_state = [0.0]\ndef kernel(x):\n    R, C = x.shape\n"
          "    out = np.zeros(R, dtype=np.float64)\n"
          "    for r0 in range(0, R, 128):\n        out[r0:r0+128] = x[r0:r0+128].sum(axis=1) + _state[0]\n"
          "    _state[0] += 1.0\n    return out\n")
    r = verifier.check(nd, 2)
    check("nondeterministic kernel caught", r["taxonomy"] == "nondeterministic",
          str(r["taxonomy"]))

    # --- the tool: REPL ---------------------------------------------------------
    s = tools.Scratch()
    s.run("x = np.arange(6).reshape(2, 3)")
    check("REPL state persists", s.run("x.shape") == "(2, 3)", s.run("x.shape"))
    check("REPL rejects import", "rejected" in s.run("import os"))
    check("REPL rejects open", "rejected" in s.run("open('/etc/passwd')"))
    check("REPL returns the exception", "ZeroDivisionError" in s.run("1/0"))

    # --- the tool: docs retrieval ------------------------------------------------
    d = tools.Docs("docs")
    hit = d.search("softmax overflow exp")
    check("docs finds overflow guidance", "overflow" in hit.lower() and "max" in hit.lower())
    hit = d.search("nki dma_copy psum matmul")
    check("docs finds NKI matmul sequence", "nc_matmul" in hit and "psum" in hit.lower())
    hit = d.search("argmax ties")
    check("docs finds tie rule", "first" in hit.lower())

    # --- auto-compaction -----------------------------------------------------------
    c = ctx_mod.Context(150)
    c.set("reference", "ref " * 200, floor=-1)     # 200 est tokens, protected
    c.set("ledger", "\n".join(f"- line {i}: failure {i}" for i in range(30)))
    c.set("scratch", "\n".join(f"SCRATCH: q{i}" for i in range(30)))
    parts, events = c.build()
    check("compaction engages under budget", len(events) > 0)
    check("  reference survives", "reference" in parts)
    check("  budget respected", c.used() <= 150 + 60, f"used={c.used()}")
    check("  events logged with sizes", all("before_tokens" in e for e in events))

    # --- taxonomy: verdict -> instruction --------------------------------------------
    for tax, verdict in [("ragged-edge", "wrong in the final partial tile"),
                         ("non-finite", "NaN at index (3, 0)"),
                         ("partial-coverage", "OUTPUT IS 100% ZEROS")]:
        instr = errors.instruction_from_failure(dict(taxonomy=tax, verdict=verdict), "case")
        check(f"instruction for {tax} names a change", instruction_ok(instr),
              instr[:60])
    check("classify_verdict maps ragged edge",
          errors.classify_verdict("this is in the final partial ROW tile -- ragged edge") == "ragged-edge")

    # --- the ladder's own references are right ------------------------------------
    ok = True
    for n in ladder.LEVELS:
        for lbl, args in ladder.LEVELS[n]["build"]():
            out = ladder.LEVELS[n]["ref"](*args)
            if out is None:
                ok = False
    check("every reference runs on every case", ok)

    n_fail = sum(1 for _n, ok, _d in results if not ok)
    print(f"\nSELFTEST {'PASSED' if n_fail == 0 else 'FAILED'} "
          f"({len(results) - n_fail}/{len(results)} checks)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
