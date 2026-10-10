#!/usr/bin/env python3
"""
diagnose.py -- turn a WRONG matmul into ONE named change, by testing what the kernel actually computed.
Owner: P3.

The referee says what is wrong ("99.9% of elements are outside tolerance"); that alone does not say what
to change, and measured on seat-101 Qwen3 returned the identical wrong kernel four times in a row. So for a
wrong answer this re-runs the kernel on the first failing shape in the simulator and checks its output
against the outputs of known mistakes. If one matches, the instruction names that mistake and its fix. If
none matches, it returns None and the referee's own message stands.

Known mistakes (output of the kernel == output of the mistake):
  first K tile only   lhsT[:128].T @ rhs[:128]             the contraction loop is missing
  last K tile only    lhsT[-128:].T @ rhs[-128:]           each k pass overwrites instead of adding
  one rhs/lhsT tile   (sum of the other's K tiles).T @ one tile   hoisted loads overwrote one variable
  shared PSUM         running sum over output tiles        one accumulator for every (m, n) tile
and combinations of the first two with the third. Only one change is named, the most basic first.

It sits in the agent's feedback path, not in the referee, so it keeps working when speedcheck.py
replaces redteam/stage12.py. Simulator only; needs nki.

    python diagnose.py path/to/kernel.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "02-kernel-agent"))
sys.path.insert(0, os.path.join(HERE, "redteam"))

import nkibench  # noqa: E402
import stage12   # noqa: E402

TK, TM, TN = nkibench.PMAX, nkibench.GEMM_STATIONARY_FMAX, nkibench.GEMM_MOVING_FMAX
MATCH_TOL = 1e-3   # same worst-error/RMS measure and fp32 tolerance as stage12

FIRST_K = ("YOUR KERNEL USES ONLY THE FIRST 128 ROWS OF K: its output equals lhsT[0:128].T @ rhs[0:128], "
           "so it is right when K is 128 and wrong for every larger K. Loop over all K // 128 tiles of the "
           "contraction (for k in nl.affine_range(K // TILE_K)), load the k-th 128-row slice of BOTH lhsT "
           "and rhs inside that loop, and nc_matmul every one into the same PSUM tile so the partial "
           "products add up.")
LAST_K = ("YOUR KERNEL KEEPS ONLY THE LAST 128 ROWS OF K: its output equals the last K tile's product "
          "alone, so each pass of the contraction loop replaces the partial sum instead of adding to it. "
          "Allocate the PSUM tile once per output tile, BEFORE the k loop, and nc_matmul every k tile into "
          "that same PSUM tile.")
ONE_TILE = ("YOUR KERNEL KEEPS ONLY ONE {op} TILE: the loop that loads the {op} tiles writes each one into "
            "the same variable, so only one survives and every k multiplies by that same {op} tile. Keep all "
            "K // 128 {op} tiles: allocate ONE SBUF tensor with room for all of them (partition dimension "
            "128 first), copy the k-th 128-row slice of {op} into slot k, and in the k loop use slot k.")
SHARED_PSUM = ("YOUR OUTPUT TILES ARE RUNNING SUMS OF EACH OTHER: one PSUM accumulator is shared by every "
               "(m, n) output tile, so each tile also contains all the tiles computed before it. Allocate "
               "a fresh PSUM tile inside the n loop, one per output tile, before the k loop.")


def _cumulative_over_tiles(out):
    """What a single never-reset accumulator produces: tile (m, n) = sum of all tiles up to it, in loop
    order (m outer, n inner)."""
    M, N = out.shape
    res, acc = np.empty_like(out), np.zeros((TM, TN), out.dtype)
    for m in range(M // TM):
        for n in range(N // TN):
            acc = acc + out[m * TM:(m + 1) * TM, n * TN:(n + 1) * TN]
            res[m * TM:(m + 1) * TM, n * TN:(n + 1) * TN] = acc
    return res


def candidates(lhsT, rhs):
    """(instruction, output of that mistake), the most basic mistake first."""
    a, b = lhsT.astype(np.float64), rhs.astype(np.float64)
    full = a.T @ b
    out = []
    if a.shape[0] > TK:   # at K == 128 the first and last tile ARE the full product: no evidence either way
        first = a[:TK].T @ b[:TK]
        last = a[-TK:].T @ b[-TK:]
        out += [(FIRST_K, first), (FIRST_K, _cumulative_over_tiles(first)),
                (LAST_K, last), (LAST_K, _cumulative_over_tiles(last))]
        # Measured on seat-101: Qwen3 hoisted the rhs loads into a loop that overwrote one variable, so
        # every k used the same rhs tile. Its output is (sum of lhsT's K tiles).T @ (that one rhs tile).
        a_sum = a.reshape(-1, TK, a.shape[1]).sum(axis=0)
        b_sum = b.reshape(-1, TK, b.shape[1]).sum(axis=0)
        for one in (b[-TK:], b[:TK]):
            out.append((ONE_TILE.format(op="rhs"), a_sum.T @ one))
        for one in (a[-TK:], a[:TK]):
            out.append((ONE_TILE.format(op="lhsT"), one.T @ b_sum))
    if a.shape[1] > TM or b.shape[1] > TN:   # one output tile: a running sum IS the full product
        out.append((SHARED_PSUM, _cumulative_over_tiles(full)))
    return out


def diagnose(path):
    """ONE named change for a wrong matmul kernel, or None if no known mistake matches."""
    try:
        kernel = nkibench.load_kernel(path, nkibench.LEVELS[stage12.LEVEL]["entry"])
    except Exception:
        return None
    ref = nkibench.LEVELS[stage12.LEVEL]["ref"]
    for case in stage12.SHAPES["dev"] + stage12.SHAPES["heldout"]:
        args, _ = nkibench.make_inputs(case, stage12.LEVEL, seed=0)
        want = ref(*args)
        try:
            got, _ = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing:
            raise
        except Exception:
            return None   # a crash is the referee's message to give, not a numerical diagnosis
        if nkibench.describe_mismatch(got, want, stage12.tolerance(args)) is None:
            continue
        for instruction, mistake in candidates(*args):
            err = stage12.worst_error(got, mistake)
            if err is not None and err <= MATCH_TOL:
                return f"On {nkibench.label(case, stage12.LEVEL)}: {instruction}"
        return None       # the first failing shape decides; a guess on a later one would mislead
    return None


if __name__ == "__main__":
    print(diagnose(sys.argv[1]) or "no known mistake matches")
