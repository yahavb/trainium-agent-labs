#!/usr/bin/env python3
"""
ladder.py — the operation ladder for the v2 kernel-agent harness.

Ten levels ported from projects/02-kernel-agent/kernelbench.py (the challenge's Stage A:
NumPy under kernel-shaped rules), plus three SELF-HOLDOUT levels (11-13) in the same
distribution. The judges hold back three of their own; these three exist so the agent can
be tested on levels it was NOT tuned on. They are ours, not theirs.

What v2 adds over kernelbench.py's ladder:

  * per-level tolerance WITH a stated justification (the challenge demands one; a single
    global 1e-4 with no reasoning is exactly the "it looked close" answer it warns about)
  * array_args per level -- which parameters are ndarrays. The static rule check needs this
    to detect whole-array arithmetic on the input (a broadcast trick) without flagging the
    scalar coefficients.
  * a self-holdout tier, so generalization is measurable rather than claimed.

A candidate is a Python file defining `kernel(...)`, same signature as the reference.

    python ladder.py --list
    python ladder.py --level 5 --show
"""

import inspect
import textwrap

import numpy as np

TILE_ROWS, TILE_COLS = 128, 512

# Functions nobody may call, in any level: they either do the whole job or hide the tiling.
# v2 adds the broadcast/outer-product family -- kernelbench.py did not ban these, and
# np.outer(a_slice, b_slice) is exactly the "broadcasting trick" the challenge outlaws.
BANNED_GLOBAL = {
    "einsum", "take", "take_along_axis", "put_along_axis", "choose", "select",
    "apply_along_axis", "apply_over_axes", "vectorize", "fromfunction",
    "argsort", "sort", "lexsort", "partition", "argpartition", "compress",
    "extract", "place", "putmask", "nditer", "ndenumerate",
    "outer", "inner", "ix_", "meshgrid", "broadcast_to", "broadcast_arrays",
    "add.outer", "multiply.outer", "subtract.outer", "maximum.outer", "minimum.outer",
    "ufunc.outer", "as_strided", "sliding_window_view",
}


def _rng(seed):
    return np.random.default_rng(seed)


def cases_2d(seed, prime=True):
    """Hostile test shapes: partial tiles, prime dims, a dimension of 1, and five value
    regimes per shape (shape battery from kernelbench.py, attributed; the 'high mean'
    regime is v2's -- the challenge asks for 'large means (layernorm cancellation)' and a
    large STANDARD DEVIATION does not produce cancellation, a large mean with small
    variance does). Every regime breaks a different class of quietly-wrong kernel."""
    r = _rng(seed)
    shapes = [(8, 16), (128, 512), (129, 513), (1, 7), (7, 1), (128, 1)]
    if prime:
        shapes += [(31, 127), (257, 61)]
    out = []
    for sh in shapes:
        out.append(("normal", r.standard_normal(sh).astype(np.float32)))
        out.append(("large", r.standard_normal(sh).astype(np.float32) * 1e3 + 1e4))
        out.append(("high mean", r.standard_normal(sh).astype(np.float32) + 1e4))
        out.append(("identical rows", np.tile(r.standard_normal((1, sh[1]))
                                              .astype(np.float32), (sh[0], 1))))
        out.append(("zeros", np.zeros(sh, np.float32)))
    return out


LEVELS = {}


def level(n, name, trap, ref, build, ban, rtol, tol_why, array_args, holdout=False):
    """Register one rung.

    rtol    -- worst |got-want| allowed, as a fraction of the reference output's RMS
               (plus a small absolute floor for all-zero outputs). Stated per level.
    tol_why -- WHY this number. The challenge: "'rtol=1e-5' is an answer; 'it looked
               close' is not."
    array_args -- indices of the reference's ndarray parameters, for the whole-array
               arithmetic check.
    holdout -- True for the self-holdout tier: never tuned against, reported separately.
    """
    LEVELS[n] = dict(n=n, name=name, trap=trap, ref=ref, build=build, ban=ban,
                     rtol=rtol, tol_why=tol_why, array_args=array_args, holdout=holdout)


# ------------------------------------------------------------------ the graded ladder

level(1, "elementwise relu(a*x+b)", "none — the 'the loop works' checkpoint",
      lambda x, a, b: np.maximum(a * x + b, 0.0),
      lambda: [(lbl, (x, 2.5, -0.5)) for lbl, x in cases_2d(1)],
      ban=set(), rtol=1e-5,
      tol_why="per-element mul/add in float32 rounds at ~1e-7 relative; anything above 1e-5 "
              "of the output RMS means real logic error, not rounding.",
      array_args=(0,))

level(2, "row-wise sum", "the partial final tile",
      lambda x: x.sum(axis=1),
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(2)],
      ban={"sum", "cumsum", "nansum", "trace", "add.reduce"},
      rtol=1e-3,
      tol_why="a naive float32 accumulation over <=513 elements drifts up to ~n*eps "
              "(~6e-5 relative on the 'large' rows); 1e-3 keeps 15x headroom for order "
              "differences while still catching a wrong tile or a dropped tail.",
      array_args=(0,))

level(3, "row-wise max", "partial tile, and the identity is -inf not 0",
      lambda x: x.max(axis=1),
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(3)],
      ban={"max", "amax", "nanmax", "ptp", "maximum.reduce"},
      rtol=0.0,
      tol_why="max is order-independent selection: a correct kernel returns the exact "
              "value. Tolerance here would hide an off-by-one over the final tile, so "
              "the check is exact up to a 1e-6 absolute floor.",
      array_args=(0,))

def _ref_4(x, eps=1e-6):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)


level(4, "RMSNorm over last axis", "accumulation order and dtype",
      _ref_4,
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(4)],
      ban={"norm", "mean", "average", "sum"},
      rtol=1e-3,
      tol_why="the reference accumulates in float64; a float32 kernel over <=513 elements "
              "drifts ~1e-5 relative before the sqrt, and the sqrt halves it. 1e-3 is 100x "
              "headroom over drift and 1000x under the wrong-answer scale.",
      array_args=(0,))


def _ref_5(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp((x - m).astype(np.float64))
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


level(5, "softmax over last axis", "naive exp OVERFLOWS; needs max subtraction",
      _ref_5,
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(5)],
      ban={"softmax", "logsumexp", "sum", "max", "amax"},
      rtol=1e-3,
      tol_why="probabilities sum to 1 and the reference works in float64; float32 kernels "
              "land within ~1e-6 of that. 1e-3 passes every correct implementation and "
              "fails every skipped-max-subtraction one, which returns NaN on the 'large' "
              "rows (exp of +thousands).",
      array_args=(0,))

level(6, "tiled transpose", "tile boundaries on non-square non-divisible shapes",
      lambda x: x.T.copy(),
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(6)],
      ban={"transpose", "swapaxes", "moveaxis", "rollaxis", "permute", "matmul", "dot"},
      rtol=0.0,
      tol_why="a transpose is a permutation: values move, none change. Exact up to a "
              "1e-6 absolute floor.",
      array_args=(0,))

def _build_7():
    r = _rng(7)
    out = []
    for m, k, n in [(8, 8, 8), (128, 512, 128), (129, 513, 65), (1, 31, 1), (257, 3, 129)]:
        out.append((f"normal {m}x{k}x{n}",
                    (r.standard_normal((m, k)).astype(np.float32),
                     r.standard_normal((k, n)).astype(np.float32))))
        out.append((f"large {m}x{k}x{n}",
                    (r.standard_normal((m, k)).astype(np.float32) * 100,
                     r.standard_normal((k, n)).astype(np.float32) * 100)))
    return out


level(7, "tiled matmul with accumulation", "accumulator dtype; K not divisible by the tile",
      lambda a, b: (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32),
      _build_7,
      ban={"matmul", "dot", "inner", "tensordot", "vdot", "einsum"},
      rtol=1e-3,
      tol_why="float64 reference; a float32 accumulator over K<=513 drifts ~sqrt(K)*eps "
              "~2.7e-6 relative. 1e-3 leaves 350x headroom, and still fails any kernel "
              "that drops a K chunk or swaps operands.",
      array_args=(0, 1))

def _ref_8(x, eps=1e-5):
    xd = x.astype(np.float64)
    mu = xd.mean(axis=-1, keepdims=True)
    var = ((xd - mu) ** 2).mean(axis=-1, keepdims=True)     # two-pass, on purpose
    return ((xd - mu) / np.sqrt(var + eps)).astype(x.dtype)


level(8, "layernorm (mean and variance)", "E[x^2]-E[x]^2 catastrophically cancels at large means",
      _ref_8,
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(8)],
      ban={"mean", "average", "var", "std", "sum", "nanmean", "nanvar"},
      rtol=1e-3,
      tol_why="the reference is two-pass float64. The 'large' rows (mean ~1e4) are the "
              "justification: one-pass E[x^2]-E[x]^2 in float32 subtracts two ~1e8 "
              "quantities to get a ~1 variance, losing everything to rounding. A correct "
              "kernel (two-pass, or one-pass in float64) sits under 1e-5; 1e-3 rejects "
              "exactly the cancellation bug.",
      array_args=(0,))


def _build_9():
    r = _rng(9)
    out = []
    for n, d, w in [(16, 8, 2), (128, 64, 8), (129, 33, 7), (5, 4, 100), (1, 4, 3)]:
        out.append((f"n={n} d={d} w={w}",
                    (r.standard_normal((n, d)).astype(np.float32),
                     r.standard_normal((n, d)).astype(np.float32),
                     r.standard_normal((n, d)).astype(np.float32), w)))
    return out


def _ref_9(q, k, v, w):
    """Windowed attention: position i attends to [i-w, i+w] only."""
    n = q.shape[0]
    qd, kd, vd = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    s = qd @ kd.T / np.sqrt(q.shape[1])
    idx = np.arange(n)
    band = np.abs(idx[:, None] - idx[None, :]) <= w
    s = np.where(band, s, -np.inf)
    m = s.max(axis=-1, keepdims=True)
    e = np.exp(s - m)
    return ((e / e.sum(axis=-1, keepdims=True)) @ vd).astype(q.dtype)


level(9, "windowed attention (band w)", "band edges, and w larger than the sequence",
      _ref_9,
      _build_9,
      ban={"softmax", "matmul", "dot", "tensordot", "einsum"},
      rtol=1e-3,
      tol_why="float64 reference; softmax output is a convex combination so errors stay "
              "small and relative. 1e-3 catches a band that is off by one at either edge "
              "-- every element past the edge carries a whole wrong attention weight.",
      array_args=(0, 1, 2))


def _build_10():
    r = _rng(10)
    out = []
    for ci, L, co, K, s, dl in [(4, 16, 3, 3, 1, 1), (8, 64, 16, 5, 2, 1),
                                (3, 31, 7, 3, 2, 3), (1, 7, 1, 7, 1, 1)]:
        out.append((f"C{ci}->{co} L{L} K{K} s{s} d{dl}",
                    (r.standard_normal((ci, L)).astype(np.float32),
                     r.standard_normal((co, ci, K)).astype(np.float32), s, dl)))
    return out


def _ref_10(x, wt, stride, dilation):
    """1D conv, valid padding. x:[C_in,L] wt:[C_out,C_in,K] -> [C_out,L_out]"""
    c_in, L = x.shape
    c_out, _, K = wt.shape
    L_out = (L - dilation * (K - 1) - 1) // stride + 1
    xd, wd = x.astype(np.float64), wt.astype(np.float64)
    out = np.zeros((c_out, L_out))
    for o in range(L_out):
        for kk in range(K):
            out[:, o] += wd[:, :, kk] @ xd[:, o * stride + kk * dilation]
    return out.astype(np.float32)


level(10, "1D conv, stride and dilation (valid)", "output-size off-by-one; dilation at the tail",
      _ref_10,
      _build_10,
      ban={"convolve", "correlate", "matmul", "dot", "tensordot"},
      rtol=1e-3,
      tol_why="float64 reference with <=64 output positions; float32 drift is ~1e-6. The "
              "1e-3 bar exists to fail the off-by-one output-size formula, which returns "
              "a wrong-SHAPE result, and any tail window that reads past dilation.",
      array_args=(0, 1))

# ------------------------------------------------------------------ self-holdout tier
#
# Same distribution as the graded ladder: reductions, normalizations, composition.
# NEVER used to tune prompts or feedback -- reported separately, as a generalization
# check. The judges hold back three levels of their own; these are ours.

level(11, "row-wise cumulative sum", "the running total crosses TILE boundaries",
      lambda x: np.cumsum(x, axis=1).astype(np.float32),
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(11)],
      ban={"cumsum", "cumprod", "sum", "nansum", "add.reduce"},
      rtol=1e-3,
      tol_why="a prefix sum over <=513 float32 elements accumulates the same n*eps drift "
              "as level 2's sum. The trap is structural, not numerical: within a row, the "
              "first element of column tile t needs the row's running total from tile "
              "t-1, so a kernel that cumsums each column tile independently is wrong by "
              "whole tile-sums -- far above 1e-3.",
      array_args=(0,), holdout=True)

def _ref_12(x):
    xd = x.astype(np.float64)
    mu = xd.mean(axis=-1, keepdims=True)
    return ((xd - mu) ** 2).mean(axis=-1)      # two-pass, like level 8


level(12, "variance over last axis", "E[x^2]-E[x]^2 cancellation, without the normalize to hide it",
      _ref_12,
      lambda: [(lbl, (x,)) for lbl, x in cases_2d(12)],
      ban={"var", "std", "nanvar", "nanstd", "mean", "average", "sum"},
      rtol=1e-3,
      tol_why="same physics as level 8, made harder by outputting the raw variance: on "
              "the 'large' rows (mean ~1e4, variance ~1) the one-pass formula in float32 "
              "returns noise around 0 instead of ~1. Two-pass or float64 kernels sit "
              "under 1e-5.",
      array_args=(0,), holdout=True)


def _build_13():
    r = _rng(13)
    out = []
    for sh in [(8, 16), (129, 513), (1, 7), (7, 1), (31, 127), (128, 1)]:
        out.append((f"normal {sh}", (r.standard_normal(sh).astype(np.float32),)))
        out.append((f"tied {sh}", np.tile(np.array([1.0, 1.0, 0.0, -1.0], np.float32),
                                          (sh[0], sh[1] // 4 + 1))[:, :sh[1]].copy()))
    return out


level(13, "row-wise argmax", "ties return the FIRST index; the result is an index, not a value",
      lambda x: np.argmax(x, axis=1).astype(np.float32),
      lambda: [(lbl, (x,)) for lbl, x in _build_13()],
      ban={"argmax", "argpartition", "argsort", "sort", "max", "amax", "nanmax"},
      rtol=0.0,
      tol_why="the output is an index: either the right integer or the wrong one. Exact "
              "match required. The 'tied' rows (runs of identical values) fail any kernel "
              "that keeps the LAST index instead of the first.",
      array_args=(0,), holdout=True)


# ------------------------------------------------------------------ helpers

def graded_levels():
    return [n for n, s in sorted(LEVELS.items()) if not s["holdout"]]


def holdout_levels():
    return [n for n, s in sorted(LEVELS.items()) if s["holdout"]]


def tile_loop_required(n):
    """True when some case is bigger than one tile, so a compliant kernel must loop."""
    for _lbl, args in LEVELS[n]["build"]():
        for a in args:
            if isinstance(a, np.ndarray):
                if a.shape[0] > TILE_ROWS or a.shape[-1] > TILE_COLS:
                    return True
    return False


def show(n):
    s = LEVELS[n]
    print(f"level {n}: {s['name']}" + ("   [SELF-HOLDOUT]" if s["holdout"] else ""))
    print(f"  trap: {s['trap']}")
    print(f"  tolerance: rtol {s['rtol']:g} of output RMS — {textwrap.fill(s['tol_why'], 78, subsequent_indent='      ')}")
    print(f"  banned: {sorted(s['ban']) or 'nothing extra'}")
    print("REFERENCE (same signature):\n")
    print(textwrap.indent(inspect.getsource(s["ref"]), "  "))
    cases = s["build"]()
    print(f"  {len(cases)} test cases, including partial tiles, prime dimensions, a dimension")
    print("  of 1, large magnitudes, and identical rows.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--level" and sys.argv[2].isdigit():
        show(int(sys.argv[2]))
    else:
        print(f"tile limit {TILE_ROWS}x{TILE_COLS}; tolerance = max |got-want| <= rtol * RMS(reference)\n")
        for n, s in sorted(LEVELS.items()):
            tag = " [holdout]" if s["holdout"] else ""
            print(f"  {n:>2}. {s['name']:<44} trap: {s['trap']}{tag}")
        print("\n  python ladder.py --level N --show")
