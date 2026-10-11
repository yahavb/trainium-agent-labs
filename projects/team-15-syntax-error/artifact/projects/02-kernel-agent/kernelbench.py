#!/usr/bin/env python3
"""
kernelbench.py — references, hostile test cases, and a verifier for the kernel-agent challenge.

This is the harness the challenge says to build first, so you don't have to. Read it before
you trust it; the failure message it produces is what your agent will learn from, and you
will almost certainly want to improve it.

    python kernelbench.py --list                     # the ladder
    python kernelbench.py --level 1 --show            # reference + signature your kernel must match
    python kernelbench.py --level 1 --check my.py     # verify a candidate
    python kernelbench.py --selftest                 # prove the harness catches planted bugs

A candidate is a Python file defining `kernel(...)` with the same signature as the reference.

Only numpy is required.

Tolerance
---------
An element passes when |got - want| <= atol + rtol * |want| (a mixed criterion, not a bare
relative one). The reference is float64-accurate; kernels compute in float32. So:

    rtol = max(8 * sqrt(n_acc) * eps32, 1e-5)        eps32 = 2**-23 ~= 1.19e-7
    atol = rtol * RMS(reference output)

n_acc is the accumulation length of the level (1 elementwise, row length for row reductions
and norms, K for matmul, d + window for attention, C_in*K for conv; see `level(...)`).
sqrt(n_acc) is the typical growth of float32 rounding error in a length-n sum; 8 is a safety
factor; atol judges near-zero outputs against the magnitude of the output, not against
themselves. The full derivation is in `tolerance()`. `--tol X` overrides rtol only.

Failure messages
----------------
Each numerical failure prints the worst element, then three tagged lines whose leading tag is
stable (grep for it): `TILE <class>:` (relation to the 128x512 tile grid), `PATTERN <class>:`
(spatial shape of the wrong elements), `DIRECTION <class>:` (how the wrong values are wrong).
"""

import argparse
import ast
import sys
import textwrap
import traceback

import numpy as np

TILE_ROWS, TILE_COLS = 128, 512

# Functions nobody may call: they either do the whole job or hide the tiling.
BANNED_GLOBAL = {
    "einsum", "take", "take_along_axis", "put_along_axis", "choose", "select",
    "apply_along_axis", "apply_over_axes", "vectorize", "fromfunction",
    "argsort", "sort", "lexsort", "partition", "argpartition", "compress",
    "extract", "place", "putmask", "nditer", "ndenumerate",
}


# ---------------------------------------------------------------- the ladder

def _ref_1(x, a, b):
    return np.maximum(a * x + b, 0.0)


def _ref_2(x):
    return x.sum(axis=1)


def _ref_3(x):
    return x.max(axis=1)


def _ref_4(x, eps=1e-6):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)


def _ref_5(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp((x - m).astype(np.float64))
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


def _ref_6(x):
    return x.T.copy()


def _ref_7(a, b):
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)


def _ref_8(x, eps=1e-5):
    xd = x.astype(np.float64)
    mu = xd.mean(axis=-1, keepdims=True)
    var = ((xd - mu) ** 2).mean(axis=-1, keepdims=True)     # two-pass, on purpose
    return ((xd - mu) / np.sqrt(var + eps)).astype(x.dtype)


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


def _rng(seed):
    return np.random.default_rng(seed)


def _cases_2d(seed, prime=True):
    """Shapes chosen to break tiling: partial tiles, prime dims, a dim of 1."""
    r = _rng(seed)
    shapes = [(8, 16), (128, 512), (129, 513), (1, 7), (7, 1), (128, 1)]
    if prime:
        shapes += [(31, 127), (257, 61)]
    out = []
    for sh in shapes:
        out.append(("normal", r.standard_normal(sh).astype(np.float32) * 1.0))
        out.append(("large", r.standard_normal(sh).astype(np.float32) * 1e3 + 1e4))
        out.append(("identical rows", np.tile(r.standard_normal((1, sh[1]))
                                             .astype(np.float32), (sh[0], 1))))
        out.append(("zeros", np.zeros(sh, np.float32)))
    return out


LEVELS = {}


def level(n, name, trap, ref, build, n_acc=None, n_acc_doc="1 (elementwise)"):
    """Register a rung. `n_acc(*args)` returns the accumulation length that sets the
    tolerance for one test case (see `tolerance`); default 1 = elementwise."""
    LEVELS[n] = dict(n=n, name=name, trap=trap, ref=ref, build=build,
                     n_acc=n_acc or (lambda *a: 1), n_acc_doc=n_acc_doc)


def _row_len(x, *_):
    """Accumulation length of a reduction/normalisation over the last axis."""
    return int(np.shape(x)[-1])


level(1, "elementwise relu(a*x+b)", "none — your 'the loop works' checkpoint", _ref_1,
     lambda: [(lbl, (x, 2.5, -0.5)) for lbl, x in _cases_2d(1)])
level(2, "row-wise sum", "the partial final tile", _ref_2,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(2)],
     n_acc=_row_len, n_acc_doc="C (row length)")
level(3, "row-wise max", "partial tile, and the identity is -inf not 0", _ref_3,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(3)],
     n_acc=_row_len, n_acc_doc="C (row length)")
level(4, "RMSNorm over last axis", "accumulation order and dtype", _ref_4,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(4)],
     n_acc=_row_len, n_acc_doc="C (row length)")
level(5, "softmax over last axis", "naive exp OVERFLOWS; needs max subtraction", _ref_5,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(5)],
     n_acc=_row_len, n_acc_doc="C (row length)")
level(6, "tiled transpose", "tile boundaries on non-square non-divisible shapes", _ref_6,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(6)])


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
     _ref_7, _build_7,
     n_acc=lambda a, b: int(a.shape[1]), n_acc_doc="K (contraction length)")
level(8, "layernorm (mean and variance)",
     "E[x^2]-E[x]^2 catastrophically cancels at large means", _ref_8,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(8)],
     n_acc=_row_len, n_acc_doc="C (row length)")


def _build_9():
    r = _rng(9)
    out = []
    for n, d, w in [(16, 8, 2), (128, 64, 8), (129, 33, 7), (5, 4, 100), (1, 4, 3)]:
        out.append((f"n={n} d={d} w={w}",
                    (r.standard_normal((n, d)).astype(np.float32),
                     r.standard_normal((n, d)).astype(np.float32),
                     r.standard_normal((n, d)).astype(np.float32), w)))
    return out


level(9, "windowed attention (band w)", "band edges, and w larger than the sequence",
     _ref_9, _build_9,
     # q.k^T sums over d, then softmax and P@V sum over the window: two chained reductions.
     n_acc=lambda q, k, v, w: int(q.shape[1]) + min(2 * int(w) + 1, int(q.shape[0])),
     n_acc_doc="d + min(2w+1, n) (score dot + window)")


def _build_10():
    r = _rng(10)
    out = []
    for ci, L, co, K, s, dl in [(4, 16, 3, 3, 1, 1), (8, 64, 16, 5, 2, 1),
                                (3, 31, 7, 3, 2, 3), (1, 7, 1, 7, 1, 1)]:
        out.append((f"C{ci}->{co} L{L} K{K} s{s} d{dl}",
                    (r.standard_normal((ci, L)).astype(np.float32),
                     r.standard_normal((co, ci, K)).astype(np.float32), s, dl)))
    return out


level(10, "1D conv with stride and dilation", "output-size off-by-one; dilation at the tail",
     _ref_10, _build_10,
     n_acc=lambda x, wt, s, d: int(wt.shape[1]) * int(wt.shape[2]),
     n_acc_doc="C_in * K (receptive field)")

# Per-level bans: the numpy call that would trivially do the whole job.
BANNED_RUNG = {
    2: {"sum", "cumsum", "nansum", "add.reduce", "trace"},
    3: {"max", "amax", "nanmax", "maximum.reduce", "ptp"},
    4: {"norm", "mean", "average", "sum"},
    5: {"softmax", "logsumexp", "sum", "max", "amax"},
    6: {"transpose", "swapaxes", "moveaxis", "rollaxis", "permute"},
    7: {"matmul", "dot", "inner", "tensordot", "vdot"},
    8: {"mean", "average", "var", "std", "sum"},
    9: {"softmax", "matmul", "dot", "tensordot"},
    10: {"convolve", "correlate", "matmul", "dot", "tensordot", "as_strided"},
}


# ---------------------------------------------------------------- rule checking

def check_rules(src, n):
    """Static scan. Returns a list of violations (empty means clean).

    This catches the obvious cheats. It does NOT prove your kernel is properly tiled --
    a human reads that. Do not mistake a clean report here for a legal kernel.
    """
    bad = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"the file does not parse: {e}"]

    banned = BANNED_GLOBAL | BANNED_RUNG.get(n, set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = (f.attr if isinstance(f, ast.Attribute)
                    else f.id if isinstance(f, ast.Name) else None)
            if name in banned:
                bad.append(f"line {node.lineno}: calls banned `{name}` for this level")
            # x.sum(...) / x.max(...) as a method is the same cheat
            if isinstance(f, ast.Attribute) and f.attr in BANNED_RUNG.get(n, set()):
                bad.append(f"line {node.lineno}: method `.{f.attr}()` is banned for this level")
        if isinstance(node, ast.Subscript):
            sl = node.slice
            # boolean-mask or integer-array indexing
            if isinstance(sl, (ast.Compare, ast.List)):
                bad.append(f"line {node.lineno}: fancy/boolean indexing is not allowed")
    return sorted(set(bad))


# ---------------------------------------------------------------- verification

EPS32 = float(np.finfo(np.float32).eps)      # 2**-23 ~= 1.19e-7, unit roundoff is eps/2
TOL_SAFETY = 8.0                             # c in rtol = c * sqrt(n_acc) * eps32
RTOL_FLOOR = 1e-5                            # never stricter than this
RAGGED_CONC = 0.90       # >= this share of wrong elements in the partial tile -> ragged edge
RAGGED_INTERIOR = 0.01   # ... and at most this share of the full-tile interior is wrong


def tolerance(want, n_acc=1, rtol=None):
    """Return (rtol, atol) for comparing a float32 kernel to a float64-accurate reference.

    Criterion, per element:  |got - want| <= atol + rtol * |want|

    Which eps. The kernels compute in float32, so the reference (float64, or float32 rounded
    from float64) is effectively exact and the only legitimate discrepancy is float32
    rounding. eps32 = np.finfo(np.float32).eps = 2**-23 ~= 1.19e-7 (unit roundoff eps32/2).
    A single elementwise op is off by <= eps32/2 relative; a handful of fused elementwise ops
    by a few eps32.

    Why sqrt(n). A length-n float32 reduction (sum, dot product, mean, variance) has a
    worst-case error bound of ~n*eps32*sum|x_i| (Higham, recursive summation), but rounding
    errors are close to independent and zero-mean, so the error that actually occurs grows
    like a random walk, ~sqrt(n)*eps32. The worst-case n*eps32 bound would make rtol ~6e-5
    at n=513, loose enough to accept wrong-but-close kernels, so we budget for the typical
    sqrt(n) growth. The accumulation length n_acc is per level: 1 for elementwise and
    transpose, the row length for sum/max/RMSNorm/softmax/layernorm, K for matmul,
    d + window for attention, C_in*K for conv (see `level(...)`).

    Why the safety factor c = 8. sqrt(n)*eps32 is a one-sigma figure; the max over up to
    ~10^5 output elements sits several sigma out (~4.5 sigma for 1e5 Gaussian samples), and
    the ops are not perfectly conditioned (layernorm amplifies the error in the mean by
    |mean|/std, which the 'large' cases make ~10). 8 covers both. Measured with correct
    float32 tiled kernels using sequential (worst-order) float32 accumulation: levels 2, 4,
    5, 7 use <= 6% of the budget; level 8 on the large-mean cases uses up to ~67% (30 seeds),
    because of that |mean|/std amplification. The level-8 trap (one-pass E[x^2]-E[x]^2
    variance in float32) lands 2.5-3x OVER the budget on the same cases, so c cannot go much
    above ~10 without letting the trap through, nor much below ~6 without false-failing
    correct layernorms. A floor of 1e-5 (= 84 eps32) keeps n_acc=1 levels from demanding
    bit-exactness from fused or reassociated arithmetic.

    Why atol = rtol * RMS(want). A purely relative test divides by |want|, so an output
    that should be ~0 (a layernorm value near the mean, a matmul entry that cancels, a
    softmax tail below float32's normal range) is held to an absolute error of ~1e-4*|want|,
    which float32 cannot deliver: those values come from subtracting O(scale) quantities,
    so their absolute error is O(eps32*scale) regardless of how small the result is. The
    honest yardstick for such an element is the magnitude of the output it belongs to, so
    atol is rtol times the RMS of the reference output. An all-zero reference gets an atol
    of ~1e-43, i.e. exact.

    Strictness. The planted bugs in --selftest (a 1.0001 scale on the ragged tile, an
    unwritten last column, skipped rows) are all still caught; see `selftest()`.

    Pass `rtol` to override the derived rtol (the CLI's --tol); atol still scales with it.
    """
    if rtol is None:
        rtol = max(TOL_SAFETY * np.sqrt(max(int(n_acc), 1)) * EPS32, RTOL_FLOOR)
    w = np.asarray(want, dtype=np.float64)
    w = w[np.isfinite(w)]
    scale = float(np.sqrt(np.mean(w * w))) if w.size else 0.0
    atol = float(rtol) * max(scale, float(np.finfo(np.float32).tiny))
    return float(rtol), atol


def _as_2d(a):
    """Rows x cols view for localisation. A 1-D output (row sum, row max) is a column."""
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1)


def _stride_pattern(mask):
    """If the True entries of a 1-D mask are (nearly) every k-th index, return (k, offset)."""
    n = mask.size
    nbad = int(mask.sum())
    if nbad < 3 or nbad == n:
        return None
    idx = np.arange(n)
    best = None
    for k in range(2, n // 3 + 1):
        for o in range(k):
            pat = (idx % k) == o
            if pat.sum() < 3:
                continue
            j = (pat & mask).sum() / (pat | mask).sum()
            if best is None or j > best[0] + 1e-12:
                best = (j, k, o)
    if best and best[0] >= 0.9:
        return best[1], best[2]
    return None


def _contiguous(mask):
    i = np.flatnonzero(mask)
    return i.size > 0 and i[-1] - i[0] + 1 == i.size, (int(i[0]), int(i[-1])) if i.size else None


def _tile_lines(bad, orig_shape):
    """TILE <class>: where the wrong elements sit relative to the full/partial tile grid."""
    R, C = bad.shape
    nbad = int(bad.sum())
    axes = [("ROWS", "rows", R, TILE_ROWS, 0)]
    if C > 1 or len(orig_shape) >= 2:
        axes.append(("COLS", "columns", C, TILE_COLS, 1))
    if all(N < T for _, _, N, T, _ in axes):
        return [f"TILE SUB-TILE-SHAPE: output shape {orig_shape} is smaller than one "
                f"{TILE_ROWS}x{TILE_COLS} tile, so every element is in the single partial tile "
                f"and tile position says nothing here; the ragged-edge hint does NOT apply. "
                f"Use the PATTERN line."]
    claims, notes = [], []
    for tag, word, N, T, ax in axes:
        full = (N // T) * T
        if N < T:
            notes.append(f"{word}: {N} < {T}, no full tile to compare against")
            continue
        if full == N:
            notes.append(f"{word}: {N} is a multiple of {T}, no partial tile")
            continue
        sl = [slice(None), slice(None)]
        sl[ax] = slice(full, None)
        in_part = int(bad[tuple(sl)].sum())
        sl[ax] = slice(0, full)
        interior = bad[tuple(sl)]
        frac = in_part / nbad
        irate = float(interior.mean())
        if frac >= RAGGED_CONC and irate <= RAGGED_INTERIOR:
            claims.append(
                f"TILE RAGGED-EDGE-{tag}: {frac:.0%} of the wrong elements are in {word} >= "
                f"{full} (the final partial {word[:-1]} tile, {N - full} of {N}) and only "
                f"{irate:.1%} of the full-tile interior is wrong. The ragged edge is the likely "
                f"cause: check the slice end and loop bound of the LAST {word[:-1]} tile "
                f"(min({full}+{T}, {N}), not {full}+{T} or {N}-1).")
        else:
            notes.append(f"{frac:.0%} of the wrong elements are in the partial {word[:-1]} tile "
                         f"({word} >= {full}); {irate:.1%} of the full-tile interior is wrong")
    if claims:
        return claims
    return ["TILE NOT-EDGE: the errors are not concentrated in a partial tile ("
            + "; ".join(notes) + "). The tile loop bounds are probably not the problem."]


def _pattern_line(bad):
    """PATTERN <class>: the spatial shape of the wrong elements."""
    R, C = bad.shape
    rows = bad.any(axis=1)
    cols = bad.any(axis=0)
    nr, nc = int(rows.sum()), int(cols.sum())
    one_col = C == 1
    rw = "elements" if one_col else "rows"
    if R > 1 and nr == 1 and rows[0]:
        return (f"PATTERN FIRST-ROW: only row 0 is wrong. The first iteration is special-cased "
                f"or initialised differently (accumulator init, loop starting at 1).")
    if R > 1 and nr == 1 and rows[-1]:
        return (f"PATTERN LAST-ROW: only the last row ({R - 1}) is wrong. Off-by-one in the row "
                f"loop bound or slice end (range(R-1), r0:R-1).")
    if not one_col and nc == 1 and cols[0]:
        return (f"PATTERN FIRST-COL: only column 0 is wrong. The first column/iteration of the "
                f"inner loop is handled differently (accumulator init, loop starting at 1).")
    if not one_col and nc == 1 and cols[-1]:
        return (f"PATTERN LAST-COL: only the last column ({C - 1}) is wrong. Off-by-one in the "
                f"column loop bound or slice end (range(C-1), c0:C-1).")
    st = _stride_pattern(rows) if R > 1 else None
    if st:
        k, o = st
        return (f"PATTERN STRIDE-ROWS: the wrong {rw} are one in every {k}, starting at {o} "
                f"({nr} of {R}). A loop step bug: the row loop steps by {k} or a tile is "
                f"indexed with a stride; every {rw[:-1]} must be visited exactly once.")
    st = _stride_pattern(cols) if not one_col else None
    if st:
        k, o = st
        return (f"PATTERN STRIDE-COLS: the wrong columns are one in every {k}, starting at {o} "
                f"({nc} of {C}). A loop step bug in the column loop; every column must be "
                f"visited exactly once.")
    ok_r, span_r = _contiguous(rows)
    if R > 1 and ok_r and nr < R:
        return (f"PATTERN ROW-BLOCK: the wrong {rw} are one contiguous block, {span_r[0]}..{span_r[1]} "
                f"of {R}; the rest is right. One tile or one loop range is wrong; find the "
                f"iteration that covers {rw} {span_r[0]}..{span_r[1]}.")
    ok_c, span_c = _contiguous(cols)
    if not one_col and ok_c and nc < C:
        return (f"PATTERN COL-BLOCK: the wrong columns are one contiguous block, "
                f"{span_c[0]}..{span_c[1]} of {C}, across {nr} of {R} rows. One column tile or "
                f"one loop range is wrong; find the iteration that covers columns "
                f"{span_c[0]}..{span_c[1]}.")
    if nr == R and one_col:
        return (f"PATTERN ALL-ROWS: every output row is wrong. The output is 1-D (a reduction "
                f"over the columns), so a column-tile bug cannot be seen by position: every row "
                f"wrong means the reduction itself is wrong, often a column tile (the partial "
                f"one first) dropped, double-counted, or started from the wrong identity.")
    if nr == R:
        return (f"PATTERN ALL-ROWS: every row has wrong elements ({bad.mean():.1%} of all "
                f"elements). This is the core arithmetic, not an edge or a loop bound.")
    return (f"PATTERN SCATTERED: {nr} of {R} {rw} and {nc} of {C} columns have wrong elements "
            f"with no block or stride structure. Suspect value-dependent arithmetic (overflow, "
            f"cancellation, a wrong identity element) rather than indexing.")


def _direction_line(G, W, bad, rtol, atol):
    """DIRECTION <class>: how the wrong values are wrong."""
    g, w = G[bad], W[bad]
    n = g.size
    if np.all(g == 0):
        return (f"DIRECTION ZERO: all {n} wrong elements are exactly 0 where the reference is "
                f"not. They were never written (a skipped iteration) or a partial result was "
                f"never added back.")
    # Shifted by one along an axis: got[i, j] == want[i, j-1] (or j+1, i-1, i+1).
    for ax, word in ((1, "column"), (0, "row")):
        if W.shape[ax] < 2:
            continue
        for s, side in ((1, "before"), (-1, "after")):
            Ws = np.roll(W, s, axis=ax)
            valid = np.ones_like(bad)
            edge = [slice(None), slice(None)]
            edge[ax] = 0 if s == 1 else -1
            valid[tuple(edge)] = False
            m = bad & valid
            if m.sum() < 3:
                continue
            hit = np.abs(G[m] - Ws[m]) <= atol + rtol * np.abs(Ws[m])
            if hit.mean() >= 0.9 and m.sum() >= 0.9 * n:
                return (f"DIRECTION SHIFTED-{word.upper()}S: {hit.mean():.0%} of the wrong "
                        f"elements hold the expected value of the {word} {side} them "
                        f"(off by one {word}). An off-by-one in a {word} index or slice start.")
    nz = np.abs(w) > 0
    if nz.sum() >= max(1, 0.9 * n):
        r = g[nz] / w[nz]
        med = float(np.median(r))
        if abs(med + 1) < 1e-2 and np.mean(np.abs(r + 1) < 1e-2) >= 0.9:
            return (f"DIRECTION SIGN-FLIPPED: got ~= -expected on {np.mean(np.abs(r + 1) < 1e-2):.0%} "
                    f"of the wrong elements. A subtraction is reversed or a negation is "
                    f"missing/extra.")
        if abs(med - 1) > 2 * rtol:
            close = np.abs(r - med) <= 0.05 * abs(med - 1)
            if close.mean() >= 0.9:
                return (f"DIRECTION SCALED: got/expected ~= {med:.6g} on {close.mean():.0%} of "
                        f"the wrong elements, a constant factor. A normalisation is wrong "
                        f"(divide by the wrong count, a factor applied twice or not at all).")
    hi = float(np.mean(g > w))
    rel = (g - w) / np.maximum(np.abs(w), atol)
    medrel = float(np.median(rel))
    if hi >= 0.9:
        return (f"DIRECTION TOO-LARGE: got > expected on {hi:.0%} of the wrong elements "
                f"(median excess {medrel:+.3g} relative). Something is added or counted that "
                f"should not be (double-counted tile, extra term, bad identity element).")
    if hi <= 0.1:
        return (f"DIRECTION TOO-SMALL: got < expected on {1 - hi:.0%} of the wrong elements "
                f"(median shortfall {medrel:+.3g} relative). Something is missing (a tile or "
                f"term not accumulated, a value clipped or truncated).")
    return (f"DIRECTION MIXED: {hi:.0%} too large and {1 - hi:.0%} too small, not a constant "
            f"bias. Suspect the wrong element being read, or precision loss "
            f"(float16 accumulation, cancellation), not a missing term.")


def describe_mismatch(got, want, tol=None, n_acc=1):
    """The message your agent has to learn from. Vague here means slow there.

    `tol` is None (derive rtol/atol from n_acc, see `tolerance`), a float (override rtol;
    atol still scales with the output RMS), or an explicit (rtol, atol) pair.
    Returns None when every element is within tolerance.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return (f"WRONG SHAPE: returned {got.shape}, reference is {want.shape}. "
                f"Check the output-size formula, not the arithmetic.")
    if not np.all(np.isfinite(got)):
        n_nan = int(np.isnan(got).sum())
        n_inf = int(np.isinf(got).sum())
        where = np.argwhere(~np.isfinite(got))[0]
        return (f"NON-FINITE OUTPUT: {n_nan} NaN and {n_inf} Inf values, first at index "
                f"{tuple(int(i) for i in where)}. Overflow in exp, or a division by a zero "
                f"denominator. Subtract the max before exponentiating.")
    if isinstance(tol, tuple):
        rtol, atol = float(tol[0]), float(tol[1])
    else:
        rtol, atol = tolerance(want, n_acc, rtol=tol)
    err = np.abs(got - want)
    allowed = atol + rtol * np.abs(want)
    bad = err > allowed
    if not bad.any():
        return None
    over = err / allowed
    i = int(np.argmax(over))
    idx = tuple(int(x) for x in np.unravel_index(i, got.shape))
    frac = float(bad.mean())
    msg = [f"NUMERICAL MISMATCH: worst element at index {idx}: |got-want| = {err.flat[i]:.3g}, "
           f"allowed {allowed.flat[i]:.3g} ({over.flat[i]:.3g}x over; rtol {rtol:.3g}, "
           f"atol {atol:.3g}, n_acc {n_acc}).",
           f"  expected {want.flat[i]:+.6g}, got {got.flat[i]:+.6g}",
           f"  {int(bad.sum())} of {bad.size} elements ({frac:.1%}) are outside tolerance"]
    G, W, B = _as_2d(got), _as_2d(want), _as_2d(bad)
    msg += ["  " + s for s in _tile_lines(B, got.shape)]
    msg.append("  " + _pattern_line(B))
    msg.append("  " + _direction_line(G, W, B, rtol, atol))
    return "\n".join(msg)


def verify(kernel_fn, n, tol=None, stop_early=True):
    """Run every case of level n. `tol`: None = derived per case, float = rtol override."""
    spec = LEVELS[n]
    cases = spec["build"]()
    failures, passed = [], 0
    for label, args in cases:
        try:
            want = spec["ref"](*args)
        except Exception as e:                       # a broken reference is our bug
            failures.append((label, f"REFERENCE ITSELF FAILED: {e}"))
            continue
        try:
            got = kernel_fn(*[a.copy() if isinstance(a, np.ndarray) else a for a in args])
        except Exception:
            tb = traceback.format_exc(limit=3).strip().splitlines()
            failures.append((label, "RAISED: " + " | ".join(tb[-2:])))
            if stop_early:
                break
            continue
        m = describe_mismatch(got, want, tol, spec["n_acc"](*args))
        if m:
            shp = tuple(np.shape(args[0]))
            failures.append((f"{label} input{shp}", m))
            if stop_early:
                break
        else:
            passed += 1
    return dict(level=n, total=len(cases), passed=passed, failures=failures,
                ok=(passed == len(cases)))


def report(res, n):
    spec = LEVELS[n]
    print(f"level {n}: {spec['name']}")
    print(f"  trap: {spec['trap']}")
    print(f"  {res['passed']}/{res['total']} cases passed")
    if res["ok"]:
        print("  VERIFIED")
        return 0
    print("  FAILED\n")
    for label, m in res["failures"][:3]:
        print(f"  case: {label}")
        print(textwrap.indent(m, "    "))
        print()
    print("  ^ this text is what you should feed back to the model. If it is not enough to")
    print("    locate the bug, improve THIS message before you touch your prompt.")
    return 1


# ---------------------------------------------------------------- selftest

_GOOD_1 = """
import numpy as np
def kernel(x, a, b):
    out = np.empty_like(x)
    R, C = x.shape
    for r0 in range(0, R, 128):
        for c0 in range(0, C, 512):
            t = x[r0:r0+128, c0:c0+512]
            y = a * t + b
            out[r0:r0+128, c0:c0+512] = np.where(y > 0, y, 0.0)
    return out
"""

# Only wrong on the ragged edge: exactly the bug the harness must catch.
_BAD_1_EDGE = _GOOD_1.replace("t = x[r0:r0+128, c0:c0+512]",
                              "t = x[r0:r0+128, c0:c0+512]\n            "
                              "t = t[:, :512] if t.shape[1] == 512 else t * 1.0001")
_BAD_1_CHEAT = """
import numpy as np
def kernel(x, a, b):
    return np.einsum('ij,->ij', a * x + b, 1.0).clip(0)
"""

# Off-by-one in the slice end of the final partial column tile: the last column is never written.
_BAD_1_LASTCOL = """
import numpy as np
def kernel(x, a, b):
    out = np.zeros_like(x)
    R, C = x.shape
    for r0 in range(0, R, 128):
        for c0 in range(0, C, 512):
            c1 = c0 + 512 if c0 + 512 <= C else C - 1
            y = a * x[r0:r0+128, c0:c1] + b
            out[r0:r0+128, c0:c1] = np.where(y > 0, y, 0.0)
    return out
"""

# Loop-step bug: only every other row of each tile is processed.
_BAD_1_STRIDE = """
import numpy as np
def kernel(x, a, b):
    out = np.zeros_like(x)
    R, C = x.shape
    for r0 in range(0, R, 128):
        for c0 in range(0, C, 512):
            y = a * x[r0:r0+128:2, c0:c0+512] + b
            out[r0:r0+128:2, c0:c0+512] = np.where(y > 0, y, 0.0)
    return out
"""

# Correct float32 tiled softmax (three passes, sequential float32 accumulation).
_GOOD_5 = """
import numpy as np
def kernel(x):
    R, C = x.shape
    out = np.empty(x.shape, np.float32)
    for r0 in range(0, R, 128):
        t = x[r0:r0+128].astype(np.float32)
        m = np.full(t.shape[0], -np.inf, np.float32)
        for c0 in range(0, C, 512):
            u = t[:, c0:c0+512]
            for j in range(u.shape[1]):
                m = np.maximum(m, u[:, j])
        s = np.zeros(t.shape[0], np.float32)
        for c0 in range(0, C, 512):
            e = np.exp(t[:, c0:c0+512] - m[:, None])
            for j in range(e.shape[1]):
                s = s + e[:, j]
        for c0 in range(0, C, 512):
            out[r0:r0+128, c0:c0+512] = np.exp(t[:, c0:c0+512] - m[:, None]) / s[:, None]
    return out
"""

# Correct float32 tiled softmax backward: dx = p * (g - <p, g>), p = softmax(x).
_GOOD_SOFTMAX_BWD = """
import numpy as np
def kernel(x, g):
    R, C = x.shape
    out = np.empty(x.shape, np.float32)
    for r0 in range(0, R, 128):
        t, gt = x[r0:r0+128], g[r0:r0+128]
        m = np.full(t.shape[0], -np.inf, np.float32)
        for c0 in range(0, C, 512):
            u = t[:, c0:c0+512]
            for j in range(u.shape[1]):
                m = np.maximum(m, u[:, j])
        s = np.zeros(t.shape[0], np.float32)
        for c0 in range(0, C, 512):
            e = np.exp(t[:, c0:c0+512] - m[:, None])
            for j in range(e.shape[1]):
                s = s + e[:, j]
        d = np.zeros(t.shape[0], np.float32)
        for c0 in range(0, C, 512):
            pg = np.exp(t[:, c0:c0+512] - m[:, None]) / s[:, None] * gt[:, c0:c0+512]
            for j in range(pg.shape[1]):
                d = d + pg[:, j]
        for c0 in range(0, C, 512):
            p = np.exp(t[:, c0:c0+512] - m[:, None]) / s[:, None]
            out[r0:r0+128, c0:c0+512] = p * (gt[:, c0:c0+512] - d[:, None])
    return out
"""

# Correct float32 tiled layernorm (two-pass, sequential float32 accumulation).
_GOOD_8 = """
import numpy as np
def kernel(x, eps=1e-5):
    R, C = x.shape
    out = np.empty_like(x)
    for r0 in range(0, R, 128):
        t = x[r0:r0+128]
        s = np.zeros(t.shape[0], np.float32)
        for c0 in range(0, C, 512):
            u = t[:, c0:c0+512]
            for j in range(u.shape[1]):
                s = s + u[:, j]
        mu = s / np.float32(C)
        v = np.zeros(t.shape[0], np.float32)
        for c0 in range(0, C, 512):
            d = t[:, c0:c0+512] - mu[:, None]
            for j in range(d.shape[1]):
                v = v + d[:, j] * d[:, j]
        inv = np.float32(1.0) / np.sqrt(v / np.float32(C) + np.float32(eps))
        for c0 in range(0, C, 512):
            out[r0:r0+128, c0:c0+512] = (t[:, c0:c0+512] - mu[:, None]) * inv[:, None]
    return out
"""

# Correct float32 tiled matmul (float32 accumulator over 128-deep K tiles).
_GOOD_7 = """
import numpy as np
def kernel(a, b):
    M, K = a.shape
    N = b.shape[1]
    out = np.zeros((M, N), np.float32)
    for m0 in range(0, M, 128):
        for n0 in range(0, N, 512):
            acc = np.zeros((min(128, M - m0), min(512, N - n0)), np.float32)
            for k0 in range(0, K, 128):
                acc += a[m0:m0+128, k0:k0+128] @ b[k0:k0+128, n0:n0+512]
            out[m0:m0+128, n0:n0+512] = acc
    return out
"""

# The criterion this harness used before: bare relative 1e-4 against max(|want|, 1e-30).
_OLD_TOL = (1e-4, 1e-34)


def _load(src, name):
    ns = {}
    exec(compile(src, name, "exec"), ns)
    return ns["kernel"]


def selftest():
    print("Proving the harness catches planted bugs. If any line says UNDETECTED, the")
    print("harness is useless and nothing built on it can be trusted.\n")
    rc = 0

    def expect(label, msg, *needles, absent=()):
        """A planted bug must be caught AND carry the expected localisation tags."""
        nonlocal rc
        if msg is None:
            print(f"  {label:<38} -> UNDETECTED")
            rc |= 1
            return
        missing = [s for s in needles if s not in msg]
        wrong = [s for s in absent if s in msg]
        if missing or wrong:
            print(f"  {label:<38} -> caught but MISLOCALISED (missing {missing}, "
                  f"should not say {wrong})")
            rc |= 1
        else:
            print(f"  {label:<38} -> caught, localised as {', '.join(needles)}")
        print(textwrap.indent(msg, "      "))

    # -- the original checks
    r = verify(_load(_GOOD_1, "good"), 1)
    print(f"  {'correct kernel (level 1)':<38} -> "
          f"{'PASS (good)' if r['ok'] else 'UNDETECTED-REGRESSION: harness rejects a correct kernel'}")
    rc |= 0 if r["ok"] else 1

    r = verify(_load(_BAD_1_EDGE, "bad"), 1)
    print(f"  {'ragged-edge bug (x1.0001, level 1)':<38} -> {'caught' if not r['ok'] else 'UNDETECTED'}")
    rc |= 0 if not r["ok"] else 1
    if not r["ok"]:
        print(textwrap.indent(r["failures"][0][1], "      "))
    # same bug on a shape with full tiles + a partial column tile: must claim the ragged edge
    xg = _rng(11).standard_normal((200, 700)).astype(np.float32)
    k = _load(_BAD_1_EDGE, "bad")
    expect("  same bug, shape (200, 700)", describe_mismatch(k(xg, 2.5, -0.5), _ref_1(xg, 2.5, -0.5)),
           "TILE RAGGED-EDGE-COLS", "PATTERN COL-BLOCK", "DIRECTION TOO-LARGE")

    # -- (a) correct kernels whose outputs reach ~0 must pass; the old criterion failed them
    print()
    r_ = _rng(14)
    x = r_.standard_normal((32, 600)).astype(np.float32)
    x[::2] = -27.6                       # peaked rows: one dominant logit, tails p ~ 1e-12
    x[::2, 0] = 0.0
    g = r_.standard_normal((32, 600)).astype(np.float32)
    xd, gd = x.astype(np.float64), g.astype(np.float64)
    e = np.exp(xd - xd.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    want = p * (gd - (p * gd).sum(axis=1, keepdims=True))   # float64 reference
    got = _load(_GOOD_SOFTMAX_BWD, "good_softmax_bwd")(x, g)
    near = np.abs(want[(np.abs(want) > 1e-13) & (np.abs(want) < 1e-11)])
    old_m = describe_mismatch(got, want, _OLD_TOL)
    new_m = describe_mismatch(got, want, None, 600)
    rt, at = tolerance(want, 600)
    print(f"  (a) correct float32 softmax-backward p*(g - <p,g>), 32x600, half the rows peaked:")
    print(f"      {near.size} outputs have |value| in (1e-13, 1e-11), e.g. {near[0]:.3g}; "
          f"rtol {rt:.3g}, atol {at:.3g}")
    print(f"      old relative 1e-4: {'pass' if old_m is None else 'FALSE FAIL'}"
          + ("" if old_m is None else " (" + old_m.splitlines()[1].strip() + "; float32 "
             "g - <p,g> cancels to 0 where float64 keeps a tiny nonzero value)"))
    print(f"      new mixed:         {'PASS (good)' if new_m is None else 'FAIL'}")
    rc |= 0 if new_m is None else 1
    if new_m:
        print(textwrap.indent(new_m, "      "))
    # Plain softmax tails are NOT a false fail under either criterion: exp and divide keep
    # relative precision, so tiny outputs are tiny-and-accurate. Shown for the record.
    x = np.tile(np.linspace(0.0, -100.0, 600, dtype=np.float32), (3, 1))
    xd = x.astype(np.float64)
    e = np.exp(xd - xd.max(axis=1, keepdims=True))
    want = e / e.sum(axis=1, keepdims=True)
    got = _load(_GOOD_5, "good5")(x)
    ok_old = describe_mismatch(got, want, _OLD_TOL) is None
    ok_new = describe_mismatch(got, want, None, 600) is None
    print(f"  (a) correct float32 softmax forward, outputs down to {want.min():.2g}: "
          f"new {'PASS (good)' if ok_new else 'FAIL'}; old {'pass' if ok_old else 'FALSE FAIL'}")
    rc |= 0 if ok_new else 1
    for n, src, name in ((5, _GOOD_5, "softmax"), (7, _GOOD_7, "matmul"), (8, _GOOD_8, "layernorm")):
        k = _load(src, name)
        new = verify(k, n, stop_early=False)
        old = verify(k, n, tol=_OLD_TOL, stop_early=False)
        print(f"  (a) correct float32 {name:<9} level {n:>2}: new {new['passed']}/{new['total']} "
              f"{'PASS (good)' if new['ok'] else 'FAIL'}; old relative 1e-4 passed "
              f"{old['passed']}/{old['total']}")
        rc |= 0 if new["ok"] else 1
        if not new["ok"]:
            print(textwrap.indent(new["failures"][0][1], "      "))
        elif not old["ok"]:
            lbl, m = old["failures"][0]
            print(f"      old criterion's first false fail, case {lbl}: " + m.splitlines()[1].strip())

    # -- (b) off-by-one in the last partial tile, shape with full tiles + a partial tile
    print()
    k = _load(_BAD_1_LASTCOL, "lastcol")
    expect("(b) off-by-one, shape (200, 700)", describe_mismatch(k(xg, 2.5, -0.5), _ref_1(xg, 2.5, -0.5)),
           "TILE RAGGED-EDGE-COLS", "PATTERN LAST-COL", "DIRECTION ZERO")
    # -- (c) the same bug on a shape smaller than one tile: must NOT claim the ragged edge
    xs = _rng(12).standard_normal((8, 100)).astype(np.float32)
    expect("(c) same off-by-one, shape (8, 100)", describe_mismatch(k(xs, 2.5, -0.5), _ref_1(xs, 2.5, -0.5)),
           "TILE SUB-TILE-SHAPE", "PATTERN LAST-COL", absent=("RAGGED-EDGE",))
    # -- (d) every other row skipped
    k = _load(_BAD_1_STRIDE, "stride")
    expect("(d) skips every other row", describe_mismatch(k(xg, 2.5, -0.5), _ref_1(xg, 2.5, -0.5)),
           "PATTERN STRIDE-ROWS", "one in every 2, starting at 1", "DIRECTION ZERO",
           absent=("RAGGED-EDGE",))
    # -- direction classes on a whole-array error
    w = _ref_5(_rng(13).standard_normal((40, 50)).astype(np.float32))
    expect("constant factor 1.5", describe_mismatch(w * 1.5, w, None, 50),
           "PATTERN ALL-ROWS", "DIRECTION SCALED")
    expect("sign flip", describe_mismatch(-w, w, None, 50), "DIRECTION SIGN-FLIPPED")

    # -- rule checks
    print()
    v = check_rules(_BAD_1_CHEAT, 1)
    print(f"  {'banned einsum':<38} -> {'caught: ' + v[0] if v else 'UNDETECTED'}")
    rc |= 0 if v else 1

    v = check_rules("import numpy as np\ndef kernel(x):\n    return x.sum(axis=1)\n", 2)
    print(f"  {'level-2 .sum() cheat':<38} -> {'caught: ' + v[0] if v else 'UNDETECTED'}")
    rc |= 0 if v else 1

    print()
    print("SELFTEST PASSED: every planted bug caught and localised, every correct kernel accepted."
          if rc == 0 else "SELFTEST FAILED: see the lines above.")
    return rc


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--level", type=int)
    ap.add_argument("--show", action="store_true", help="print the reference for this level")
    ap.add_argument("--check", metavar="FILE.py", help="verify a candidate kernel")
    ap.add_argument("--tol", type=float, default=None,
                    help="override the derived rtol (atol still scales with the output RMS)")
    ap.add_argument("--all-failures", action="store_true",
                    help="do not stop at the first failing case")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    if a.list or (not a.level):
        print(f"tile limit {TILE_ROWS}x{TILE_COLS}\n")
        for n, s in LEVELS.items():
            print(f"  {n:>2}. {s['name']:<36} trap: {s['trap']}")
        print("\n  python kernelbench.py --level N --show")
        print("  python kernelbench.py --level N --check my_kernel.py")
        return

    n = a.level
    if n not in LEVELS:
        sys.exit(f"no level {n}; 1..{max(LEVELS)}")

    if a.show:
        import inspect
        s = LEVELS[n]
        print(f"level {n}: {s['name']}\ntrap: {s['trap']}\n")
        print("REFERENCE (your kernel must match this, with the same signature):\n")
        print(textwrap.indent(inspect.getsource(s["ref"]), "  "))
        print(f"tolerance: |got-want| <= atol + rtol*|want|, rtol = max(8*sqrt(n_acc)*eps32, 1e-5), "
              f"atol = rtol*RMS(reference); n_acc = {s['n_acc_doc']}")
        print(f"banned for this level: {sorted(BANNED_RUNG.get(n, set())) or 'nothing extra'}")
        print(f"always banned: {sorted(BANNED_GLOBAL)}")
        print(f"\n{len(s['build']())} test cases, including partial tiles, prime dimensions, "
              f"a dimension of 1, large magnitudes, and identical rows.")
        return

    if not a.check:
        sys.exit("give me something to check: --check my_kernel.py (or --show)")

    src = open(a.check).read()
    viol = check_rules(src, n)
    if viol:
        print(f"RULE VIOLATIONS in {a.check} — this scores zero regardless of correctness:")
        for v in viol:
            print(f"  {v}")
        sys.exit(2)

    ns = {}
    try:
        exec(compile(src, a.check, "exec"), ns)
    except Exception as e:
        sys.exit(f"{a.check} failed to import: {e}")
    if "kernel" not in ns:
        sys.exit(f"{a.check} defines no `kernel` function")

    sys.exit(report(verify(ns["kernel"], n, a.tol, not a.all_failures), n))


if __name__ == "__main__":
    main()
