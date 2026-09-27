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


def level(n, name, trap, ref, build):
    LEVELS[n] = dict(n=n, name=name, trap=trap, ref=ref, build=build)


level(1, "elementwise relu(a*x+b)", "none — your 'the loop works' checkpoint", _ref_1,
     lambda: [(lbl, (x, 2.5, -0.5)) for lbl, x in _cases_2d(1)])
level(2, "row-wise sum", "the partial final tile", _ref_2,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(2)])
level(3, "row-wise max", "partial tile, and the identity is -inf not 0", _ref_3,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(3)])
level(4, "RMSNorm over last axis", "accumulation order and dtype", _ref_4,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(4)])
level(5, "softmax over last axis", "naive exp OVERFLOWS; needs max subtraction", _ref_5,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(5)])
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
     _ref_7, _build_7)
level(8, "layernorm (mean and variance)",
     "E[x^2]-E[x]^2 catastrophically cancels at large means", _ref_8,
     lambda: [(lbl, (x,)) for lbl, x in _cases_2d(8)])


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
     _ref_9, _build_9)


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
     _ref_10, _build_10)

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

def describe_mismatch(got, want, tol):
    """The message your agent has to learn from. Vague here means slow there."""
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
    denom = np.maximum(np.abs(want), 1e-30)
    rel = np.abs(got - want) / denom
    i = int(np.argmax(rel))
    idx = np.unravel_index(i, got.shape)
    worst = float(rel.flat[i])
    if worst <= tol:
        return None
    frac = float((rel > tol).mean())
    msg = [f"NUMERICAL MISMATCH: worst relative error {worst:.3g} at index "
           f"{tuple(int(x) for x in idx)} (tolerance {tol:g}).",
           f"  expected {want.flat[i]:+.6g}, got {got.flat[i]:+.6g}",
           f"  {frac:.1%} of elements are outside tolerance"]
    # Localise it, because "somewhere" is useless feedback.
    if got.ndim >= 2:
        r, c = int(idx[0]), int(idx[-1])
        in_partial_row = r >= (got.shape[0] // TILE_ROWS) * TILE_ROWS
        in_partial_col = c >= (got.shape[-1] // TILE_COLS) * TILE_COLS
        if in_partial_row or in_partial_col:
            which = " and ".join(x for x in
                                 ["the final PARTIAL ROW tile" if in_partial_row else "",
                                  "the final PARTIAL COLUMN tile" if in_partial_col else ""] if x)
            msg.append(f"  this is in {which} (shape {got.shape}, tile "
                       f"{TILE_ROWS}x{TILE_COLS}) — the ragged edge is the likely cause")
        elif frac > 0.5:
            msg.append("  most elements are wrong, so this is the core arithmetic, "
                       "not an edge case")
        else:
            msg.append(f"  row {r} of {got.shape[0]}, column {c} of {got.shape[-1]}, "
                       "inside a full tile")
    return "\n".join(msg)


def verify(kernel_fn, n, tol=1e-4, stop_early=True):
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
        m = describe_mismatch(got, want, tol)
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


def selftest():
    print("Proving the harness catches planted bugs. If any line says UNDETECTED, the")
    print("harness is useless and nothing built on it can be trusted.\n")
    rc = 0
    ns = {}
    exec(compile(_GOOD_1, "good", "exec"), ns)
    r = verify(ns["kernel"], 1)
    print(f"  correct kernel                 -> {'PASS (good)' if r['ok'] else 'UNDETECTED-REGRESSION: harness rejects a correct kernel'}")
    rc |= 0 if r["ok"] else 1

    ns = {}
    exec(compile(_BAD_1_EDGE, "bad", "exec"), ns)
    r = verify(ns["kernel"], 1)
    print(f"  ragged-edge bug                -> {'caught' if not r['ok'] else 'UNDETECTED'}")
    rc |= 0 if not r["ok"] else 1
    if not r["ok"]:
        print(textwrap.indent(r["failures"][0][1], "      "))

    v = check_rules(_BAD_1_CHEAT, 1)
    print(f"\n  banned einsum                  -> {'caught: ' + v[0] if v else 'UNDETECTED'}")
    rc |= 0 if v else 1

    v = check_rules("import numpy as np\ndef kernel(x):\n    return x.sum(axis=1)\n", 2)
    print(f"  level-2 .sum() cheat            -> {'caught: ' + v[0] if v else 'UNDETECTED'}")
    rc |= 0 if v else 1
    return rc


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--level", type=int)
    ap.add_argument("--show", action="store_true", help="print the reference for this level")
    ap.add_argument("--check", metavar="FILE.py", help="verify a candidate kernel")
    ap.add_argument("--tol", type=float, default=1e-4)
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
