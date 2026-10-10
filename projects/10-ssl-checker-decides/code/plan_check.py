#!/usr/bin/env python3
"""
plan_check.py -- check a TILING PLAN for level 4 (tiled matmul) before any kernel is written.

Levels 1, 3 and 4 stall on the same idiom every run: a 1-D tile, a reshape instead of a slice, one
tile for the whole tensor (STATE.md). Those are mistakes in the PLAN -- which slices go where -- that
only surface after the model has spent a round writing NKI around them. This file checks the plan on
its own, in plain Python, in microseconds, so the agent can be told what is wrong with it first.

A plan is JSON, written in terms of the shape (M, K, N), not for one shape:

    {"loops":  {"m": "M // 128", "n": "N // 512", "k": "K // 128"},
     "accumulate_over": ["k"],
     "reads":  {"lhsT": [["k*128", "(k+1)*128"], ["m*128", "(m+1)*128"]],
                "rhs":  [["k*128", "(k+1)*128"], ["n*512", "(n+1)*512"]]},
     "writes": {"out":  [["m*128", "(m+1)*128"], ["n*512", "(n+1)*512"]]}}

Each access is a ["start", "stop"] pair of strings per axis (stop exclusive), partition axis first. Expressions may use + - * / // %,
ceil, min, max, the shape names and the loop names. Hypothesis generates shapes, the plan is
evaluated on each, and a failure is SHRUNK to the smallest shape that still fails. On that shape the
checker says which element was missed or doubled, which tile is too big, and which expression did it.

    python plan_check.py --selftest
    python plan_check.py --plan my_plan.json
    python plan_check.py --plan my_plan.json --beyond-contract     # ragged shapes too

What it checks, on every shape:
  1. every loop count is a non-negative integer, every slice is non-empty and inside its tensor
  2. tile limits: partition axis <= 128 for every tile; lhsT (stationary) free <= 128; rhs
     (moving) and the output tile free <= 512
  3. the math: for each output tile, lhsT's M slice is the tile's rows, rhs's N slice is its
     columns, both read the same K slice, and the K slices cover [0, K) exactly once
  4. coverage: every output element is written exactly once
"""

import argparse
import ast
import itertools
import json
import math
import sys
import textwrap
import time

import numpy as np

try:
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover
    sys.exit("plan_check.py needs Hypothesis:  pip install hypothesis")

import nkibench

PMAX = nkibench.PMAX                       # partition axis, every tile
STAT_FMAX = nkibench.GEMM_STATIONARY_FMAX  # lhsT free axis
MOV_FMAX = nkibench.GEMM_MOVING_FMAX       # rhs free axis, and the PSUM output tile
SHAPE_NAMES = ("M", "K", "N")
MAX_ITERATIONS = 100_000


class PlanError(Exception):
    """The plan cannot be evaluated at all. Its text is the agent-facing message."""


# ---------------------------------------------------------------- expressions

_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)
_FUNCS = {"ceil": math.ceil, "min": min, "max": max}


def _compile(src, where):
    if not isinstance(src, (str, int)):
        raise PlanError(f"{where}: expected an expression string, got {src!r}")
    src = str(src)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise PlanError(f"{where}: '{src}' is not a valid expression")
    for node in ast.walk(tree):
        bad = not isinstance(node, _NODES) or (
            isinstance(node, ast.Call)
            and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS))
        if bad:
            raise PlanError(f"{where}: '{src}' uses something other than + - * / // %, "
                            f"ceil, min, max, numbers, the shape names M K N and the loop names")
    return src, compile(tree, where, "eval")


def _eval(expr, env, where):
    src, code = expr
    try:
        v = eval(code, {"__builtins__": {}}, {**_FUNCS, **env})
    except NameError as e:
        name = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise PlanError(f"{where}: '{src}' uses '{name}', which is not defined here. Allowed: "
                        f"{', '.join(sorted(env))}")
    except ZeroDivisionError:
        raise PlanError(f"{where}: '{src}' divides by zero when {_fmt_env(env)}")
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if not isinstance(v, int):
        raise PlanError(f"{where}: '{src}' is {v!r} when {_fmt_env(env)}, not an integer. "
                        f"Use // or ceil().")
    return v


def _fmt_env(env):
    return ", ".join(f"{k}={v}" for k, v in env.items())


# ---------------------------------------------------------------- the plan

def parse_plan(plan):
    """Static checks, once per plan. Returns the compiled plan or raises PlanError."""
    if not isinstance(plan, dict):
        raise PlanError("the plan must be a JSON object with loops, reads and writes")
    for key in ("loops", "reads", "writes"):
        if key not in plan:
            raise PlanError(f"the plan has no '{key}'. It needs loops, reads and writes.")
    loops = {v: _compile(e, f"loops.{v}") for v, e in plan["loops"].items()}
    for v in loops:
        if v in SHAPE_NAMES or not v.isidentifier():
            raise PlanError(f"loop name '{v}' is not allowed; use a new name like m, n, k")
    acc = plan.get("accumulate_over", ["k"])
    missing = [v for v in acc if v not in loops]
    if missing:
        raise PlanError(f"accumulate_over names {missing}, which are not loops")

    def access(table, name, ndim_names):
        if name not in table:
            raise PlanError(f"the plan does not say how it accesses '{name}'")
        a = table[name]
        if not isinstance(a, list) or len(a) != 2:
            raise PlanError(
                f"{name}: the access has {len(a) if isinstance(a, list) else 'no'} axes; a tile "
                f"must be 2-D, partition axis first ([start, stop] for {ndim_names[0]}, then for "
                f"{ndim_names[1]}). A 1-D tile is the level-1 wall.")
        out = []
        for i, ax in enumerate(a):
            if not isinstance(ax, list) or len(ax) != 2:
                raise PlanError(f"{name} axis {i}: expected [start, stop]")
            out.append((_compile(ax[0], f"{name}[{i}].start"),
                        _compile(ax[1], f"{name}[{i}].stop")))
        return out

    return dict(loops=loops, acc=list(acc),
                lhsT=access(plan["reads"], "lhsT", ("K", "M")),
                rhs=access(plan["reads"], "rhs", ("K", "N")),
                out=access(plan["writes"], "out", ("M", "N")))


def _slices(acc, env, name, dims, bounds=True, p=None, counts=None):
    """Evaluate one access to [(start, stop)] and, unless bounds=False, check it fits the tensor."""
    got = []
    for i, (start, stop) in enumerate(acc):
        a = _eval(start, env, f"{name}[{i}].start")
        b = _eval(stop, env, f"{name}[{i}].stop")
        dim_name, dim = dims[i]
        if bounds and not (0 <= a < b <= dim):
            why = ("is empty" if b <= a else f"runs past the end of {dim_name}={dim}" if b > dim
                   else "starts before 0")
            raise PlanError(
                f"{name} axis {i} ({dim_name}): the slice from {a} to {b} (stop exclusive) {why}, at {_fmt_env(env)}. "
                f"From '{start[0]}' and '{stop[0]}'. "
                + (_axis_fix(p, acc, i, dim_name, dim, min(b - a, dim), counts)
                   if p is not None and b > a else ""))
        got.append((a, b))
    return got


def _loop_vars(expr, loops):
    """The loop names an expression uses, in order of appearance."""
    return [n.id for n in ast.walk(ast.parse(expr[0], mode="eval"))
            if isinstance(n, ast.Name) and n.id in loops]


def _axis_fix(p, access, axis, dim_name, dim, width, counts=None):
    """The concrete change that makes tiles of `width` cover [0, dim) along one axis exactly once.

    The plan asks that feedback NAME the fix ("make the m loop M // 128"), not the symptom.
    """
    vs = _loop_vars(access[axis][0], p["loops"])
    v = vs[0] if vs else None
    ragged = dim % width != 0
    count = f"ceil({dim_name} / {width})" if ragged else f"{dim_name} // {width}"
    stop = (f"min(({v or 'i'}+1)*{width}, {dim_name})" if ragged else f"({v or 'i'}+1)*{width}")
    if v is None:
        return (f"Fix: add a loop, say `i`, that runs `{count}` times, and give this axis as "
                f'["i*{width}", "{stop}"].')
    now = f" (now `{p['loops'][v][0]}`, which is {counts[v]} here)" if counts and v in counts else ""
    return (f"Fix: make loop `{v}` run `{count}` times{now}, and give this axis as "
            f'["{v}*{width}", "{stop}"].')


def _limit(p, name, access, axis, size, cap, env, what, dim_name, dim, counts):
    if size > cap:
        raise PlanError(
            f"{name} tile is {size} on its {what} axis (axis {axis}, {dim_name}) at "
            f"{_fmt_env(env)}; the limit is {cap}. Split that axis into tiles of {cap}. "
            + _axis_fix(p, access, axis, dim_name, dim, cap, counts))


def _expr_stride(p, access, axis, shape):
    """How far apart consecutive tiles START, read from the start expression itself (works even when
    the shape has only one tile along this axis)."""
    vs = _loop_vars(access[axis][0], p["loops"])
    if not vs:
        return None
    base = {**shape, **{v: 0 for v in p["loops"]}}
    try:
        return (_eval(access[axis][0], {**base, vs[0]: 1}, "stride")
                - _eval(access[axis][0], base, "stride"))
    except PlanError:
        return None


def _stride(starts):
    """Smallest gap between consecutive tile starts along one axis, or None for a single tile."""
    xs = sorted(starts)
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    return min(gaps) if gaps else None


def _region(mask):
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    return int(len(idx)), lo, hi


def _tile_checks(p, env2, rows, cols, counts, M, K, N):
    """One (output tile, accumulation step): operand slices match the output tile and each other,
    stay inside their tensors, and respect the tile limits. Returns lhsT's (K slice, M slice)."""
    (r0, r1), (c0, c1) = rows, cols
    # Match against the output tile BEFORE bounds: a wrong loop variable also runs off
    # the edge, and "past the end of M" hides the cause, which is the wrong index.
    lhs_dims, rhs_dims = (("K", K), ("M", M)), (("K", K), ("N", N))
    (lk0, lk1), (m0, m1) = _slices(p["lhsT"], env2, "lhsT", lhs_dims, bounds=False)
    (rk0, rk1), (n0, n1) = _slices(p["rhs"], env2, "rhs", rhs_dims, bounds=False)
    if (m0, m1) != (r0, r1):
        raise PlanError(
            f"at {_fmt_env(env2)} the output tile is rows {r0}..{r1 - 1} but lhsT is read "
            f"at columns {m0}..{m1 - 1}. lhsT's second axis is M, so it must use the same "
            f"slice as the output rows: '{p['out'][0][0][0]}' to '{p['out'][0][1][0]}'.")
    if (n0, n1) != (c0, c1):
        raise PlanError(
            f"at {_fmt_env(env2)} the output tile is columns {c0}..{c1 - 1} but rhs is "
            f"read at columns {n0}..{n1 - 1}. rhs's second axis is N, so it must use the "
            f"same slice as the output columns: '{p['out'][1][0][0]}' to "
            f"'{p['out'][1][1][0]}'.")
    if (lk0, lk1) != (rk0, rk1):
        raise PlanError(
            f"at {_fmt_env(env2)} lhsT reads K rows {lk0}..{lk1 - 1} but rhs reads "
            f"{rk0}..{rk1 - 1}. Both operands must take the same K slice: use "
            f"lhsT's '{p['lhsT'][0][0][0]}' to '{p['lhsT'][0][1][0]}' for rhs too.")
    _slices(p["lhsT"], env2, "lhsT", lhs_dims, p=p, counts=counts)
    _slices(p["rhs"], env2, "rhs", rhs_dims, p=p, counts=counts)
    _limit(p, "lhsT", p["lhsT"], 0, lk1 - lk0, PMAX, env2, "partition", "K", K, counts)
    _limit(p, "lhsT", p["lhsT"], 1, m1 - m0, STAT_FMAX, env2, "free (stationary)",
           "M", M, counts)
    _limit(p, "rhs", p["rhs"], 0, rk1 - rk0, PMAX, env2, "partition", "K", K, counts)
    _limit(p, "rhs", p["rhs"], 1, n1 - n0, MOV_FMAX, env2, "free (moving)", "N", N,
           counts)
    return (lk0, lk1), (m0, m1)


TILE_FOR = dict(M=STAT_FMAX, K=PMAX, N=MOV_FMAX)   # the largest tile each dimension allows


def _cap_fix(p, counts, shape):
    """Name, per loop, the fix for loops that run once per ELEMENT instead of once per TILE."""
    fixes = []
    for v, expr in p["loops"].items():
        dims = [n.id for n in ast.walk(ast.parse(expr[0], mode="eval"))
                if isinstance(n, ast.Name) and n.id in SHAPE_NAMES]
        if not dims:
            continue
        d = dims[0]
        t = TILE_FOR[d]
        if counts[v] > max(1, shape[d] // t):
            fixes.append(f'loop `{v}` runs `{expr[0]}` times ({counts[v]} here), once per element of '
                         f'{d}; a loop should run once per TILE. Fix: make it "{d} // {t}" and give '
                         f'that axis as ["{v}*{t}", "({v}+1)*{t}"]')
    return ("; ".join(fixes) + ".") if fixes else ""


def check(p, M, K, N):
    """Check a parsed plan on one shape. Returns None if it is right, else the message."""
    shape = dict(M=M, K=K, N=N)
    try:
        counts = {}
        for v, expr in p["loops"].items():
            c = _eval(expr, shape, f"loops.{v}")
            if c < 0:
                raise PlanError(f"loops.{v} = '{expr[0]}' is {c} at {_fmt_env(shape)}")
            counts[v] = c
        if math.prod(counts.values()) > MAX_ITERATIONS:
            # Measured on seat 49: loops of "M", "K", "N" with whole-tensor slices. A bare cap message
            # named no fix and hid the swapped lhsT axes, and the model resent the same plan.
            env0 = {**shape, **{v: 0 for v in p["loops"]}}
            (r0, r1), (c0, c1) = _slices(p["out"], env0, "out", (("M", M), ("N", N)), p=p,
                                         counts=counts)
            _limit(p, "out", p["out"], 0, r1 - r0, PMAX, env0, "partition", "M", M, counts)
            _limit(p, "out", p["out"], 1, c1 - c0, MOV_FMAX, env0, "free (PSUM)", "N", N, counts)
            _tile_checks(p, env0, (r0, r1), (c0, c1), counts, M, K, N)
            raise PlanError(f"the loops run {math.prod(counts.values()):,} iterations at "
                            f"{_fmt_env(shape)}. " + (_cap_fix(p, counts, shape)
                                                      or "Each loop should run once per tile."))
        outer = [v for v in p["loops"] if v not in p["acc"]]
        written = np.zeros((M, N), np.int32)
        tile = None
        starts = (set(), set())   # distinct output tile starts along M and N

        for ovals in itertools.product(*(range(counts[v]) for v in outer)):
            env = {**shape, **dict(zip(outer, ovals))}
            (r0, r1), (c0, c1) = _slices(p["out"], env, "out", (("M", M), ("N", N)), p=p,
                                         counts=counts)
            _limit(p, "out", p["out"], 0, r1 - r0, PMAX, env, "partition", "M", M, counts)
            _limit(p, "out", p["out"], 1, c1 - c0, MOV_FMAX, env, "free (PSUM)", "N", N, counts)
            written[r0:r1, c0:c1] += 1
            tile = tile or (r1 - r0, c1 - c0)
            starts[0].add(r0)
            starts[1].add(c0)
            kcover = np.zeros(K, np.int32)
            kwidth = PMAX
            for avals in itertools.product(*(range(counts[v]) for v in p["acc"])):
                env2 = {**env, **dict(zip(p["acc"], avals))}
                (lk0, lk1), _ = _tile_checks(p, env2, (r0, r1), (c0, c1), counts, M, K, N)
                kwidth = lk1 - lk0
                kcover[lk0:lk1] += 1
            if not np.all(kcover == 1):
                bad = np.flatnonzero(kcover != 1)
                kind = "never read" if kcover[bad[0]] == 0 else f"read {kcover[bad[0]]} times"
                accs = ", ".join(f"{v} in range({counts[v]}) from '{p['loops'][v][0]}'"
                                 for v in p["acc"]) or "no accumulation loop"
                fix = (_axis_fix(p, p["lhsT"], 0, "K", K, kwidth, counts) if p["acc"] else
                       f"Fix: add an accumulation loop `k` that runs `K // {PMAX}` times, list it "
                       f'in accumulate_over, and give K as ["k*{PMAX}", "(k+1)*{PMAX}"] in lhsT and rhs.')
                raise PlanError(
                    f"for the output tile at {_fmt_env(env)}, K rows {bad[0]}..{bad[-1]} are "
                    f"{kind}, so the sum is wrong. The accumulation loop is {accs}; its K slices "
                    f"must cover 0..{K - 1} exactly once. " + fix)

        if not np.all(written == 1):
            gap = written == 0
            if gap.any():
                n, lo, hi = _region(gap)
                loops = ", ".join(f"{v} in range({counts[v]}) from '{p['loops'][v][0]}'"
                                  for v in outer)
                msg = (f"at {_fmt_env(shape)}, {n:,} output elements are never written, in rows "
                       f"{lo[0]}..{hi[0]} and columns {lo[1]}..{hi[1]}. The output tiles come "
                       f"from {loops}" + (f", and are {tile[0]} x {tile[1]}." if tile else "."))
                zero = [v for v in outer if counts[v] == 0]
                if zero:
                    v = zero[0]
                    msg += (f" Loop `{v}` runs 0 times here ('{p['loops'][v][0]}'), so nothing is "
                            f"written. Fix: round the count up, e.g. `ceil(M / 128)`, and clamp each "
                            f"stop with min(..., M) so the last, partial tile stays inside.")
                elif tile:
                    # Rows that are never written mean the M loop falls short; otherwise whole
                    # columns are missing and the N loop does.
                    axis = 0 if gap.all(axis=1).any() else 1
                    dim_name, dim = (("M", M), ("N", N))[axis]
                    st = _stride(starts[axis]) or _expr_stride(p, p["out"], axis, shape)
                    if st and st > tile[axis]:
                        # Tiles are placed far enough apart but are too narrow: an off-by-one stop.
                        v = (_loop_vars(p["out"][axis][0], p["loops"]) or ["i"])[0]
                        msg += (f" Tiles along {dim_name} start {st} apart but are only "
                                f"{tile[axis]} wide, so {st - tile[axis]} of every {st} are skipped. "
                                f"Fix: make the stop `({v}+1)*{st}` (it is "
                                f"'{p['out'][axis][1][0]}').")
                    else:
                        msg += " " + _axis_fix(p, p["out"], axis, dim_name, dim, tile[axis], counts)
                raise PlanError(msg)
            n, lo, hi = _region(written > 1)
            msg = (f"at {_fmt_env(shape)}, {n:,} output elements are written more than once, in "
                   f"rows {lo[0]}..{hi[0]} and columns {lo[1]}..{hi[1]}: two output tiles overlap, "
                   f"because consecutive tiles start closer together than they are wide.")
            if tile:
                axis = 0 if hi[0] - lo[0] + 1 < M else 1
                dim_name, dim = (("M", M), ("N", N))[axis]
                msg += " " + _axis_fix(p, p["out"], axis, dim_name, dim, tile[axis], counts)
            raise PlanError(msg)
    except PlanError as e:
        return str(e)
    return None


# ---------------------------------------------------------------- shapes

def _contract():
    # reference_level4.py's contract: multiples of the tile sizes, up to 4 tiles each way.
    return dict(M=st.integers(1, 4).map(lambda t: 128 * t),
                K=st.integers(1, 4).map(lambda t: 128 * t),
                N=st.integers(1, 2).map(lambda t: 512 * t))


def _beyond():
    return dict(M=st.integers(1, 600), K=st.integers(1, 600), N=st.integers(1, 1100))


def search(p, examples=300, beyond=False):
    """Returns dict(ok, message, shape, tried, seconds)."""
    stats = dict(tried=0, fail=None)
    t0 = time.time()

    @settings(max_examples=examples, derandomize=True, database=None, deadline=None,
              phases=(Phase.generate, Phase.shrink), report_multiple_bugs=False,
              print_blob=False, suppress_health_check=list(HealthCheck))
    @given(**(_beyond() if beyond else _contract()))
    def prop(M, K, N):
        stats["tried"] += 1
        m = check(p, M, K, N)
        if m is not None:
            stats["fail"] = ((M, K, N), m)
            raise AssertionError(m)

    try:
        prop()
        ok = True
    except AssertionError:
        ok = False
    secs = time.time() - t0
    scope = "ragged shapes up to M,K 600 and N 1100" if beyond else \
        "the level's contract (multiples of the tile sizes, up to 4 tiles)"
    if ok:
        return dict(ok=True, shape=None, tried=stats["tried"], seconds=secs,
                    message=f"PLAN OK on {stats['tried']} generated shapes across {scope}.")
    (M, K, N), m = stats["fail"]
    return dict(ok=False, shape=(M, K, N), tried=stats["tried"], seconds=secs,
                message=f"PLAN FAILS. Smallest failing shape found: M={M} K={K} N={N}.\n  {m}")


def fixed_shapes(p):
    """What the 4 fixed nkibench shapes alone would say: (passed, total)."""
    shapes = nkibench.LEVELS[4]["shapes"]
    return sum(check(p, s["M"], s["K"], s["N"]) is None for s in shapes), len(shapes)


# ---------------------------------------------------------------- selftest

def _plan(m="M // 128", n="N // 512", k="K // 128", lhsT=None, rhs=None, out=None, acc=("k",)):
    return {"loops": {"m": m, "n": n, "k": k}, "accumulate_over": list(acc),
            "reads": {"lhsT": lhsT or [["k*128", "(k+1)*128"], ["m*128", "(m+1)*128"]],
                      "rhs": rhs or [["k*128", "(k+1)*128"], ["n*512", "(n+1)*512"]]},
            "writes": {"out": out or [["m*128", "(m+1)*128"], ["n*512", "(n+1)*512"]]}}


# Read off reference_level4.py: the shipped answer, as a plan.
REFERENCE = _plan()

# The same loops made ragged-safe: ceil counts, every stop clamped.
RAGGED = _plan("ceil(M / 128)", "ceil(N / 512)", "ceil(K / 128)",
               lhsT=[["k*128", "min((k+1)*128, K)"], ["m*128", "min((m+1)*128, M)"]],
               rhs=[["k*128", "min((k+1)*128, K)"], ["n*512", "min((n+1)*512, N)"]],
               out=[["m*128", "min((m+1)*128, M)"], ["n*512", "min((n+1)*512, N)"]])

# Mutants, each one a mistake the model makes or could make. (name, plan, what the message must say)
MUTANTS = [
    ("one tile for the whole tensor (the level-4 wall)",
     _plan("1", "1", "1", lhsT=[["0", "K"], ["0", "M"]], rhs=[["0", "K"], ["0", "N"]],
           out=[["0", "M"], ["0", "N"]]), "the limit is"),
    ("off-by-one stop on the output columns",
     _plan(rhs=[["k*128", "(k+1)*128"], ["n*512", "(n+1)*512 - 1"]],
           out=[["m*128", "(m+1)*128"], ["n*512", "(n+1)*512 - 1"]]), "never written"),
    ("forgot to loop over K",
     _plan(k="1"), "never read"),
    ("lhsT indexed by the wrong loop",
     _plan(lhsT=[["k*128", "(k+1)*128"], ["n*128", "(n+1)*128"]]), "same slice as the output"),
    ("output tiles overlap (stride 64, tile 128)",
     _plan(m="M // 64 - 1", lhsT=[["k*128", "(k+1)*128"], ["m*64", "m*64 + 128"]],
           out=[["m*64", "m*64 + 128"], ["n*512", "(n+1)*512"]]), "more than once"),
]


def selftest(examples):
    print("Proving the plan checker before trusting it.\n")
    rc = 0

    def line(what, ok, extra=""):
        nonlocal rc
        rc |= 0 if ok else 1
        print(f"  {what:<52} -> {'ok' if ok else 'FAIL'}{'  ' + extra if extra else ''}")

    # A 1-D tile is caught before any shape is tried.
    try:
        parse_plan(_plan(lhsT=[["k*128", "(k+1)*128"]]))
        line("rejects a 1-D tile statically (the level-1 wall)", False)
    except PlanError as e:
        line("rejects a 1-D tile statically (the level-1 wall)", "2-D" in str(e))
    try:
        parse_plan(_plan(m="__import__('os')"))
        line("rejects code that is not arithmetic", False)
    except PlanError:
        line("rejects code that is not arithmetic", True)

    print()
    r = search(parse_plan(REFERENCE), examples)
    line("reference plan passes the level's contract", r["ok"],
         f"({r['tried']} shapes, {r['seconds'] * 1000:.0f} ms)")
    r = search(parse_plan(REFERENCE), examples, beyond=True)
    line("reference plan FAILS ragged shapes, as it should", not r["ok"],
         f"(shrunk to M={r['shape'][0]} K={r['shape'][1]} N={r['shape'][2]})" if r["shape"] else "")
    ragged_msg = r["message"]
    r = search(parse_plan(RAGGED), examples, beyond=True)
    line("ceil + min plan passes ragged shapes too", r["ok"],
         f"({r['tried']} shapes, {r['seconds'] * 1000:.0f} ms)")
    if not r["ok"]:
        print(textwrap.indent(r["message"], "      "))

    print("\n  mutants: does the 4-shape nkibench set catch it, and does the search?\n")
    for name, plan, must in MUTANTS:
        p = parse_plan(plan)
        passed, total = fixed_shapes(p)
        r = search(p, examples)
        ok = (not r["ok"]) and must in r["message"] and ("Fix:" in r["message"]
                                                        or "must use the same slice" in r["message"])
        line(name, ok, f"fixed shapes {passed}/{total} pass; search shrinks to "
             f"M={r['shape'][0]} K={r['shape'][1]} N={r['shape'][2]}" if r["shape"] else "")
        if not ok:
            print(textwrap.indent(r["message"], "      "))

    print("\n  what the agent is told for the reference plan on ragged shapes:\n")
    print(textwrap.indent(ragged_msg, "    "))
    print("\n  ...and for one tile over the whole tensor:\n")
    print(textwrap.indent(search(parse_plan(MUTANTS[0][1]), examples)["message"], "    "))

    print("\nPLAN SELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", metavar="PLAN.json")
    ap.add_argument("--examples", type=int, default=300)
    ap.add_argument("--beyond-contract", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.examples))
    if not a.plan:
        ap.error("give --selftest or --plan PLAN.json")
    try:
        p = parse_plan(json.load(open(a.plan)))
    except (PlanError, json.JSONDecodeError) as e:
        print(f"PLAN REJECTED: {e}")
        sys.exit(2)
    r = search(p, a.examples, a.beyond_contract)
    print(r["message"])
    print(f"({r['tried']} shapes, {r['seconds'] * 1000:.0f} ms)")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
