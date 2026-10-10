#!/usr/bin/env python3
"""solver_tool.py -- the model names the waves; sympy does the rest.

The model chooses the eigenfunction family X_n(x) -- the step that encodes the boundary conditions.
This tool derives each decay rate from that family, projects the model's own starting shape onto it,
keeps enough terms, and writes u(x, t). It never reads the problem: a wrong family or a mis-copied k
gets a faithfully computed wrong answer, and the checker says why.
"""
import concurrent.futures as cf
import json
import re
import sys

import numpy as np
import sympy as sp

import pdecheck
from pdecheck import x, t

n = sp.Symbol("n", integer=True, positive=True)
MAX_TERMS, TARGET_ERR, SCAN = 15, 1e-3, 60

FIELDS = ("pde", "method", "k", "L", "initial", "left", "right", "eigenfunction", "n_terms")
_CANON = {f.lower(): f for f in FIELDS}

FAMILIES = {  # control arm T3 only
    ("dirichlet", "dirichlet"): "sin(n*pi*x/L)",
    ("dirichlet", "neumann"): "sin((2*n - 1)*pi*x/(2*L))",
    ("neumann", "dirichlet"): "cos((2*n - 1)*pi*x/(2*L))",
    ("neumann", "neumann"): "cos((n - 1)*pi*x/L)",
}

INSTRUCTIONS = (
    "Do not compute coefficients or decay rates yourself: a solver does that. Your job is to read the\n"
    "problem and choose the method. Reply with one JSON object and nothing after it:\n"
    '{"pde": "heat", "method": "eigenfunction_expansion",\n'
    ' "k": "<conductivity>", "L": "<rod length>", "initial": "<u(x, 0)>",\n'
    ' "left": "dirichlet or neumann", "right": "dirichlet or neumann",\n'
    ' "eigenfunction": "<X_n(x), the n-th allowed wave, in n and x, for n = 1, 2, 3, ...>"}\n'
    "Write expressions with sin, cos, exp, sqrt, pi, * and ** only (no np. or math.).\n"
    "The solver finds each wave's decay rate from your eigenfunction, works out every coefficient\n"
    "from your starting shape, keeps enough terms, and writes u(x, t)."
)


INSTRUCTIONS_DERIVED = (
    "Do not compute anything yourself: a solver does the mathematics. Your job is to read the problem\n"
    "and identify its structure. Reply with one JSON object and nothing after it:\n"
    '{"pde": "heat", "method": "eigenfunction_expansion",\n'
    ' "k": "<conductivity>", "L": "<rod length>", "initial": "<u(x, 0)>",\n'
    ' "left": "dirichlet or neumann", "right": "dirichlet or neumann"}\n'
    "dirichlet means the temperature u is held at 0 at that end; neumann means the end is insulated\n"
    "(the slope u_x is 0). Write expressions with sin, cos, exp, sqrt, pi, * and ** only (no np. or math.).\n"
    "The solver derives the allowed waves from your end conditions, their decay rates, every\n"
    "coefficient from your starting shape, and writes u(x, t)."
)


def instructions(mode):
    return INSTRUCTIONS_DERIVED if mode == "full" else INSTRUCTIONS


def parse_spec(text):
    """-> (spec dict, None) or (None, reason). Takes the last flat JSON object that has an eigenfunction."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for raw in reversed(re.findall(r"\{[^{}]*\}", text)):
        for fix in (lambda s: s, lambda s: s.replace("'", '"'), lambda s: re.sub(r",\s*}", "}", s)):
            try:
                d = json.loads(fix(raw))
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and ("eigenfunction" in d or "left" in d or "right" in d):
                return {_CANON.get(str(k).strip().lower(), str(k)): str(v).strip() for k, v in d.items()}, None
    return None, ("No spec found. Reply with one JSON object with the fields pde, method, k, L, "
                  "initial, left, right, eigenfunction.")


def _expr(s, L=None):
    s = str(s).replace("^", "**").replace("π", "pi")
    s = re.sub(r"\b(?:np|numpy|math|sp|sympy)\.", "", s)       # np.sin -> sin: harness tolerance
    e = pdecheck.parse(s).subs(sp.Symbol("n"), n)
    return e.subs(sp.Symbol("L"), L) if L is not None else e


def _coeff(f, X, L):
    num, den = sp.integrate(f * X, (x, 0, L)), sp.integrate(X ** 2, (x, 0, L))
    if num.has(sp.Integral) or den.has(sp.Integral):          # SciPy fallback: numeric projection
        from scipy.integrate import quad
        fn, Xn = sp.lambdify(x, f), sp.lambdify(x, X)
        num = quad(lambda s: fn(s) * Xn(s), 0, float(L), limit=200)[0]
        den = quad(lambda s: Xn(s) ** 2, 0, float(L), limit=200)[0]
        return sp.Float(num / den) if den else sp.Integer(0)
    return sp.nsimplify(sp.simplify(num / den)) if den != 0 else sp.Integer(0)


def solve(spec, mode="ansatz"):
    """-> (answer line or None, one-line report)."""
    if spec.get("pde", "heat").lower() not in ("heat", "heat equation", "diffusion"):
        return None, f"unsupported pde {spec.get('pde')!r}: only the heat equation"
    field = "k"
    try:
        k = _expr(spec["k"])
        field = "L"
        L = _expr(spec["L"])
        field = "initial"
        f = _expr(spec["initial"])
        field = "eigenfunction"
        if mode == "full":     # derived from the stated end conditions, never from the problem
            field = "left/right"
            fam = FAMILIES[(spec["left"].strip().lower(), spec["right"].strip().lower())]
        else:
            fam = spec["eigenfunction"]
        X = _expr(fam, L=L)
    except KeyError as e:
        if field == "left/right":
            return None, "left and right must each be dirichlet or neumann"
        return None, f"the spec is missing {e}"
    except Exception as e:
        return None, (f"could not read the {field!r} field ({type(e).__name__}); write it with "
                      f"sin, cos, exp, sqrt, pi, * and ** only")
    if k.free_symbols or L.free_symbols or f.free_symbols - {x} or X.free_symbols - {x, n}:
        return None, "k and L must be numbers, initial a function of x, eigenfunction a function of n and x"
    notes = consistency(spec, X, L) if mode == "ansatz" else ""
    mu2 = sp.simplify(-sp.diff(X, x, 2) / X)
    if mu2.has(x):
        return None, f"{fam} is not an eigenfunction of d2/dx2: it must be a sine or cosine in x"

    grid = np.linspace(0, float(L), 801)
    f0 = np.broadcast_to(np.asarray(sp.lambdify(x, f, "numpy")(grid), dtype=float), grid.shape)
    norm = float(np.sqrt(np.trapezoid(f0 ** 2, grid))) or 1.0
    approx, terms, err = np.zeros_like(grid), [], float("inf")
    for m in range(1, SCAN + 1):
        Xm = X.subs(n, m)
        if Xm == 0:
            continue
        c = _coeff(f, Xm, L)
        if c == 0 or abs(float(c)) < 1e-14:
            continue
        rate = sp.simplify(k * mu2.subs(n, m))
        terms.append(c * sp.exp(-rate * t) * Xm)
        approx = approx + float(c) * np.broadcast_to(
            np.asarray(sp.lambdify(x, Xm, "numpy")(grid), dtype=float), grid.shape)
        err = float(np.sqrt(np.trapezoid((approx - f0) ** 2, grid))) / norm
        if err < TARGET_ERR or len(terms) >= MAX_TERMS:
            break
    if not terms:
        return None, f"every coefficient of your initial shape on {fam} is zero"
    u = sp.Add(*terms)
    return f"u(x, t) = {sp.sstr(u)}", (f"{len(terms)} term(s) of {fam}; starting-shape error against "
                                       f"your initial condition {err * 100:.3g}%." + notes)


def consistency(spec, X, L):
    """Check the model's eigenfunction against the end conditions the model itself stated.
    A statement about its own spec, never about the problem: if it states the wrong conditions,
    this check passes and the checker still fails the answer."""
    out = []
    for side, at in (("left", 0), ("right", L)):
        kind = str(spec.get(side, "")).lower()
        if kind not in ("dirichlet", "neumann"):
            continue
        expr = X if kind == "dirichlet" else sp.diff(X, x)
        val = sp.simplify(expr.subs(x, at))
        if val != 0:
            what = "X_n" if kind == "dirichlet" else "the slope X_n'"
            out.append(f" You stated the {side} end is {kind}, but {what} at x = {sp.sstr(at)} is "
                       f"{sp.sstr(val)}, which is not 0 for every n, so your eigenfunction does not "
                       f"satisfy that end condition.")
    return "".join(out)


_POOL = None


def shutdown():
    """Stop the worker processes; agent.py calls this before it exits."""
    if _POOL is not None:
        for proc in list(getattr(_POOL, "_processes", {}).values()):
            proc.terminate()


def run(spec, mode="ansatz", timeout=20):
    """solve() in a worker process, so a pathological spec cannot hang a round."""
    global _POOL
    _POOL = _POOL or cf.ProcessPoolExecutor(max_workers=4)
    try:
        return _POOL.submit(solve, spec, mode).result(timeout=timeout)
    except cf.TimeoutError:
        return None, f"the solver gave up after {timeout} s"


def selftest():
    import time
    import level0_heatrod as L0
    import level1_heatrod as L1

    bad = 0
    corpus = [  # (reply text, should it parse, reward it should end with)
        ('```json\n{"pde": "heat", "k": "2", "L": "2", "initial": "x*(2-x)", "left": "dirichlet", '
         '"right": "dirichlet", "eigenfunction": "sin(n*pi*x/2)"}\n```', True, 1.0),
        ('Both ends are fixed, so sine waves.\n{"k": "2", "L": "2", "initial": "x*(2 - x)", '
         '"left": "dirichlet", "right": "dirichlet", "eigenfunction": "sin(n*pi*x/2)",}', True, 1.0),
        ("{'k': '2', 'L': '2', 'initial': 'x*(2-x)', 'left': 'dirichlet', 'right': 'dirichlet', "
         "'eigenfunction': 'sin(n*pi*x/2)'}", True, 1.0),
        ('First try {"eigenfunction": "cos(n*x)"} then\n**Final:** {"k": "2", "L": "2", '
         '"initial": "x*(2-x)", "left": "dirichlet", "right": "dirichlet", "eigenfunction": "sin(n*pi*x/2)"}', True, 1.0),
        ("u(x, t) = exp(-2*pi**2*t)*sin(pi*x)", False, None),
        ('{"k": 2, "L": 2, "initial": "x*(2-x)", "left": "dirichlet", "right": "dirichlet", '
         '"eigenfunction": "\\\\sin(n\\\\pi x/2)"}', True, None),   # parses; the solver must refuse LaTeX
    ]
    p13 = L1.make(3, 0)
    for text, want, want_r in corpus:
        spec, err = parse_spec(text)
        got = spec is not None
        line, rep = solve(spec) if spec else (None, err)
        r = pdecheck.check(p13, line)["reward"] if line else None
        bad += (got != want) + (r != want_r)
        print(f"  parse={'ok ' if got else 'no '} want={'ok ' if want else 'no '}  reward={r} (want {want_r})  {rep[:70]}")

    for mode in ("ansatz", "full"):
        for mod in (L0, L1):
            for sub in mod.SUBS:
                for seed in range(5):
                    p = mod.make(sub, seed)
                    Ls = sp.sstr(p["L"])
                    fam = (f"sin((2*n - 1)*pi*x/(2*{Ls}))" if p["right"] == "neumann" else f"sin(n*pi*x/{Ls})")
                    spec = dict(pde="heat", k=sp.sstr(p["k"]), L=Ls, initial=sp.sstr(p["f"]),
                                left=p["left"], right=p["right"], eigenfunction=fam)
                    t0 = time.perf_counter()
                    line, rep = run(spec, mode)
                    r = pdecheck.check(p, line)["reward"] if line else 0.0
                    if r != 1.0:
                        bad += 1
                        print("  FAIL", mode, p["name"], seed, r, rep)
        print(f"  mode {mode}: 30 problems checked")
    # the neumann-neumann family the levels never use: cos((n-1)*pi*x/L), constant mode included
    pnn = dict(L1.make(1, 0), left="neumann", right="neumann", f=1 + sp.cos(sp.pi * x), exact=None, tol=1e-6)
    line, rep = solve(dict(k="1", L="1", initial="1 + cos(pi*x)", left="neumann", right="neumann",
                           eigenfunction="cos((n - 1)*pi*x)"))
    r = pdecheck.check(pnn, line)
    print("  neumann-neumann with constant mode:", r["reward"], r["parts"], line)
    bad += r["reward"] != 1.0
    print("\nSELFTEST " + ("PASSED" if bad == 0 else f"FAILED ({bad})"))
    _POOL and _POOL.shutdown()
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(solve(json.loads(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "ansatz"))
