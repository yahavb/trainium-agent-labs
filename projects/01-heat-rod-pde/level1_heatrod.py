#!/usr/bin/env python3
"""
level1_heatrod.py — the real problems. Two things change from level 0, and each breaks a
habit the model has.

  1. INSULATED END. One end has zero slope instead of zero temperature, so no heat leaves
     there. The allowed shapes are no longer sin(n*pi*x/L): they are sin((2n-1)*pi*x/(2*L)),
     with HALF-INTEGER frequencies, because the slope has to vanish at the far end rather
     than the value. A model that reaches for sin(n*pi*x/L) out of habit satisfies the
     equation and the near end and fails the insulated end, which is exactly the partial
     credit the checker is built to report.

  2. STARTING SHAPE THAT IS NOT A SINE WAVE. Sub-problem 3 starts from a parabola. No finite
     expression solves it: the true answer is an infinite sum, and every term has to be
     computed rather than read off the question. The model must recognise the pattern, get
     the coefficients right, and keep enough terms -- so the checker accepts a truncation
     whose starting shape is within a stated tolerance, and reports the error so the agent
     knows whether to add terms or fix coefficients.

    python level1_heatrod.py --list
    python level1_heatrod.py --show 3 --answer
    python level1_heatrod.py --selftest
"""

import argparse
import random
import sys

import sympy as sp

import pdecheck
from pdecheck import x, t

LEVEL = 1
TITLE = "one insulated end, and starting shapes that are not sine waves"
SUBS = (1, 2, 3)

# Sub-problem 3 keeps 3 odd terms at this tolerance and only 2 at 1%, so the bar is what
# decides how much work the model has to do. Measured in --selftest, not guessed.
SERIES_TOL = 0.005


def _insulated(L, k, modes):
    """Left end held at zero, right end insulated. Allowed frequencies are odd multiples of
    pi/(2L): sin(mu*x) is zero at x=0, and its slope mu*cos(mu*L) vanishes only when
    mu*L is an odd multiple of pi/2."""
    f = exact = 0
    for j, a in modes:
        mu = (2 * j - 1) * sp.pi / (2 * L)
        f += a * sp.sin(mu * x)
        exact += a * sp.exp(-k * mu ** 2 * t) * sp.sin(mu * x)
    return f, exact


def make(sub, seed=0):
    r = random.Random(1000 + seed * 100 + sub)
    if sub == 1:
        L = sp.Integer(r.choice([1, 2]))
        k = sp.Integer(r.choice([1, 2]))
        f, exact = _insulated(L, k, [(r.choice([1, 2]), 1)])
        left, right, tol = "dirichlet", "neumann", 1e-6
        basis = lambda n, L=L: sp.sin((2 * n - 1) * sp.pi * x / (2 * L))
        lam = lambda n, L=L: (2 * n - 1) * sp.pi / (2 * L)
    elif sub == 2:
        L = sp.Integer(r.choice([1, 2, 3]))
        k = sp.Rational(r.choice([1, 2, 3]), r.choice([1, 2, 4]))
        js = r.sample([1, 2, 3], 2)
        f, exact = _insulated(L, k, [(j, r.choice([1, 2, -1])) for j in sorted(js)])
        left, right, tol = "dirichlet", "neumann", 1e-6
        basis = lambda n, L=L: sp.sin((2 * n - 1) * sp.pi * x / (2 * L))
        lam = lambda n, L=L: (2 * n - 1) * sp.pi / (2 * L)
    elif sub == 3:
        # A parabola, zero at both ends so it is compatible with the boundaries. Its sine
        # coefficients are 8*L^2/(n^3*pi^3) for odd n and zero for even n.
        L = sp.Integer(r.choice([1, 2]))
        k = sp.Rational(r.choice([1, 2]), r.choice([1, 2]))
        f, exact = x * (L - x), None
        left, right, tol = "dirichlet", "dirichlet", SERIES_TOL
        basis = lambda n, L=L: sp.sin(n * sp.pi * x / L)
        lam = lambda n, L=L: n * sp.pi / L
    else:
        raise ValueError(f"level 1 has sub-problems {SUBS}")

    return dict(name=f"level1.{sub}", level=1, sub=sub, seed=seed,
                L=L, k=k, left=left, right=right, f=sp.expand(f), exact=exact, tol=tol,
                basis=basis, lam=lam)


def series_answer(p, n_terms):
    """The truncated answer to sub-problem 3, for the selftest: how many terms are enough."""
    L, k = p["L"], p["k"]
    out = 0
    for n in range(1, 2 * n_terms, 2):
        b = 8 * L ** 2 / (n ** 3 * sp.pi ** 3)
        out += b * sp.exp(-k * (n * sp.pi / L) ** 2 * t) * sp.sin(n * sp.pi * x / L)
    return out


def selftest():
    print(f"level {LEVEL}: {TITLE}\n")
    rc = 0

    p1 = make(1)
    L, k = p1["L"], p1["k"]
    mu = sp.pi / (2 * L)
    j = 1 if p1["f"].has(sp.sin(mu * x)) else 2
    mu = (2 * j - 1) * sp.pi / (2 * L)
    trap = sp.exp(-k * (sp.pi / L) ** 2 * t) * sp.sin(sp.pi * x / L)

    cases = [
        ("insulated: exact", p1, sp.sstr(p1["exact"]), 1.0),
        ("insulated: used sin(n*pi*x/L)", p1, sp.sstr(trap), 0.6),
        # Flat in x, so its slope vanishes everywhere and it earns the insulated end
        # honestly while failing everything else. 0.2, not 0.4.
        ("insulated: decay but no wave", p1,
         sp.sstr(sp.exp(-k * mu ** 2 * t)), 0.2),
    ]
    rc |= pdecheck.verify([make(s, seed) for s in SUBS for seed in range(5)], cases)

    print()
    p3 = make(3)
    for n_terms, want_solved in ((1, False), (2, False), (3, True), (5, True)):
        r = pdecheck.check(p3, sp.sstr(series_answer(p3, n_terms)))
        solved = r["reward"] == 1.0
        mark = "ok" if solved == want_solved else "FAIL"
        rc |= 0 if solved == want_solved else 1
        print(f"  parabola with {n_terms} term(s): reward {r['reward']:.1f}, "
              f"starting shape off by {r['start_error'] * 100:.2f}%  "
              f"{'solved' if solved else 'not solved'}  {mark}")
    print(f"  ^ the {SERIES_TOL * 100:g}% bar is what forces 3 terms rather than 1.")

    r = pdecheck.check(p3, sp.sstr(series_answer(p3, 3) * 2))
    rc |= 0 if r["reward"] < 1.0 else 1
    print(f"\n  parabola, coefficients doubled: reward {r['reward']:.1f}  "
          f"{'ok' if r['reward'] < 1.0 else 'FAIL — wrong coefficients accepted'}")

    print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", type=int, metavar="SUB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.show:
        p = make(a.show, a.seed)
        print(pdecheck.prompt_of(p))
        if a.answer:
            known = p["exact"] if p["exact"] is not None else series_answer(p, 4)
            label = "known answer" if p["exact"] is not None else "4-term truncation"
            print(f"\n{label}: u(x, t) = {sp.sstr(known)}")
        return
    print(f"level {LEVEL}: {TITLE}")
    for sub in SUBS:
        p = make(sub, a.seed)
        ends = f"{p['left']}/{p['right']}"
        print(f"  {sub}. u_t = {sp.sstr(p['k'])} u_xx on [0, {sp.sstr(p['L'])}], {ends}, "
              f"u(x,0) = {sp.sstr(p['f'])}")


if __name__ == "__main__":
    main()
