#!/usr/bin/env python3
"""
level0_heatrod.py — the warm-up. Heat along a rod with BOTH ends held at zero, and a starting
shape built only from sine waves.

Every problem here has an exact closed-form answer: each sine wave keeps its shape and decays
at its own rate. Qwen3-8B solves all three of these on the first attempt, which is the point
of a level 0 -- it proves the loop works before anything is hard. Go to level 1 for problems
the model actually has to iterate on.

    python level0_heatrod.py --list
    python level0_heatrod.py --show 3 --answer
    python level0_heatrod.py --selftest
"""

import argparse
import random
import sys

import sympy as sp

import pdecheck
from pdecheck import x, t

LEVEL = 0
TITLE = "both ends held at zero, starting shape built from sine waves"
SUBS = (1, 2, 3)


def make(sub, seed=0):
    r = random.Random(seed * 100 + sub)
    if sub == 1:
        L, k, modes = sp.Integer(1), sp.Integer(r.choice([1, 2, 3])), [(r.choice([1, 2, 3]), 1)]
    elif sub == 2:
        L = sp.Integer(r.choice([2, 3, 4]))
        k = sp.Rational(r.choice([1, 2, 3]), r.choice([2, 4, 5]))
        modes = [(r.choice([1, 2, 3]), r.choice([1, 2, 5]))]
    elif sub == 3:
        L = sp.Integer(r.choice([1, 2, 3]))
        k = sp.Rational(r.choice([1, 3]), r.choice([1, 2, 4]))
        ns = r.sample([1, 2, 3, 4, 5], r.choice([2, 3]))
        modes = [(n, r.choice([1, 2, 3, -1, -2])) for n in sorted(ns)]
    else:
        raise ValueError(f"level 0 has sub-problems {SUBS}")

    f = sum(a * sp.sin(n * sp.pi * x / L) for n, a in modes)
    exact = sum(a * sp.exp(-k * (n * sp.pi / L) ** 2 * t) * sp.sin(n * sp.pi * x / L)
                for n, a in modes)
    return dict(name=f"level0.{sub}", level=0, sub=sub, seed=seed,
                L=L, k=k, left="dirichlet", right="dirichlet",
                f=f, exact=exact, tol=1e-6,
                basis=lambda n, L=L: sp.sin(n * sp.pi * x / L),
                lam=lambda n, L=L: n * sp.pi / L)


def selftest():
    print(f"level {LEVEL}: {TITLE}\n")
    p = make(1)
    n = [m for m in range(1, 4) if p["f"].has(sp.sin(m * sp.pi * x))][0]
    k = p["k"]
    cases = [
        ("exact, as a model writes it", p,
         f"reasoning...\nu(x, t) = exp(-{sp.sstr(k)}*{n}**2*pi**2*t)*sin({n}*pi*x)", 1.0),
        ("wrong decay rate", p, f"exp(-{sp.sstr(k)}*t)*sin({n}*pi*x)", 0.6),
        ("forgot the decay entirely", p, f"sin({n}*pi*x)", 0.6),
        ("cosine instead of sine", p,
         f"exp(-{sp.sstr(k)}*{n}**2*pi**2*t)*cos({n}*pi*x)", 0.4),
        ("no answer line", p, "I think it decays exponentially.", 0.0),
        ("left a symbol in", p, "exp(-k*t)*sin(pi*x)", 0.0),
    ]
    rc = pdecheck.verify([make(s, seed) for s in SUBS for seed in range(5)], cases)
    print("\n  exact answers score 1.0 on all 15 generated problems" if not rc else "")
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
            print(f"\nknown answer: u(x, t) = {sp.sstr(p['exact'])}")
        return
    print(f"level {LEVEL}: {TITLE}")
    for sub in SUBS:
        p = make(sub, a.seed)
        print(f"  {sub}. u_t = {sp.sstr(p['k'])} u_xx on [0, {sp.sstr(p['L'])}], "
              f"u(x,0) = {sp.sstr(p['f'])}")


if __name__ == "__main__":
    main()
