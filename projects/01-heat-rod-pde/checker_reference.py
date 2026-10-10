#!/usr/bin/env python3
"""
pdecheck.py — the shared checker. Turns a candidate u(x, t) into a reward and a readable
reason, for every level.

It never looks at the known answer. It plugs the candidate into the equation, into each
boundary condition, and into the starting shape, and scores those four things separately:

    0.4  the equation holds
    0.2  the left boundary condition holds
    0.2  the right boundary condition holds
    0.2  the starting shape matches
    ---
    1.0  solved

A problem is a dict:

    L, k          rod length and conductivity (sympy numbers)
    left, right   "dirichlet" (u = 0) or "neumann" (u_x = 0, an insulated end)
    f             the starting shape, a sympy expression in x
    exact         a known closed form, or None when the true answer is an infinite series
    tol           relative tolerance for the starting-shape match

`tol` is the interesting field. When the true solution is an infinite series the model can
only write finitely many terms, so the starting shape can never match exactly. The equation
and both boundaries still hold exactly for any truncation -- only the starting shape reveals
how many terms were kept. So tol decides how many terms the problem demands, and the feedback
always reports the actual error so an agent knows whether to add terms or fix coefficients.
"""

import re

import numpy as np
import sympy as sp

x, t = sp.symbols("x t", real=True)

WEIGHTS = dict(equation=0.4, left_bc=0.2, right_bc=0.2, start_shape=0.2)

# When True the checker prints the target coefficients. Useful for a teaching walkthrough,
# and ruinous as a reward signal: the model copied them out of the message verbatim and
# solved the parabola without computing a single coefficient. Default off; see _coeff_report.
REVEAL_COEFFICIENTS = False

_ALLOWED = {"exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "pi": sp.pi, "sqrt": sp.sqrt,
            "sinh": sp.sinh, "cosh": sp.cosh, "tan": sp.tan, "E": sp.E, "x": x, "t": t}


def extract(text):
    """Pull the expression out of the model's reply."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    hits = re.findall(r"u\s*\(\s*x\s*,\s*t\s*\)\s*=\s*(.+)", text)
    if not hits:
        return None
    s = hits[-1].strip().strip("`$ ").rstrip(".")
    return s.replace("^", "**").replace("\\pi", "pi").replace("π", "pi")


def parse(s):
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application)
    tr = standard_transformations + (implicit_multiplication_application,)
    # sympy's parser emits Integer/Float/Rational calls, so those names must be reachable;
    # __builtins__ stays empty so a reply cannot reach anything else.
    glob = {"__builtins__": {}}
    glob.update({n: getattr(sp, n) for n in ("Integer", "Float", "Rational", "Symbol")})
    return parse_expr(s, local_dict=dict(_ALLOWED), global_dict=glob, transformations=tr)


def basis_label(p, n="n"):
    """How the problem's own waves are written, with n left symbolic."""
    L = sp.sstr(p["L"])
    if p["right"] == "neumann":
        return f"sin((2*{n} - 1)*pi*x/(2*{L}))"
    if p["left"] == "neumann":
        return f"cos(({n} - 1)*pi*x/{L})"
    return f"sin({n}*pi*x/{L})"


def coeff_method(p):
    """The standard formula for the coefficients.

    Saying this is not giving the answer: it is the textbook projection, true for every
    problem of this shape, and it is what a teaching assistant would say. The answer is the
    specific numbers, and those still have to be worked out. Without this the model guesses
    coefficients and the guesses do not converge, because a direction alone ("too large")
    invites fiddling rather than deriving.
    """
    L, b = sp.sstr(p["L"]), basis_label(p)
    return (f"Each coefficient is an integral: c_n = (2/{L}) * "
            f"Integral(u(x, 0) * {b}, (x, 0, {L})). Work it out for n = 1, 2, 3, ... "
            f"instead of guessing, and simplify each one to a number.")


def _shape_hint(s):
    """Say what to DO, not just that it failed.

    Observed on the real model: it answers with `X(x)T(t)`, or with a LaTeX sum over
    undetermined coefficients A_n. Both are correct derivations and neither can be
    evaluated. A verdict ("could not read it") never tells it what to change, so name the
    single fix instead.
    """
    if "\\" in s or "sum" in s.lower() or "∑" in s or "infty" in s:
        return ("Do not write a summation or LaTeX. Expand the sum: write each term "
                "separately with its number in front, joined by + and -, for example "
                "3*exp(-2*pi**2*t)*sin(pi*x) - 0.5*exp(-8*pi**2*t)*sin(2*pi*x).")
    if re.search(r"\b[A-Za-z]_?\d*\s*\(", s) or re.search(r"\b[A-Z]_", s):
        return ("Do not leave named functions or coefficients unevaluated. Work out every "
                "coefficient as a number and write the result out in full.")
    return ("Write the answer as one arithmetic expression in x and t only, using exp, "
            "sin, cos, pi and **.")


def _coeff_report(p, ug, f0, grid, L, n_max=8, rel=0.02):
    """Compare the candidate's coefficients with the required ones, term by term.

    This exists because of what the model actually did. Told only "your starting shape is
    off by 341 percent, add more terms or correct the coefficients", Qwen3-8B sat at the same
    score for four rounds: it kept alternating signs and kept including the even terms, which
    must vanish. The number was true and useless. Projecting both shapes onto the problem's
    own basis turns it into "the coefficient of sin(pi*x) should be 0 but yours is -0.64",
    which names the change to make.
    """
    basis = p.get("basis")
    if basis is None:
        return ""
    lines = []
    for n in range(1, n_max + 1):
        b = _num(basis(n), grid, 0 * grid)
        norm = float(np.trapezoid(b * b, grid)) or 1.0
        got = float(np.trapezoid(ug * b, grid)) / norm
        want = float(np.trapezoid(f0 * b, grid)) / norm
        if abs(got - want) > rel * max(abs(want), 1e-12) + 1e-9:
            lines.append((abs(got - want), sp.sstr(basis(n)), got, want))
    if not lines:
        return ""
    lines.sort(reverse=True)
    if REVEAL_COEFFICIENTS:
        def fmt(v):
            return "0" if abs(v) < 1e-9 else f"{v:+.4g}"
        shown = [f"the coefficient of {name} should be {fmt(want)} but yours is {fmt(got)}"
                 for _, name, got, want in lines[:4]]
        return " Term by term: " + "; ".join(shown) + "."

    parts, absent = [], []
    for _, name, got, want in lines[:4]:
        if abs(want) < 1e-9:
            absent.append(name)
        elif abs(got) < 1e-9:
            parts.append(f"{name} is missing and belongs in the sum")
        elif got * want < 0:
            parts.append(f"the coefficient of {name} has the wrong sign")
        elif abs(got) > abs(want):
            parts.append(f"the coefficient of {name} is too large")
        else:
            parts.append(f"the coefficient of {name} is too small")
    if absent:
        parts.append(f"the term{'s' if len(absent) > 1 else ''} "
                     f"{', '.join(absent)} should not be there at all")
    if not parts:
        return ""
    # Deliberately qualitative. Printing the target values made the model copy them
    # straight out of the message -- it solved the problem without ever computing a
    # coefficient, which is fine for a demo and useless as a reward signal. Direction
    # only: enough to fix the mistake, not enough to skip the work. Set
    # REVEAL_COEFFICIENTS to True for a teaching walkthrough.
    return " Term by term: " + "; ".join(parts) + "."


def _decay_report(p, u, grid, L, k, n_max=8):
    """Say WHICH term decays at the wrong rate.

    The third time this lesson turned up. Given the calculator the model gets the
    coefficients exactly right -- 32/pi**3, 32/(27*pi**3), 32/(125*pi**3) on the parabola --
    and then pairs them with the wrong exponents: exp(-2*pi**2*t) against sin(pi*x/2), using
    frequency pi where the wave it multiplies uses pi/2. "u_t - 2*u_xx = -8.739 instead of 0"
    is true and says nothing about which of four terms is at fault. Each term is independent,
    so each one's rate can be measured on its own and named.
    """
    lam = p.get("lam")
    if lam is None:
        return ""
    lam1 = float(lam(1))
    t1 = 0.5 / max(float(k) * lam1 * lam1, 1e-12)
    try:
        u0 = _num(u, grid, 0 * grid)
        u1 = _num(u, grid, t1 + 0 * grid)
    except Exception:
        return ""
    if not (np.all(np.isfinite(u0)) and np.all(np.isfinite(u1))):
        return ""

    parts = []
    for i in range(1, n_max + 1):
        b = _num(p["basis"](i), grid, 0 * grid)
        norm = float(np.trapezoid(b * b, grid)) or 1.0
        c0 = float(np.trapezoid(u0 * b, grid)) / norm
        c1 = float(np.trapezoid(u1 * b, grid)) / norm
        if abs(c0) < 1e-6 or c1 / c0 <= 0:
            continue
        got = -np.log(c1 / c0) / t1
        want = float(k) * float(lam(i)) ** 2
        if abs(got - want) > 0.02 * max(abs(want), 1e-12):
            name = sp.sstr(p["basis"](i))
            parts.append(f"the {name} term decays too "
                         f"{'fast' if got > want else 'slowly'}")
    if not parts:
        return ""
    return (" Term by term: " + "; ".join(parts[:4]) + ". Each term decays at "
            "its own rate, and that rate is k times the square of the frequency in that "
            "same term's sine. Check that the frequency in each exponent matches the "
            "frequency of the wave it multiplies.")


def _num(expr, X, T):
    fn = sp.lambdify((x, t), expr, "numpy")
    out = np.asarray(fn(X, T), dtype=float)
    return np.broadcast_to(out, np.shape(X))


def prompt_of(p):
    left = "u(0, t) = 0" if p["left"] == "dirichlet" else "u_x(0, t) = 0"
    right = (f"u({sp.sstr(p['L'])}, t) = 0" if p["right"] == "dirichlet"
             else f"u_x({sp.sstr(p['L'])}, t) = 0")
    notes = []
    if "neumann" in (p["left"], p["right"]):
        notes.append("One end is insulated: its condition is on the slope, not the value.")
    if p["exact"] is None:
        notes.append("The starting shape is not a single sine wave, so the answer is a sum "
                     "of the allowed waves, each with its own coefficient and its own decay "
                     f"rate. Keep enough terms that the starting shape matches to within "
                     f"{p['tol'] * 100:g} percent.\n" + coeff_method(p))
    note = ("\n" + "\n".join(notes) + "\n") if notes else ""
    # The worked example is doing real work here. Asked without one, Qwen3-8B answered
    # level1.2 with "X(x)T(t)" and then with a LaTeX sum over undetermined coefficients
    # A_n -- correct derivations, but nothing a checker can evaluate. Showing the shape of
    # an acceptable answer is far more effective than forbidding the alternatives.
    return (f"Solve the heat equation on a rod.\n\n"
            f"  u_t = {sp.sstr(p['k'])} * u_xx,   for 0 < x < {sp.sstr(p['L'])}, t > 0\n"
            f"  {left} and {right}\n"
            f"  u(x, 0) = {sp.sstr(p['f'])}\n{note}\n"
            f"On the last line write exactly\n"
            f"  u(x, t) = <expression>\n"
            f"Write every term out with its numbers filled in, in Python syntax, like this:\n"
            f"  u(x, t) = 3*exp(-2*pi**2*t)*sin(pi*x) - 0.5*exp(-8*pi**2*t)*sin(2*pi*x)\n"
            f"Use exp, sin, cos, pi and ** for powers. No summation sign, no unknown "
            f"constants, no LaTeX.")


def check(problem, answer_text, n_points=24):
    p = problem
    s = extract(answer_text) if "=" in (answer_text or "") else (answer_text or "").strip()
    if not s:
        return dict(reward=0.0, parts={}, expr=None, start_error=None,
                    feedback="No answer found. End with a line: u(x, t) = <expression>")
    try:
        u = parse(s)
    except Exception as e:
        return dict(reward=0.0, parts={}, expr=s, start_error=None,
                    feedback=f"Could not read the expression {s[:70]!r}: "
                             f"{type(e).__name__}. {_shape_hint(s)}")
    if u.free_symbols - {x, t}:
        bad = sorted(map(str, u.free_symbols - {x, t}))
        return dict(reward=0.0, parts={}, expr=s, start_error=None,
                    feedback=f"The answer still contains the unknown {bad}. {_shape_hint(s)}")

    L, k, tol = float(p["L"]), p["k"], float(p.get("tol", 1e-6))
    rng = np.random.default_rng(p.get("seed", 0))
    xs = rng.uniform(0, L, n_points)
    ts = rng.uniform(0, 0.05 * L * L, n_points)

    grid = np.linspace(0, L, 801)
    f0 = _num(p["f"], grid, 0 * grid)
    scale = float(np.max(np.abs(f0))) or 1.0

    parts, feedback = {}, []
    try:
        resid = _num(sp.diff(u, t) - k * sp.diff(u, x, 2), xs, ts)
        ug = _num(u, grid, 0 * grid)
        ux = sp.diff(u, x)
        edge = {"dirichlet": lambda at: _num(u, at + 0 * ts, ts),
                "neumann": lambda at: _num(ux, at + 0 * ts, ts) * L}
        left = edge[p["left"]](0.0)
        right = edge[p["right"]](L)
    except Exception as e:
        return dict(reward=0.0, parts={}, expr=s, start_error=None,
                    feedback=f"The expression could not be evaluated: {type(e).__name__}")

    def ok(v, rel=1e-6):
        return bool(np.all(np.isfinite(v)) and np.max(np.abs(v)) / scale < rel)

    # Relative L2 distance between the candidate at t=0 and the required starting shape.
    denom = float(np.sqrt(np.trapezoid(f0 ** 2, grid))) or 1.0
    start_err = (float(np.sqrt(np.trapezoid((ug - f0) ** 2, grid))) / denom
                 if np.all(np.isfinite(ug)) else float("inf"))

    parts["equation"] = ok(resid)
    parts["left_bc"] = ok(left)
    parts["right_bc"] = ok(right)
    parts["start_shape"] = bool(start_err < tol)

    if not parts["equation"]:
        i = int(np.argmax(np.abs(resid)))
        feedback.append(f"The equation u_t = {sp.sstr(k)}*u_xx does not hold: at "
                        f"x={xs[i]:.3f}, t={ts[i]:.4f}, u_t - {sp.sstr(k)}*u_xx = "
                        f"{resid[i]:+.4g} instead of 0."
                        + _decay_report(p, u, grid, L, k))
    for side, name, vals, at in (("left", p["left"], left, 0.0),
                                 ("right", p["right"], right, L)):
        if not parts[f"{side}_bc"]:
            i = int(np.argmax(np.abs(vals)))
            what = ("u" if name == "dirichlet" else "the slope u_x")
            feedback.append(f"At the {side} end x={at:g}, {what} should be 0 but is "
                            f"{vals[i] / (1 if name == 'dirichlet' else L):+.4g} "
                            f"at t={ts[i]:.4f}.")
    if not parts["start_shape"]:
        feedback.append(f"At t=0 the answer differs from the required starting shape by "
                        f"{start_err * 100:.2f} percent, and it must be under "
                        f"{tol * 100:g} percent."
                        + _coeff_report(p, ug, f0, grid, L)
                        + (" " + coeff_method(p) if p["exact"] is None else ""))

    reward = sum(WEIGHTS[key] for key, good in parts.items() if good)
    return dict(reward=round(reward, 3), parts=parts, expr=s,
                start_error=None if start_err != start_err else round(start_err, 6),
                feedback=" ".join(feedback) or "Solved.")


def verify(problems, extra_cases=()):
    """Selftest helper: every known exact answer must score 1.0."""
    rc = 0
    for p in problems:
        if p["exact"] is None:
            continue
        r = check(p, sp.sstr(p["exact"]))
        if r["reward"] != 1.0:
            print(f"  FAIL: exact answer to {p['name']} scored {r['reward']}: "
                  f"{r['feedback'][:120]}")
            rc = 1
    for label, p, answer, want in extra_cases:
        r = check(p, answer)
        good = abs(r["reward"] - want) < 1e-9
        rc |= 0 if good else 1
        print(f"  {label:<34} reward {r['reward']:.1f} (want {want:.1f})  "
              f"{'ok' if good else 'FAIL'}")
        if want < 1.0 and r["feedback"]:
            print(f"      {r['feedback'][:150]}")
    return rc
