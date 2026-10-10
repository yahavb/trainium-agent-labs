#!/usr/bin/env python3
"""
tool_calc.py — the one tool the agent gets: work out an exact integral.

Why this exists. Given the projection formula and only directional feedback, Qwen3-8B does
not compute the coefficients at all. It guesses a pattern -- measured on the parabola
problem it proposed 1.6, -0.8, 0.4, -0.2, 0.1, each half the last with alternating signs --
and then repeats itself, byte for byte, until the rounds run out. Told the target numbers
instead, it copies them out of the message and derives nothing. Neither is a project.

So the split is: the MODEL decides which integral to evaluate, and the TOOL evaluates it.
Deciding what to compute is the reasoning and stays with the model; the arithmetic it cannot
do reliably moves to sympy. Nothing is revealed -- the tool answers exactly the question it
is asked, including a wrong question, so a model that sets up the wrong integral gets the
wrong number back and has to notice.

    python tool_calc.py "Integral(x*(2-x)*sin(pi*x/2), (x, 0, 2))"
"""

import re
import sys

import sympy as sp

x, t = sp.symbols("x t", real=True)
n = sp.Symbol("n", integer=True, positive=True)

# Deliberately small. Anything not here is rejected rather than evaluated.
_ALLOWED = {
    "Integral": sp.Integral, "integrate": sp.integrate, "Sum": sp.Sum,
    "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "sqrt": sp.sqrt, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "log": sp.log, "Abs": sp.Abs,
    "pi": sp.pi, "E": sp.E, "Rational": sp.Rational, "simplify": sp.simplify,
    "x": x, "t": t, "n": n,
}

MAX_LEN = 400


def compute(expr_str):
    """Evaluate one expression exactly. Returns a human-readable string."""
    s = (expr_str or "").strip().rstrip(";")
    if not s:
        return "nothing to compute"
    if len(s) > MAX_LEN:
        return f"expression too long ({len(s)} characters, limit {MAX_LEN})"
    if re.search(r"__|import|lambda|open|eval|exec", s):
        return "rejected: only arithmetic, sin, cos, exp, sqrt, pi and Integral are allowed"

    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application)
    tr = standard_transformations + (implicit_multiplication_application,)
    glob = {"__builtins__": {}}
    glob.update({k: getattr(sp, k) for k in ("Integer", "Float", "Rational", "Symbol")})
    try:
        expr = parse_expr(s, local_dict=dict(_ALLOWED), global_dict=glob, transformations=tr)
    except Exception as e:
        return f"could not read that expression: {type(e).__name__}"

    bad = expr.free_symbols - {x, t, n}
    if bad:
        return (f"the expression contains the unknown {sorted(map(str, bad))}; substitute a "
                f"number for it before asking")
    try:
        value = sp.simplify(expr.doit())
    except Exception as e:
        return f"could not evaluate that: {type(e).__name__}"

    out = sp.sstr(value)
    if not value.free_symbols:
        try:
            out += f"  (= {float(value):.6g})"
        except (TypeError, ValueError):
            pass
    return out


ASK = re.compile(r"^[\s*_`>#-]*COMPUTE[\s*_`]*:[\s*_`]*(.*?)\s*$", re.M)

INSTRUCTIONS = (
    "You have a calculator. To get an exact value, write a line of the form\n"
    "  COMPUTE: <expression>\n"
    "for example\n"
    "  COMPUTE: Integral(x*(2 - x)*sin(pi*x/2), (x, 0, 2))\n"
    "You may write several such lines. Stop there and send them; I will reply with the "
    "values, and then you give the final answer. Do the integrals this way rather than "
    "guessing the coefficients."
)


def requests_in(text):
    """The COMPUTE: lines a reply is asking for."""
    lines, out = (text or "").splitlines(), []
    for i, line in enumerate(lines):
        m = ASK.match(line)
        if not m:
            continue
        expr, j = m.group(1).strip("`* "), i + 1
        while not expr and j < len(lines):        # **COMPUTE:** then the expression, maybe fenced
            cand = lines[j].strip()
            j += 1
            if not cand.startswith("```"):
                expr = cand.strip("`* ")
        if expr:
            out.append(expr)
    return out[:8]


def answer_block(asks):
    return "\n".join(f"COMPUTE: {a}\n  = {compute(a)}" for a in asks)


if __name__ == "__main__":
    print(compute(" ".join(sys.argv[1:])))
