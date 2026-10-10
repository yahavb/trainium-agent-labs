"""Numerical protocol copy for the optional safe worker, not the public Checker.

The worker supplies an AST-built expression. Public pdecheck/checker_runtime stay
identical to master. This module never imports the historical string parser.
"""
import numpy as np
import sympy as sp
from pdecheck import x, t, WEIGHTS, extract, _shape_hint, _num, _coeff_report, _decay_report, coeff_method
from checker_syntax import parse_expression as parse


def _check_original(problem, answer_text, n_points=24, _expression=None):
    """Historical numerical scoring; only called inside the bounded worker."""
    p = problem
    s = extract(answer_text) if "=" in (answer_text or "") else (answer_text or "").strip()
    if not s:
        return dict(reward=0.0, parts={}, expr=None, start_error=None,
                    feedback="No answer found. End with a line: u(x, t) = <expression>")
    try:
        u = parse(s) if _expression is None else _expression
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
