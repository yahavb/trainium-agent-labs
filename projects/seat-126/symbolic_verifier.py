#!/usr/bin/env python3
"""Independent symbolic checks for a proposed heat-equation solution."""

import sympy as sp
import pdecheck

x, t = pdecheck.x, pdecheck.t


def verify_expression(expression, problem):
    try:
        u = pdecheck.parse(expression)

        if u.free_symbols - {x, t}:
            return {
                "valid": False,
                "feedback": "Expression contains unknown symbols."
            }

        L = problem["L"]
        k = problem["k"]

        residual = sp.simplify(
            sp.diff(u, t) - k * sp.diff(u, x, 2)
        )

        left_expr = (
            u if problem["left"] == "dirichlet"
            else sp.diff(u, x)
        )

        right_expr = (
            u if problem["right"] == "dirichlet"
            else sp.diff(u, x)
        )

        left = sp.simplify(left_expr.subs(x, 0))
        right = sp.simplify(right_expr.subs(x, L))

        equation_ok = residual == 0
        left_ok = left == 0
        right_ok = right == 0

        issues = []

        if not equation_ok:
            issues.append(
                "Heat-equation residual is nonzero. "
                "Check that every exponential decay rate matches "
                "the spatial frequency."
            )

        if not left_ok:
            issues.append("Left boundary condition is not satisfied.")

        if not right_ok:
            issues.append("Right boundary condition is not satisfied.")

        if equation_ok and left_ok and right_ok:
            issues.append(
                "PDE and boundary conditions pass symbolic checks. "
                "The initial-condition coefficients and truncation "
                "accuracy still need checking."
            )

        return {
            "valid": bool(equation_ok and left_ok and right_ok),
            "equation_ok": bool(equation_ok),
            "left_ok": bool(left_ok),
            "right_ok": bool(right_ok),
            "feedback": " ".join(issues)
        }

    except Exception as exc:
        return {
            "valid": False,
            "feedback": f"Verification failed: {type(exc).__name__}"
        }


def verify_fourier_coefficients(expression, problem, max_n=9):
    """Compare a proposed solution's coefficients with exact Fourier integrals."""
    if problem["left"] != "dirichlet" or problem["right"] != "dirichlet":
        return {"supported": False, "feedback": "Only Dirichlet-Dirichlet supported."}

    try:
        u = pdecheck.parse(expression)
        L = problem["L"]
        f = problem["f"]

        initial = sp.expand(u.subs(t, 0))
        issues = []
        checked = []

        for n in range(1, max_n + 1):
            basis = sp.sin(n * sp.pi * x / L)

            expected = sp.simplify(
                (2 / L) * sp.integrate(f * basis, (x, 0, L))
            )

            actual = sp.simplify(
                (2 / L) * sp.integrate(initial * basis, (x, 0, L))
            )

            difference = sp.simplify(actual - expected)
            checked.append(n)

            if difference != 0:
                if expected == 0:
                    issue = f"Mode n={n}: coefficient should be zero."
                elif actual == 0:
                    issue = f"Mode n={n}: required term is missing."
                elif sp.sign(actual) != sp.sign(expected):
                    issue = f"Mode n={n}: coefficient has the wrong sign."
                else:
                    issue = f"Mode n={n}: coefficient magnitude is incorrect."

                issues.append(issue)

        return {
            "supported": True,
            "valid": len(issues) == 0,
            "modes_checked": checked,
            "issues": issues,
            "feedback": " ".join(issues) if issues else
                        "All checked Fourier coefficients match."
        }

    except Exception as exc:
        return {
            "supported": True,
            "valid": False,
            "feedback": f"Coefficient verification failed: {type(exc).__name__}"
        }


def verify_initial_condition(expression, problem):
    """Numerically measure relative L2 error at t=0."""
    import numpy as np

    try:
        u = pdecheck.parse(expression)
        L = float(problem["L"])
        grid = np.linspace(0, L, 801)

        actual = pdecheck._num(u.subs(t, 0), grid, np.zeros_like(grid))
        expected = pdecheck._num(problem["f"], grid, np.zeros_like(grid))

        numerator = np.sqrt(
            np.trapezoid((actual - expected) ** 2, grid)
        )
        denominator = np.sqrt(
            np.trapezoid(expected ** 2, grid)
        )

        error = float(numerator / denominator) if denominator else float(numerator)
        tolerance = float(problem.get("tol", 1e-6))

        return {
            "valid": error < tolerance,
            "relative_error": error,
            "error_percent": round(error * 100, 4),
            "tolerance_percent": round(tolerance * 100, 4),
            "feedback": (
                f"Initial-condition error: {error * 100:.4f}%. "
                f"Required: below {tolerance * 100:.4f}%."
            )
        }

    except Exception as exc:
        return {
            "valid": False,
            "feedback": f"Initial-condition verification failed: {type(exc).__name__}"
        }
