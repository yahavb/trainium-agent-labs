"""Isolated starter checker with corrected hints and bounded mathematical inputs."""
import ast
import importlib.util
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import sympy as sp

STARTER = Path(__file__).resolve().parents[2] / "01-heat-rod-pde"
spec = importlib.util.spec_from_file_location("controller_starter_checker", STARTER / "pdecheck.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
x, t = checker.x, checker.t
n = sp.Symbol("n", integer=True, positive=True)
# NumPy 1.26 compatibility without changing the starter or NumPy module.
if not hasattr(np, "trapezoid"):
    class NumpyCompat:
        def __getattr__(self, key):
            return np.trapz if key == "trapezoid" else getattr(np, key)
    checker.np = NumpyCompat()


def arithmetic(source, calculator=False, profile=False):
    if not isinstance(source, str) or not source or len(source) > (400 if calculator else 12000):
        raise ValueError("expression length limit exceeded")
    tree = ast.parse(source.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 2000:
        raise ValueError("expression too complex")
    names = {"x": x, "t": t, "pi": sp.pi, "E": sp.E}
    if calculator:
        names["n"] = n
    funcs = {key: getattr(sp, key) for key in ("sin", "cos", "exp", "sqrt", "sinh", "cosh", "tan", "Abs")}
    if calculator:
        funcs.update({key: getattr(sp, key) for key in ("Integral", "integrate", "Sum", "simplify", "Rational", "log")})
    if profile:
        funcs["Piecewise"] = sp.Piecewise

    def visit(node):
        if isinstance(node, ast.Constant):
            if profile and type(node.value) is bool:
                return sp.true if node.value else sp.false
            if type(node.value) not in (int, float) or not math.isfinite(node.value) or abs(node.value) > 1e9:
                raise ValueError("invalid numeric constant")
            return sp.Integer(node.value) if type(node.value) is int else sp.Float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if not b.is_number or not b.is_real or abs(float(b)) > 64:
                    raise ValueError("power out of bounds")
                return a ** b
            for op, fn in ((ast.Add, lambda: a+b), (ast.Sub, lambda: a-b),
                           (ast.Mult, lambda: a*b), (ast.Div, lambda: a/b)):
                if isinstance(node.op, op):
                    return fn()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs and not node.keywords:
            return funcs[node.func.id](*(visit(arg) for arg in node.args))
        if (calculator or profile) and isinstance(node, ast.Tuple):
            return tuple(visit(item) for item in node.elts)
        if profile and isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.LtE):
            return sp.Le(visit(node.left), visit(node.comparators[0]))
        raise ValueError("unsupported mathematical syntax")

    value = visit(tree.body)
    if not isinstance(value, sp.Expr) or value.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError("nonfinite expression")
    return value


checker.parse = arithmetic


def basis_for(left, right, L):
    if left == right == "neumann":
        return lambda j: sp.cos((j-1)*sp.pi*x/L), lambda j: (j-1)*sp.pi/L
    if left == "neumann":
        return lambda j: sp.cos((2*j-1)*sp.pi*x/(2*L)), lambda j: (2*j-1)*sp.pi/(2*L)
    if right == "neumann":
        return lambda j: sp.sin((2*j-1)*sp.pi*x/(2*L)), lambda j: (2*j-1)*sp.pi/(2*L)
    return lambda j: sp.sin(j*sp.pi*x/L), lambda j: j*sp.pi/L


def coeff_method(p):
    return ("Compute each coefficient as Integral(f(x)*phi(x), (x,0,L)) / "
            "Integral(phi(x)**2, (x,0,L)), substituting the given initial profile and "
            "numeric L. A constant mode has norm L; other standard modes have norm L/2.")


def decay_report(p, u, grid, L, k, n_max=8):
    # The NN constant mode has zero frequency: use the first nonzero mode for time scale.
    first = next((float(p["lam"](j)) for j in range(1, n_max+1) if p["lam"](j) != 0), 1.0)
    dt = 0.5 / (float(k)*first**2)
    try:
        u0, u1 = checker._num(u, grid, 0*grid), checker._num(u, grid, dt+0*grid)
        if not np.all(np.isfinite(u0)) or not np.all(np.isfinite(u1)):
            return ""
        issues = []
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        for j in range(1, n_max+1):
            wave = checker._num(p["basis"](j), grid, 0*grid)
            norm = float(trap(wave*wave, grid))
            c0, c1 = float(trap(u0*wave, grid))/norm, float(trap(u1*wave, grid))/norm
            if abs(c0) < 1e-6:
                continue
            want_ratio = np.exp(-float(k)*float(p["lam"](j))**2*dt)
            if abs(c1/c0-want_ratio) > 0.02*max(want_ratio, 1e-6):
                issues.append(f"check the decay of {sp.sstr(p['basis'](j))}")
        return (" Term by term: " + "; ".join(issues[:4]) +
                ". Each rate is k times that mode's frequency squared; the constant mode does not decay.") if issues else ""
    except Exception:
        return ""


checker.coeff_method = coeff_method
checker._decay_report = decay_report
checker.REVEAL_COEFFICIENTS = False


def problem_of(record):
    L, k = arithmetic(record["L"]), arithmetic(record["k"])
    basis, lam = basis_for(record["left"], record["right"], L)
    return dict(name=record["id"], level=record["level"], sub=record["sub"], seed=record["seed"],
                L=L, k=k, left=record["left"], right=record["right"],
                f=arithmetic(record["f"], profile=True), tol=record["tol"], exact=None,
                basis=basis, lam=lam)


def prompt_of(record):
    p = problem_of(record)
    L, k, f = map(sp.sstr, (p["L"], p["k"], p["f"]))
    left = "u(0,t)=0" if p["left"] == "dirichlet" else "u_x(0,t)=0"
    right = f"u({L},t)=0" if p["right"] == "dirichlet" else f"u_x({L},t)=0"
    return (f"Solve u_t = {k}*u_xx on 0 < x < {L}, t > 0.\n"
            f"Boundaries: {left}; {right}. Initial profile: u(x,0) = {f}.\n"
            f"Relative L2 initial-profile error must be below {p['tol']}.\n"
            "Use finite explicit terms with numeric coefficients. End with one line:\n"
            "u(x, t) = <Python expression using x, t, exp, sin, cos, pi and **>\n"
            "Example syntax: u(x,t) = 3*exp(-2*pi**2*t)*sin(pi*x).\n"
            "No unknown coefficients, LaTeX, or unevaluated sums.")


def reference(record, terms=16):
    """Selftest/offline fixture only; production prompts never call this."""
    p = problem_of(record)
    if record.get("reference"):
        return arithmetic(record["reference"])
    out = 0
    for j in range(1, terms+1):
        phi = p["basis"](j)
        norm = sp.integrate(phi**2, (x, 0, p["L"]))
        coeff = sp.integrate(p["f"]*phi, (x, 0, p["L"]))/norm
        out += coeff*phi*sp.exp(-p["k"]*p["lam"](j)**2*t)
    return out


def _worker(conn, kind, payload):
    try:
        if kind == "check":
            record, answer = payload
            result = checker.check(problem_of(record), answer)
            error = result["start_error"]
            if error is not None and not math.isfinite(error):
                result["start_error"] = None
        elif kind == "compute":
            expr = arithmetic(payload, calculator=True)
            result = sp.sstr(sp.simplify(expr.doit()))
        elif kind == "http":
            from .experiment import http_request
            result = http_request(*payload)
        else:
            raise ValueError("unknown worker operation")
        conn.send((True, result))
    except Exception as exc:
        conn.send((False, f"{type(exc).__name__}: {str(exc)[:200]}"))
    finally:
        conn.close()


def bounded(kind, payload, timeout=20):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(child, kind, payload))
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise TimeoutError(f"{kind} exceeded {timeout:g}s")
        good, value = parent.recv()
        if not good:
            raise ValueError(value)
        return value
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def check(record, answer, timeout=20):
    try:
        return bounded("check", (record, answer), timeout)
    except (ValueError, TimeoutError, EOFError) as exc:
        return dict(reward=0.0, parts={}, expr=None, start_error=None, feedback=str(exc))
