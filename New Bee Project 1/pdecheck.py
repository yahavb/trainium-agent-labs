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

# CHANGE C4: Added bounded token/AST parsing; candidate text is never evaluated as Python.
import ast
import math
import re

import numpy as np
import sympy as sp

x, t = sp.symbols("x t", real=True)

WEIGHTS = dict(equation=0.4, left_bc=0.2, right_bc=0.2, start_shape=0.2)

# When True the checker prints the target coefficients. Useful for a teaching walkthrough,
# and ruinous as a reward signal: the model copied them out of the message verbatim and
# solved the parabola without computing a single coefficient. Default off; see _coeff_report.
REVEAL_COEFFICIENTS = False

_FUNCTIONS = {"exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "sqrt": sp.sqrt,
              "sinh": sp.sinh, "cosh": sp.cosh, "tan": sp.tan}
_CONSTANTS = {"pi": sp.pi, "E": sp.E, "e": sp.E, "x": x, "t": t}
_ALLOWED = {**_FUNCTIONS, **_CONSTANTS}
_MAX_EXPRESSION_CHARS = 8192
_MAX_TOKENS = 2048
_ASSIGNMENT = re.compile(r"u\s*(?:\\left\s*)?\(\s*x\s*,\s*t\s*(?:\\right\s*)?\)\s*=")


def extract(text):
    """Return the final u(x,t) assignment, including continuation lines."""
    # CHANGE C4: Replaced one-line extraction; the last assignment may be multiline.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    # Remove balanced boxed wrappers before cutting off the assignment's left-hand side.
    for count in range(129):
        box = re.search(r"\\boxed\s*\{", text)
        if box is None:
            break
        if count == 128:
            raise ValueError("Too many boxed answers")
        depth, end = 1, box.end()
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise ValueError("Unclosed boxed answer")
        text = text[:box.start()] + text[box.end():end - 1] + text[end:]
    else:
        raise ValueError("Too many boxed answers")
    hits = list(_ASSIGNMENT.finditer(text))
    return text[hits[-1].end():].strip() if hits else None


def _normalise_format(s):
    """A deliberately small finite-LaTeX subset, not a general TeX interpreter."""
    # CHANGE C4: Added bounded finite-format rewrites; reject sums and unknown commands.
    if not isinstance(s, str) or len(s) > _MAX_EXPRESSION_CHARS:
        raise ValueError(f"Expression must be text of at most {_MAX_EXPRESSION_CHARS} characters")
    if re.search(r"\\(?:sum|prod|int|infty)\b|∑|∞|\b(?:Sum|Integral|Derivative|Lambda)\s*\(", s):
        raise ValueError("Unevaluated sums/integrals are not accepted; write finitely many numeric terms")
    original = s
    s = re.sub(r"^\s*```(?:python|text|math)?\s*$", "", s, flags=re.M)
    s = re.sub(r"\\(?:begin|end)\{(?:aligned|equation\*?|displaymath)\}", "", s)
    s = s.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    s = s.replace("$", "").replace("`", "").replace("&", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\boxed", "")
    s = s.replace("\\\\", " ")

    def group(at):
        while at < len(s) and s[at].isspace():
            at += 1
        if at >= len(s) or s[at] != "{":
            raise ValueError("Finite LaTeX fractions require two braced arguments")
        start, depth = at + 1, 1
        at += 1
        while at < len(s) and depth:
            depth += (s[at] == "{") - (s[at] == "}")
            if depth > 48:
                raise ValueError("Expression nesting exceeds 48 levels")
            at += 1
        if depth:
            raise ValueError("Unclosed LaTeX brace")
        return s[start:at - 1], at

    # Rewriting innermost fractions first keeps nested \frac constructs bounded.
    for count in range(129):
        matches = list(re.finditer(r"\\frac\b", s))
        if not matches:
            break
        if count == 128:
            raise ValueError("Too many LaTeX fractions")
        m = matches[-1]
        numerator, at = group(m.end())
        denominator, end = group(at)
        s = s[:m.start()] + f"(({numerator})/({denominator}))" + s[end:]
    else:
        raise ValueError("Too many LaTeX fractions")
    commands = {"exp": "exp", "sin": "sin", "cos": "cos", "pi": "pi",
                "sqrt": "sqrt", "cdot": "*", "times": "*"}
    unknown = sorted(set(re.findall(r"\\([A-Za-z]+)", s)) - set(commands))
    if unknown:
        raise ValueError("Unsupported LaTeX commands: " + ", ".join(unknown))
    s = re.sub(r"\\([A-Za-z]+)", lambda m: commands[m.group(1)], s)
    s = re.sub(r"\\[,;! ]", " ", s)
    s = s.replace("π", "pi").replace("^", "**").replace("{", "(").replace("}", ")")
    s = s.strip().rstrip(".").strip()
    if len(s) > _MAX_EXPRESSION_CHARS:
        raise ValueError("Normalised expression exceeds the length limit")
    return s, {"rewritten": s != original.strip(), "finite_latex_subset": True,
               "max_expression_chars": _MAX_EXPRESSION_CHARS, "max_tokens": _MAX_TOKENS}


def parse(s):
    """Build SymPy objects from arithmetic/allowed calls, without eval or parse_expr."""
    # CHANGE C4: Replaced eval-based SymPy parser with tokens plus an AST whitelist.
    s, _ = _normalise_format(s)
    token = re.compile(r"\s*(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|\*\*|[()+*/-])")
    tokens, at = [], 0
    while at < len(s):
        m = token.match(s, at)
        if not m:
            raise ValueError(f"Unsupported syntax near {s[at:at + 24]!r}")
        tokens.append(m.group(1))
        at = m.end()
        if len(tokens) > _MAX_TOKENS:
            raise ValueError("Expression token limit exceeded")
    unknown = sorted({v for v in tokens if re.fullmatch(r"[A-Za-z_]\w*", v) and v not in _ALLOWED})
    if unknown:
        raise ValueError("Unknown symbols/functions: " + ", ".join(unknown))
    expanded = []
    for i, value in enumerate(tokens):
        if i:
            previous = tokens[i - 1]
            ends_atom = previous == ")" or previous in _CONSTANTS or previous[0].isdigit() or previous[0] == "."
            starts_atom = value == "(" or value in _ALLOWED or value[0].isdigit() or value[0] == "."
            if ends_atom and starts_atom:
                expanded.append("*")
        expanded.append(value)
    tree = ast.parse(" ".join(expanded), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_TOKENS:
        raise ValueError("Expression AST limit exceeded")

    def build(node, depth=0):
        if depth > 48:
            raise ValueError("Expression nesting exceeds 48 levels")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(str(node.value)) > 100 or not math.isfinite(float(node.value)):
                raise ValueError("Numeric literal is too large or non-finite")
            return sp.Integer(node.value) if type(node.value) is int else sp.Float(str(node.value))
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = build(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = build(node.left, depth + 1), build(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and right.is_number and (right.is_real is not True or abs(float(right)) > 1000):
                raise ValueError("Numeric powers must be real and have magnitude at most 1000")
            if isinstance(node.op, ast.Pow) and left.is_number and right.is_number and (not math.isfinite(float(left)) or abs(float(left)) > 1e6):
                raise ValueError("Numeric power bases must be finite and have magnitude at most 1e6")
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{node.func.id} requires exactly one expression argument")
            return _FUNCTIONS[node.func.id](build(node.args[0], depth + 1))
        raise ValueError("Only arithmetic in x,t and whitelisted single-argument functions is accepted")

    u = build(tree.body)
    if u.has(sp.zoo, sp.oo, -sp.oo, sp.nan, sp.I) or u.free_symbols - {x, t}:
        raise ValueError("Expression contains a non-finite, complex value or unknown symbol")
    return u


def frequency_of_mode(p, n):
    # CHANGE C3: Added a shared four-boundary-condition frequency helper; n starts at 1.
    pair = (p["left"], p["right"])
    mode = n if pair == ("dirichlet", "dirichlet") else n - 1 if pair == ("neumann", "neumann") else sp.sympify(n) - sp.Rational(1, 2)
    return mode * sp.pi / p["L"]


def boundary_basis(p, n):
    # CHANGE C3: Added DD/DN sine and ND/NN cosine bases, including the NN constant.
    return (sp.sin if p["left"] == "dirichlet" else sp.cos)(frequency_of_mode(p, n) * x)


def basis_label(p, n="n"):
    """How the problem's own waves are written, with n left symbolic."""
    L = sp.sstr(p["L"])
    # CHANGE C3: Replaced ambiguous branches with all four boundary pairs.
    pair = (p["left"], p["right"])
    if pair == ("dirichlet", "neumann"):
        return f"sin((2*{n} - 1)*pi*x/(2*({L})))"
    if pair == ("neumann", "dirichlet"):
        return f"cos((2*{n} - 1)*pi*x/(2*({L})))"
    if pair == ("neumann", "neumann"):
        return f"cos(({n} - 1)*pi*x/({L}))"
    return f"sin({n}*pi*x/({L}))"


def coeff_method(p):
    """The standard formula for the coefficients.

    Saying this is not giving the answer: it is the textbook projection, true for every
    problem of this shape, and it is what a teaching assistant would say. The answer is the
    specific numbers, and those still have to be worked out. Without this the model guesses
    coefficients and the guesses do not converge, because a direction alone ("too large")
    invites fiddling rather than deriving.
    """
    # CHANGE C3: Replaced NN projection normalisation: the constant mode has norm L.
    L, b = sp.sstr(p["L"]), basis_label(p)
    if p["left"] == p["right"] == "neumann":
        return (f"The constant coefficient is c_1 = (1/({L})) * Integral(u(x,0), (x,0,{L})). "
                f"For n=2,3,... use c_n = (2/({L})) * Integral(u(x,0)*{b}, (x,0,{L})). "
                "The constant mode does not decay. Compute each coefficient explicitly.")
    return (f"Each coefficient is an integral: c_n = (2/({L})) * "
            f"Integral(u(x, 0) * {b}, (x, 0, {L})). Work it out for n = 1, 2, 3, ... "
            f"instead of guessing, and simplify each one to a number.")


# CHANGE SHARED C8: parse calculator setup without evaluating model text as Python.
# The existing complete-answer parser, grading weights and tolerances are unchanged.
def _parse_calculation(source):
    if not isinstance(source, str) or not source.strip() or len(source) > 400:
        raise ValueError("Use a concrete Python calculation of at most 400 characters")
    tree = ast.parse(source.strip().rstrip(";"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("Calculation is too complex")
    functions = {**_FUNCTIONS, "log": sp.log, "Abs": sp.Abs}

    def build(node, depth=0):
        if depth > 32:
            raise ValueError("Calculation nesting exceeds 32 levels")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(float(node.value)) or abs(float(node.value)) > 1e12:
                raise ValueError("Numeric literal is too large or non-finite")
            return sp.Integer(node.value) if type(node.value) is int else sp.Float(str(node.value))
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if isinstance(node, ast.Tuple):
            return tuple(build(arg, depth + 1) for arg in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = build(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = build(node.left, depth + 1), build(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (right.is_number is not True or right.is_real is not True
                                               or abs(float(right)) > 64):
                raise ValueError("Calculation powers must be concrete and bounded")
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [build(arg, depth + 1) for arg in node.args]
            name = node.func.id
            if name in functions and len(args) == 1:
                return functions[name](args[0])
            if name == "Rational" and len(args) == 2 and all(arg.is_Integer for arg in args):
                return sp.Rational(*args)
            if name == "simplify" and len(args) == 1:
                return args[0]  # Defer all expensive evaluation until the calculator worker.
            if name in ("Integral", "integrate") and len(args) == 2:
                limit = args[1]
                if not isinstance(limit, tuple) or len(limit) != 3 or limit[0] != x:
                    raise ValueError("Use a definite integral with limit (x, 0, L)")
                if any(bound.free_symbols for bound in limit[1:]):
                    raise ValueError("Integral limits must be concrete")
                return sp.Integral(args[0], limit)
        raise ValueError("Unsupported calculation syntax")

    result = build(tree.body)
    if not isinstance(result, sp.Expr) or result.has(sp.zoo, sp.oo, -sp.oo, sp.nan, sp.I):
        raise ValueError("Calculation must be a finite real expression")
    return result


# CHANGE SHARED C9: certify the actual basis and its boundary conditions before sharing.
def _verified_wave(p, wave):
    if wave == 1:
        if p["left"] != "neumann" or p["right"] != "neumann":
            raise ValueError("The constant mode requires two insulated ends")
        return sp.Integer(0), sp.sympify(p["L"])
    if wave.func not in (sp.sin, sp.cos):
        raise ValueError("Use one unscaled sine/cosine mode, or the NN constant mode 1")
    omega = _wave_frequency(wave)
    if omega is None or omega.is_positive is not True or sp.simplify(wave.args[0] - omega*x) != 0:
        raise ValueError("The mode needs a concrete positive linear frequency")
    ratio = sp.simplify(omega * p["L"] / sp.pi)
    pair = (p["left"], p["right"])
    index = ratio if pair == ("dirichlet", "dirichlet") else ratio + 1 if pair == ("neumann", "neumann") else ratio + sp.Rational(1, 2)
    if index.is_integer is not True or index.is_positive is not True or float(index) > 128:
        raise ValueError("Frequency is outside the allowed boundary-compatible modes")
    if sp.simplify(wave - boundary_basis(p, index)) != 0:
        raise ValueError("This sine/cosine does not use the problem's allowed basis")
    for side, location in (("left", 0), ("right", p["L"])):
        boundary = wave if p[side] == "dirichlet" else sp.diff(wave, x)
        if sp.simplify(boundary.subs(x, location)) != 0:
            raise ValueError(f"The proposed mode fails the {side} boundary")
    return omega, sp.sympify(p["L"]) / 2


def _same_integral_setup(left, right):
    """Compare integral setup algebraically, without performing the integration."""
    integrals = sorted(left.atoms(sp.Integral) | right.atoms(sp.Integral), key=sp.srepr)
    representatives, replacements = [], {}
    for integral in integrals:
        for known, symbol in representatives:
            if integral.limits == known.limits and sp.simplify(integral.function - known.function) == 0:
                replacements[integral] = symbol
                break
        else:
            symbol = sp.Dummy(f"projection_{len(representatives)}")
            representatives.append((integral, symbol))
            replacements[integral] = symbol
    return sp.cancel(left.xreplace(replacements) - right.xreplace(replacements)) == 0


def _projection_metadata(p, expression):
    """Recognise verified projections/norms; never infer correctness from the result alone."""
    integrals = list(expression.atoms(sp.Integral))
    if not 1 <= len(integrals) <= 2:
        return None
    if any(i.limits != ((x, sp.Integer(0), sp.sympify(p["L"])),) for i in integrals):
        return None
    waves = set().union(*(i.function.atoms(sp.sin, sp.cos) for i in integrals))
    if p["left"] == p["right"] == "neumann":
        waves.add(sp.Integer(1))
    for wave in sorted(waves, key=sp.srepr):
        try:
            omega, norm = _verified_wave(p, wave)
        except (ValueError, TypeError):
            continue
        projection = sp.Integral(p["f"] * wave, (x, 0, p["L"]))
        norm_integral = sp.Integral(wave**2, (x, 0, p["L"]))
        if (_same_integral_setup(expression, projection/norm)
                or _same_integral_setup(expression, projection/norm_integral)):
            kind = "coefficient"
        elif _same_integral_setup(expression, projection):
            kind = "projection_integral"
        elif _same_integral_setup(expression, norm_integral):
            kind = "mode_norm"
        else:
            continue
        return dict(kind=kind, wave=sp.sstr(wave), omega=sp.sstr(omega),
                    norm=sp.sstr(norm), rate=sp.sstr(sp.simplify(p["k"]*omega**2)))
    return None


# CHANGE SHARED C10: validate setup first; only verified finite results can enter a cache.
def prepare_intermediate(p, request):
    """Prepare a requested calculation, not an automatic PDE solution.

    MODE requests must supply their own wave, projection setup and rate.
    Legacy COMPUTE requests remain usable; unrecognised setup is never certified/shared.
    No known answer, canned coefficient formula or complete-answer tolerance is used here.
    """
    kind = request.get("kind", "compute")
    if kind not in ("compute", "mode"):
        raise ValueError("Unknown intermediate request type")
    expression = _parse_calculation(request["expression"])
    if kind == "mode":
        wave = parse(request["wave"])
        omega, norm = _verified_wave(p, wave)
        rate = parse(request["rate"])
        if rate.free_symbols or sp.simplify(rate - p["k"]*omega**2) != 0:
            raise ValueError("Decay rate fails beta=k*omega**2 for this same wave")
        projection = sp.Integral(p["f"]*wave, (x, 0, p["L"]))
        norm_integral = sp.Integral(wave**2, (x, 0, p["L"]))
        if not (_same_integral_setup(expression, projection/norm)
                or _same_integral_setup(expression, projection/norm_integral)):
            raise ValueError("Coefficient setup must be the normalized projection of this problem's f onto this same wave")
        metadata = dict(kind="coefficient", wave=sp.sstr(wave), omega=sp.sstr(omega),
                        norm=sp.sstr(norm), rate=sp.sstr(sp.simplify(p["k"]*omega**2)))
    else:
        metadata = _projection_metadata(p, expression)
    shareable = metadata is not None
    # Arithmetic constants are safe only as answers to the identical explicit calculation.
    # They are not broadcast as coefficients or accepted mathematical construction steps.
    if metadata is None and not expression.free_symbols and not expression.has(sp.Integral):
        shareable = True
    # Equivalent, certified projection setups share one entry, including MODE/COMPUTE.
    # The agent scopes these keys to L,k,f and both boundaries, never to a seat-wide cache.
    canonical = "verified:" + metadata["kind"] + ":" + metadata["wave"] if metadata else sp.srepr(expression)
    return dict(kind=kind, expression=expression, metadata=metadata, shareable=shareable,
                canonical=canonical, request=request)


def evaluate_intermediate(p, prepared):
    """Calculate in an isolated tool worker after lightweight setup validation."""
    value = sp.simplify(prepared["expression"].doit())
    if (value.free_symbols or value.has(sp.Integral, sp.Sum, sp.Derivative)
            or value.is_real is not True or not math.isfinite(float(value))):
        raise ValueError("Calculation did not produce a finite real numeric value")
    exact = sp.sstr(value)
    metadata = prepared["metadata"]
    mode = None
    if metadata and metadata["kind"] == "coefficient":
        mode = dict(wave=metadata["wave"], coefficient=exact, rate=metadata["rate"])
    scope = metadata["kind"] if metadata else "arithmetic_only"
    return dict(text=f"{exact}  (= {float(value):.6g})", verified=bool(metadata),
                shareable=prepared["shareable"], scope=scope, mode=mode)


def check_intermediate(p, request):
    """Public local-check entry point. Rejection never receives a reward or a verified mode."""
    try:
        return evaluate_intermediate(p, prepare_intermediate(p, request))
    except Exception as error:
        return dict(text=f"Intermediate rejected: {error}", verified=False,
                    shareable=False, scope="rejected", mode=None)


def _boundary_hint(p):
    # CHANGE C3: Added why each boundary pair selects its sine/cosine frequencies.
    explanations = {
        ("dirichlet", "dirichlet"): "A sine has zero value at x=0; zero value at x=L requires sin(omega*L)=0, hence omega=n*pi/L.",
        ("dirichlet", "neumann"): "A sine has zero value at x=0. Its slope is omega*cos(omega*x), so insulation at L requires cos(omega*L)=0: half-integer sine frequencies. Integer sine frequencies have nonzero slope at L.",
        ("neumann", "dirichlet"): "A cosine has zero slope at x=0; zero value at L requires cos(omega*L)=0: half-integer cosine frequencies. Integer cosine frequencies have nonzero value at L.",
        ("neumann", "neumann"): "A cosine has zero slope at x=0. Its slope at L is proportional to sin(omega*L), so both insulated ends allow integer cosine frequencies and a constant mode.",
    }
    return explanations[(p["left"], p["right"])]


def _shape_hint(s):
    """Say what to DO, not just that it failed.

    Observed on the real model: it answers with `X(x)T(t)`, or with a LaTeX sum over
    undetermined coefficients A_n. Both are correct derivations and neither can be
    evaluated. A verdict ("could not read it") never tells it what to change, so name the
    single fix instead.
    """
    # CHANGE C4: Replaced blanket LaTeX rejection with finite-term guidance.
    if "sum" in s.lower() or "∑" in s or "infty" in s or "Integral" in s:
        return ("Do not leave a sum or integral unevaluated. Expand the sum: write each term "
                "separately with its number in front, joined by + and -, for example "
                "3*exp(-2*pi**2*t)*sin(pi*x) - 0.5*exp(-8*pi**2*t)*sin(2*pi*x).")
    if re.search(r"\b[A-Za-z]_?\d*\s*\(", s) or re.search(r"\b[A-Z]_", s):
        return ("Do not leave named functions or coefficients unevaluated. Work out every "
                "coefficient as a number and write the result out in full.")
    return ("Write the answer as one arithmetic expression in x and t only, using exp, "
            "sin, cos, pi and **.")


def _coeff_report(p, ug, f0, grid, L, n_max=8, rel=None):
    """Compare the candidate's coefficients with the required ones, term by term.

    This exists because of what the model actually did. Told only "your starting shape is
    off by 341 percent, add more terms or correct the coefficients", Qwen3-8B sat at the same
    score for four rounds: it kept alternating signs and kept including the even terms, which
    must vanish. The number was true and useless. Projecting both shapes onto the problem's
    own basis turns it into "the coefficient of sin(pi*x) should be 0 but yours is -0.64",
    which names the change to make.
    """
    # CHANGE C6: Retained tol/2 coefficient diagnostics from the earlier improvement.
    # A fixed 2% diagnostic threshold can hide coefficient errors that fail the
    # starting-shape tolerance (0.5% on level 1.3). Only refine the feedback;
    # leave grading unchanged and keep the target coefficients undisclosed.
    if rel is None:
        rel = min(0.02, float(p.get("tol", 1e-6)) / 2)
    # CHANGE C3: Added the four-BC basis fallback for caller-supplied problem dictionaries.
    basis = p.get("basis") or (lambda n: boundary_basis(p, n))
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
            # CHANGE C6: Replaced ambiguous signed comparisons with coefficient magnitude.
            parts.append(f"the magnitude of the coefficient of {name} is too large")
        else:
            parts.append(f"the magnitude of the coefficient of {name} is too small")
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


def _wave_frequency(wave):
    # CHANGE C1: Added direct frequency extraction from the actual sine/cosine argument.
    argument = wave.args[0]
    omega = sp.diff(argument, x)
    if argument.has(t) or omega.free_symbols or omega.is_real is not True:
        return None
    return omega


def _candidate_frequencies(u):
    # CHANGE C5: Added mode-aware scales; nonlinear wave arguments remain unsupported here.
    frequencies = []
    for term in sp.Add.make_args(u):
        total = 0.0
        for wave in term.atoms(sp.sin, sp.cos):
            omega = _wave_frequency(wave)
            power = sp.sympify(term.as_powers_dict().get(wave, 1))
            if omega is not None and power.is_number:
                total += abs(float(omega)) * max(1.0, abs(float(power)))
        if total and math.isfinite(total):
            frequencies.append(total)
    return frequencies


def _decay_details(p, u):
    # CHANGE C1: Replaced late-time log-ratios with symbolic amplitude derivatives.
    # This avoids underflow and tests sin, cos and the NN constant at their own rates.
    details = []
    for term in sp.Add.make_args(u)[:64]:
        waves = list(term.atoms(sp.sin, sp.cos))
        if len(waves) == 1:
            wave = waves[0]
            omega = _wave_frequency(wave)
            amplitude = term / wave
            name = sp.sstr(wave)
            if omega is None or amplitude.has(x):
                details.append(dict(wave=name, status="unsupported_separation"))
                continue
        elif not waves and not term.has(x):
            amplitude, omega, name = term, sp.Integer(0), "constant mode"
        else:
            details.append(dict(wave=sp.sstr(term), status="unsupported_separation"))
            continue
        expected = sp.simplify(p["k"] * omega ** 2)
        if amplitude == 0:
            continue
        try:
            rate = sp.simplify(-sp.diff(amplitude, t) / amplitude)
            residual = sp.simplify(sp.diff(amplitude, t) + expected * amplitude)
            if residual == 0:
                status = "correct"
            elif rate.free_symbols or rate.is_real is not True or not math.isfinite(float(rate)):
                status = "not_single_exponential"
            elif abs(float(rate - expected)) <= 1e-8 * max(1.0, abs(float(expected))):
                status = "correct_within_diagnostic_tolerance"
            else:
                status = "too_fast" if float(rate) > float(expected) else "too_slow"
            details.append(dict(wave=name, omega=sp.sstr(omega), expected_rate=sp.sstr(expected),
                                observed_rate=sp.sstr(rate), status=status))
        except Exception:
            details.append(dict(wave=name, expected_rate=sp.sstr(expected), status="diagnostic_unavailable"))
    return details


def _decay_report(p, u, grid, L, k, n_max=8):
    """Explain separated term rates without sampling a vanished high-frequency mode."""
    # CHANGE C1: Kept the public helper signature; grid/L/k no longer drive log-ratios.
    parts = []
    for item in _decay_details(p, u):
        if item["status"] in ("too_fast", "too_slow"):
            direction = "fast" if item["status"] == "too_fast" else "slowly"
            parts.append(f"{item['wave']} decays too {direction}; its required rate is k*omega**2 = {item['expected_rate']}")
        elif item["status"] == "not_single_exponential":
            parts.append(f"{item['wave']} needs a time factor exp(-({item['expected_rate']})*t)")
    if not parts:
        return ""
    return (" Term by term: " + "; ".join(parts[:max(1, min(n_max, 8))]) +
            ". Use the frequency omega in each sine or cosine, including its length divisor; "
            "a spatial constant has omega=0 and must remain constant in time.")


def _num(expr, X, T):
    # CHANGE C5: Added complex-value rejection before float conversion.
    fn = sp.lambdify((x, t), expr, "numpy")
    out = np.asarray(fn(X, T))
    if np.iscomplexobj(out):
        if np.any(np.imag(out) != 0):
            raise ValueError("The candidate evaluates to a complex value")
        out = np.real(out)
    out = np.asarray(out, dtype=float)
    return np.broadcast_to(out, np.shape(X))


def _sampling(p, u, n_points):
    # CHANGE C5: Added independent random holdouts and length/conductivity/mode-aware times.
    count = int(n_points)
    if count < 1 or count > 4096:
        raise ValueError("n_points must be between 1 and 4096")
    L, k = float(p["L"]), float(p["k"])
    if not (math.isfinite(L) and math.isfinite(k) and L > 0 and k > 0):
        raise ValueError("The problem requires finite L>0 and k>0")
    seed = int(p.get("seed", 0))
    if seed < 0:
        raise ValueError("The problem seed must be nonnegative")
    streams = np.random.SeedSequence(seed).spawn(2)
    fixed, holdout = (np.random.default_rng(stream) for stream in streams)
    frequencies = _candidate_frequencies(u)
    target_frequencies = _candidate_frequencies(p["f"])
    candidate_highest = max(frequencies, default=0.0)
    target_highest = max(target_frequencies, default=0.0)
    highest = max(candidate_highest, target_highest)
    tau = L * L / k
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError("L squared over k must have a finite positive time scale")
    fast_tau = min(tau, (1 / highest) / highest / k) if highest else tau
    fast_tau = max(1e-300, fast_tau)
    earliest = max(1e-300, fast_tau * 1e-6)
    times = np.unique(np.array([tau * v for v in (1e-4, .01, .1, .5, 2)] +
                               [fast_tau * v for v in (1e-6, .01, .1, 1)]))
    if not (np.all(np.isfinite(times)) and np.all(times > 0) and earliest <= 2 * tau):
        raise ValueError("The requested time scales are outside finite numerical resolution")
    spatial = np.concatenate((fixed.uniform(0, L, count), holdout.uniform(0, L, count),
                              L * np.array([1e-6, 1e-4, .01, .25, .5, .75, .99, 1-1e-4, 1-1e-6])))
    X, T = np.meshgrid(spatial, times, indexing="ij")
    random_times = np.exp(holdout.uniform(np.log(earliest), np.log(2 * tau), count))
    xs = np.concatenate((X.ravel(), holdout.uniform(0, L, count)))
    ts = np.concatenate((T.ravel(), random_times))
    boundary_times = np.concatenate(([0.0], times, random_times))
    estimated_samples = 32 * highest / math.pi * L
    # Record a bounded lower bound instead of constructing an enormous grid-size integer.
    requested_grid = max(801, int(math.ceil(min(estimated_samples, 16385))) + 1)
    grid_count = min(requested_grid, 16385)
    grid = np.linspace(0, L, grid_count)
    # A jittered grid uses independent locations while retaining bounded integration gaps.
    jittered = grid.copy()
    jittered[1:-1] += holdout.uniform(-.4, .4, grid_count - 2) * (L / (grid_count - 1))
    diagnostic = dict(seed=seed, fixed_spawn_key=list(streams[0].spawn_key),
                      holdout_spawn_key=list(streams[1].spawn_key), replayable=True,
                      independent_holdout=True, n_points=count, pde_points=len(xs),
                      boundary_time_points=len(boundary_times), time_scale_L2_over_k=tau,
                      fastest_mode_time_scale=fast_tau, candidate_max_frequency=candidate_highest,
                      initial_target_max_frequency=target_highest, sampling_max_frequency=highest,
                      time_anchors=times.tolist(), initial_grid_points=grid_count,
                      initial_grid_requested=requested_grid,
                      initial_grid_requested_is_lower_bound=estimated_samples > 16385,
                      initial_resolution_capped=requested_grid > grid_count,
                      initial_holdout="independent jittered nonuniform quadrature",
                      guarantee="finite numerical coverage, not a proof over the whole domain")
    return xs, ts, boundary_times, grid, jittered, diagnostic


def _result(parts, expr, start_error, feedback, repairs=(), diagnostics=None, failure=None):
    # CHANGE C2: Added structured keys while preserving every original result key.
    passed = [name for name, good in parts.items() if good]
    failed = [name for name, good in parts.items() if not good]
    if failure:
        failed.append(failure)
    labels = {"equation": "PDE", "left_bc": "left boundary", "right_bc": "right boundary",
              "start_shape": "initial shape"}
    messages = list(feedback)
    if passed:
        messages.insert(0, "Passed: " + ", ".join(labels.get(name, name) for name in passed) + ".")
    if parts and not failed:
        messages.append("Solved.")
    reward = sum(WEIGHTS[name] for name in passed if name in WEIGHTS)
    return dict(reward=round(reward, 3), parts=parts, expr=expr,
                start_error=start_error, feedback=" ".join(messages),
                passed_conditions=passed, failed_conditions=failed,
                repair_actions=list(repairs), diagnostics=diagnostics or {})


def prompt_of(p):
    left = "u(0, t) = 0" if p["left"] == "dirichlet" else "u_x(0, t) = 0"
    right = (f"u({sp.sstr(p['L'])}, t) = 0" if p["right"] == "dirichlet"
             else f"u_x({sp.sstr(p['L'])}, t) = 0")
    notes = []
    if "neumann" in (p["left"], p["right"]):
        # CHANGE C3: Replaced the one-insulated-end wording, including NN problems.
        notes.append("Each insulated end imposes u_x=0, a slope condition.")
    # CHANGE C3/C7: Added the correct basis and a coefficient-before-decay work order.
    notes.append(f"Allowed modes: {basis_label(p)} for n=1,2,3,... . "
                 + ("For NN, n=1 is the spatial constant; n>=2 uses cos((n-1)*pi*x/L)."
                    if p["left"] == p["right"] == "neumann" else ""))
    if p["exact"] is None:
        notes.append("Represent the starting shape with one or more allowed modes, "
                     "each with its own coefficient and its own decay "
                     f"rate. Keep enough terms that the starting shape matches to within "
                     f"{p['tol'] * 100:g} percent.\n" + coeff_method(p))
    notes.append("Work in this order: choose the boundary-compatible basis; compute each "
                 "coefficient; record that term's actual frequency omega, including /L; "
                 "compute its rate k*omega**2; pair that rate with the same wave. "
                 "Check u(x,0) first, then the PDE and both boundaries. If u(x,0) already "
                 "passes together with both boundaries, preserve its coefficients and "
                 "basis while correcting decay rates. If a boundary fails, preserve "
                 "the initial shape but reproject coefficients when changing basis. "
                 "Recheck all conditions. Give a finite answer, not the derivation.")
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
            # CHANGE C4: Replaced the blanket LaTeX prohibition with the bounded subset.
            f"Use exp, sin, cos, pi and ** for powers. Finite LaTeX with explicit "
            f"exp/sin/cos arguments, pi, braced fractions and powers is also accepted. "
            f"No unevaluated sums, integrals or unknown coefficients.")


def check(problem, answer_text, n_points=24):
    # CHANGE C2/C4/C5: Replaced check internals, preserving the signature and old result keys.
    # Weights/tolerances are retained; broader finite sampling intentionally can change scores.
    p = problem
    diagnostics = {"format": {}, "sampling": {}}
    try:
        text = answer_text or ""
        s = extract(text) if _ASSIGNMENT.search(text) else text.strip()
    except Exception as e:
        return _result({}, None, None, [f"Could not extract the final answer: {e}"],
                       ["Write one final u(x,t) assignment with finitely many numeric terms."],
                       diagnostics, failure="format")
    if not s:
        return _result({}, None, None, ["No answer found. End with u(x,t) = <expression>."],
                       ["Supply the final explicit expression."], diagnostics, failure="format")
    try:
        normalised, format_diagnostic = _normalise_format(s)
        diagnostics["format"] = {**format_diagnostic, "normalised_expression": normalised}
        u = parse(s)
    except Exception as e:
        diagnostics["format"].update(accepted=False, error=str(e))
        return _result({}, s, None,
                       [f"Could not read the expression: {e}. {_shape_hint(s)}"],
                       ["Replace unknown symbols/coefficients with computed numbers; use the supported finite syntax."],
                       diagnostics, failure="format")
    diagnostics["format"]["accepted"] = True
    try:
        L, k, tol = float(p["L"]), p["k"], float(p.get("tol", 1e-6))
        if not (math.isfinite(tol) and tol > 0):
            raise ValueError("The initial-shape tolerance must be finite and positive")
        xs, ts, boundary_ts, grid, holdout_grid, sampling = _sampling(p, u, n_points)
        diagnostics["sampling"] = sampling
        f0 = _num(p["f"], grid, 0 * grid)
        holdout_f0 = _num(p["f"], holdout_grid, 0 * holdout_grid)
        if not (np.all(np.isfinite(f0)) and np.all(np.isfinite(holdout_f0))):
            raise ValueError("The problem's initial shape is non-finite")
        scale = max(float(np.max(np.abs(f0))), float(np.max(np.abs(holdout_f0)))) or 1.0
        resid = _num(sp.diff(u, t) - k * sp.diff(u, x, 2), xs, ts)
        ug = _num(u, grid, 0 * grid)
        holdout_ug = _num(u, holdout_grid, 0 * holdout_grid)
        ux = sp.diff(u, x)
        edge = {"dirichlet": lambda at: _num(u, at + 0 * boundary_ts, boundary_ts),
                "neumann": lambda at: _num(ux, at + 0 * boundary_ts, boundary_ts) * L}
        left = edge[p["left"]](0.0)
        right = edge[p["right"]](L)
    except Exception as e:
        diagnostics["evaluation_error"] = str(e)
        return _result({}, s, None, [f"The expression could not be evaluated: {e}."],
                       ["Use a finite real expression throughout the rod and sampled times."],
                       diagnostics, failure="evaluation")

    def ok(v, rel=1e-6):
        return bool(np.all(np.isfinite(v)) and np.max(np.abs(v)) / scale < rel)

    # CHANGE C5: Replaced one-grid L2 grading with a uniform/nonuniform integration cross-check.
    def l2_error(candidate, target, points):
        if not np.all(np.isfinite(candidate)):
            return float("inf")
        denom = float(np.sqrt(np.trapezoid(target ** 2, points))) or 1.0
        return float(np.sqrt(np.trapezoid((candidate - target) ** 2, points))) / denom
    grid_error = l2_error(ug, f0, grid)
    holdout_error = l2_error(holdout_ug, holdout_f0, holdout_grid)
    start_err = max(grid_error, holdout_error)
    diagnostics["sampling"].update(initial_uniform_l2=grid_error if math.isfinite(grid_error) else None,
                                     initial_holdout_l2=holdout_error if math.isfinite(holdout_error) else None,
                                     initial_grading_rule="max of both relative L2 estimates")
    parts, feedback, repairs = {}, [], []
    parts["equation"] = ok(resid)
    parts["left_bc"] = ok(left)
    parts["right_bc"] = ok(right)
    parts["start_shape"] = bool(start_err < tol)

    def worst(values):
        return int(np.argmax(np.nan_to_num(np.abs(values), nan=np.inf, posinf=np.inf)))

    if not parts["equation"]:
        i = worst(resid)
        try:
            diagnostics["decay_terms"] = _decay_details(p, u)
            decay = _decay_report(p, u, grid, L, k)
        except Exception as e:
            diagnostics["decay_diagnostic_error"] = str(e)
            decay = " Match each term's decay rate to k times its actual frequency squared."
        feedback.append(f"The equation u_t = {sp.sstr(k)}*u_xx does not hold: at "
                        f"x={xs[i]:.3f}, t={ts[i]:.4f}, u_t - {sp.sstr(k)}*u_xx = "
                        f"{resid[i]:+.4g} instead of 0."
                        + decay)
        if parts["start_shape"] and parts["left_bc"] and parts["right_bc"]:
            # CHANGE C2: Added a precise preservation action after a passed initial condition.
            repairs.append("Preserve u(x,0), its coefficients and the spatial basis; change only each decay rate to k*omega**2, then recheck all four conditions.")
        else:
            repairs.append("For every wave use its own frequency omega and rate k*omega**2; a spatial constant must not decay.")
    for side, name, vals, at in (("left", p["left"], left, 0.0),
                                 ("right", p["right"], right, L)):
        if not parts[f"{side}_bc"]:
            i = worst(vals)
            what = ("u" if name == "dirichlet" else "the slope u_x")
            feedback.append(f"At the {side} end x={at:g}, {what} should be 0 but is "
                            f"{vals[i] / (1 if name == 'dirichlet' else L):+.4g} "
                            f"at t={boundary_ts[i]:.4f}.")
            repairs.append(f"Repair the {side} {name} boundary using the allowed basis {basis_label(p)}; recheck the initial shape after a basis change.")
    if not (parts["left_bc"] and parts["right_bc"]):
        feedback.append(f"The allowed basis is {basis_label(p)}. " + _boundary_hint(p))
    if not parts["start_shape"]:
        coeff = ""
        if np.all(np.isfinite(ug)):
            try:
                coeff = _coeff_report(p, ug, f0, grid, L)
            except Exception as e:
                diagnostics["coefficient_diagnostic_error"] = str(e)
        feedback.append(f"At t=0 the answer differs from the required starting shape by "
                        f"{start_err * 100:.2f} percent, and it must be under "
                        f"{tol * 100:g} percent."
                        + coeff
                        + (" " + coeff_method(p) if p["exact"] is None else ""))
        repairs.append("Compute the coefficients by projection in the boundary-compatible basis; keep enough terms to pass both initial-shape L2 estimates.")
    elif not all(parts.values()):
        if parts["left_bc"] and parts["right_bc"]:
            feedback.append("The initial shape and both boundaries already pass. Preserve u(x,0), its coefficients and basis; repair decay rates, then recheck every condition.")
        else:
            feedback.append("The initial shape already passes. Preserve u(x,0); a boundary repair may require a different basis and recomputed coefficients. Recheck every condition.")
    return _result(parts, s, round(start_err, 6) if math.isfinite(start_err) else None,
                   feedback, repairs, diagnostics)


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
