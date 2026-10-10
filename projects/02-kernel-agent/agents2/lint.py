"""What can be caught without running the kernel: names the installed nki doesn't have, and calls its
real signatures would reject.

Measured on seat-35 (168 level-1 attempts): `nisa.multiply` in 75-100% of attempts, and invented
keywords such as `nc_matmul(transpose_moving=...)`. Both fail in the simulator anyway, but there the
message comes without the real alternatives. Here each finding carries its line and what exists.

Conservative on purpose. Only names reached through an nki module are checked, and a call is checked
only when its signature is a plain one (no **kwargs, not a wrapper). Anything it cannot resolve is left
to the simulator. A lint finding must never stop a kernel that would have run.
"""

import ast
import importlib
import inspect


def _import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def lint(source, traffic_level=False):
    """Returns a list of findings, first line first. Each is dict(kind, line, name, msg).

    The messages copy Python's own wording ("module 'nki.isa' has no attribute 'multiply'"), so the
    error kinds and the checked examples in agent.FRAGMENTS match them exactly as they match the
    simulator's errors. traffic_level=True (levels 5-7) also flags importing dma_copy by name, which
    the byte counter cannot see.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    env, found, seen = {}, [], set()

    def add(kind, line, name, msg):
        if (kind, line, name) not in seen:
            seen.add((kind, line, name))
            found.append(dict(kind=kind, line=line, name=name, msg=msg))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] != "nki":
                    continue
                mod = _import(al.name)
                if mod is None:
                    add("NAME", node.lineno, al.name, f"No module named '{al.name}'")
                elif al.asname:
                    env[al.asname] = mod
                else:
                    env["nki"] = _import("nki")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "nki":
            mod = _import(node.module)
            if mod is None:
                add("NAME", node.lineno, node.module, f"No module named '{node.module}'")
                continue
            for al in node.names:
                obj = getattr(mod, al.name, None)
                if obj is None:
                    obj = _import(f"{node.module}.{al.name}")
                if obj is None:
                    add("NAME", node.lineno, f"{node.module}.{al.name}",
                        f"module '{node.module}' has no attribute '{al.name}'")
                    continue
                env[al.asname or al.name] = obj
                if traffic_level and al.name == "dma_copy":
                    add("ALIAS", node.lineno, "dma_copy",
                        "dma_copy is imported by name, so the traffic check cannot count its bytes. "
                        "Import the module (import nki.isa as nisa) and call nisa.dma_copy.")

    def resolve(expr, report):
        """The object an expression names, following modules only. None when unknown."""
        if isinstance(expr, ast.Name):
            return env.get(expr.id)
        if not isinstance(expr, ast.Attribute):
            return None
        base = resolve(expr.value, report)
        if base is None or not inspect.ismodule(base):
            return None
        if hasattr(base, expr.attr):
            return getattr(base, expr.attr)
        sub = _import(f"{base.__name__}.{expr.attr}")
        if sub is not None:
            return sub
        if report:
            add("NAME", expr.lineno, f"{base.__name__}.{expr.attr}",
                f"module '{base.__name__}' has no attribute '{expr.attr}'")
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            resolve(node, report=True)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = resolve(node.func, report=False)
        if fn is None or inspect.ismodule(fn) or not callable(fn) or inspect.isclass(fn):
            continue
        if inspect.unwrap(fn) is not fn:
            continue                       # a wrapper may accept what its target doesn't
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = sig.parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            continue
        name = _dotted(node.func) or getattr(fn, "__name__", "?")
        bad = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in params]
        if bad:
            add("KWARG", node.lineno, name,
                f"{getattr(fn, '__name__', name)}() got an unexpected keyword argument '{bad[0]}'")
            continue
        if any(isinstance(x, ast.Starred) for x in node.args) or any(kw.arg is None for kw in node.keywords):
            continue
        try:
            sig.bind(*[None] * len(node.args), **{kw.arg: None for kw in node.keywords})
        except TypeError as e:
            add("KWARG", node.lineno, name, f"{getattr(fn, '__name__', name)}(): {e}")

    found.sort(key=lambda f: (f["line"] or 0))
    return found
