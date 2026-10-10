"""AST rule check: runs before any execution, catches violations on code paths the test cases
never reach, and attributes each to a line."""
import ast

from . import rules


def check(source, fn_name="kernel"):
    """Returns a list of (line, kind, what). A syntax error is reported as a single violation."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [(e.lineno, "syntax", f"{e.msg} (line {e.lineno})")]

    out = []
    np_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in rules.ALLOWED_IMPORTS:
                    out.append((node.lineno, "import", f"import {a.name}"))
                elif a.name != root:
                    out.append((node.lineno, rules.np_kind(a.name.split(".")[1]), f"import {a.name}"))
                elif root == "numpy":
                    np_aliases.add(a.asname or "numpy")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root not in rules.ALLOWED_IMPORTS:
                out.append((node.lineno, "import", f"from {mod} import ..."))
            elif root == "numpy":
                for a in node.names:
                    if a.name in rules.BANNED_NP or mod != "numpy":
                        out.append((node.lineno, rules.np_kind(a.name), f"from {mod} import {a.name}"))

    # red team (PR #2): `xp = np` then `xp.max` slipped past a check that only knew import names.
    # Propagate plain-name aliases of numpy to a fixed point (xp = np; yp = xp; ...).
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            value = getattr(node, "value", None)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and \
                    isinstance(value, ast.Name) and value.id in np_aliases:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id not in np_aliases:
                        np_aliases.add(t.id)
                        changed = True

    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if isinstance(node, ast.Attribute):
            base_is_np = isinstance(node.value, ast.Name) and node.value.id in np_aliases
            if base_is_np and node.attr in rules.BANNED_NP:
                out.append((line, rules.np_kind(node.attr), f"np.{node.attr}"))
            elif not base_is_np and node.attr in rules.BANNED_ATTRS:
                out.append((line, "layout", f".{node.attr}"))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in rules.BANNED_METHODS:
                base_is_np = isinstance(f.value, ast.Name) and f.value.id in np_aliases
                if not base_is_np:  # np.x is covered by the Attribute branch
                    kind = rules.np_kind(f.attr) if f.attr in rules.BANNED_NP else "layout"
                    out.append((line, kind, f".{f.attr}()"))
            if isinstance(f, ast.Attribute) and f.attr in {"reduce", "accumulate", "reduceat", "outer", "at"}:
                out.append((line, "reduction", f"ufunc.{f.attr}()"))
            if isinstance(f, ast.Name):
                if f.id in rules.BANNED_BUILTINS:
                    out.append((line, "builtin", f.id))
                elif f.id in rules.REDUCING_BUILTINS and len(node.args) == 1:
                    out.append((line, "builtin-reduce", f"{f.id}(...)"))
            for kw in node.keywords:
                if kw.arg == "keepdims":
                    out.append((line, "keepdims", "keepdims="))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            out.append((line, "matmul-op", "@"))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.MatMult):
            out.append((line, "matmul-op", "@="))
        elif isinstance(node, ast.Subscript):
            parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for p in parts:
                if isinstance(p, ast.Constant) and p.value is None:
                    out.append((line, "newaxis", "indexing with None"))
                elif isinstance(p, (ast.List, ast.ListComp, ast.Set, ast.Compare, ast.BoolOp)):
                    out.append((line, "fancy-index", "index list or boolean mask"))
                elif (isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Invert)):
                    out.append((line, "fancy-index", "inverted boolean mask"))

    if not any(isinstance(n, ast.FunctionDef) and n.name == fn_name for n in tree.body):
        out.append((None, "no-kernel", f"def {fn_name}(...)"))

    seen, uniq = set(), []
    for v in out:
        if (v[0], v[1]) not in seen:
            seen.add((v[0], v[1]))
            uniq.append(v)
    return sorted(uniq, key=lambda v: (v[0] or 0, v[1]))
