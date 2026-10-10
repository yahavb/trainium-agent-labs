#!/usr/bin/env python3
"""
kernel_lint.py -- static check for HARD-CODED DIMENSIONS in an NKI kernel.

Why this exists. In the level-2 A/B, 58% of all attempts died on `tile_limits`, almost all of them on
the smallest checker shape, (32, 12):

    dma_copy requires src and dst to have the same number of elements, got src=384, dst=512
    Out-of-bound access ... index range [0, 127] exceed dimension size of 32

Both mean the kernel wrote a fixed number (a 128x512 tile, a loop to 127) where the tensor's own size
belonged. lint_api cannot see this: every call it makes has a legal signature. A literal is not a type
error, so the only way to catch it is to ask "where did this number come from?".

The check is a small taint analysis over the kernel function:

  * the function's parameters, and anything computed from a parameter or from `.shape`, are DERIVED;
  * a name bound only to constants (`TILE = 128`) is a CONSTANT;
  * an allocation `nl.ndarray(<shape>, ...)` or a slice bound that folds to a constant integer >= 2
    is HARD-CODED.

A finding is "likely" when some checker input is smaller than the literal on that axis, so the literal
cannot be right for every shape. Findings are advisory and are only ever attached to kernels that
already failed: a kernel that passes the simulator is never penalised for one (a tiled matmul
legitimately allocates a literal 128-row tile).

No Neuron SDK needed. Pure `ast`.

    from kernel_lint import lint_hardcoded_dims, format_hardcoded
    findings = lint_hardcoded_dims(code, [(32, 12), (128, 64), (64, 128), (8, 35)])
"""

import ast

MIN_LITERAL = 2          # 0 and 1 are structural (ds(j, 1), range(1)); anything bigger is a size


def _const_eval(node, consts):
    """Fold +, -, *, // over ints and constant names. Returns an int, or None if not constant."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, int) and not isinstance(node.value, bool) else None
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_eval(node.operand, consts)
        return -v if v is not None else None
    if isinstance(node, ast.BinOp):
        a, b = _const_eval(node.left, consts), _const_eval(node.right, consts)
        if a is None or b is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.FloorDiv):
                return a // b
        except ZeroDivisionError:
            return None
    return None


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _mentions_shape(node):
    return any(isinstance(n, ast.Attribute) and n.attr in ("shape", "ndim", "size")
               for n in ast.walk(node))


def _is_derived(node, derived):
    return _mentions_shape(node) or bool(_names(node) & derived)


def _target_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for t in target.elts:
            out.extend(_target_names(t))
        return out
    return []


def _find_kernel(tree):
    """The @nki.jit function, or the first function if none is decorated."""
    fns = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in fns:
        for d in fn.decorator_list:
            text = ast.unparse(d)
            if "jit" in text:
                return fn
    return fns[0] if fns else None


def _classify_names(fn, module_consts):
    """-> (derived_names, constant_names -> value), by a few passes over the body in source order."""
    derived = {a.arg for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs}
    consts = dict(module_consts)
    for _ in range(3):                                     # fixpoint for chained assignments
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                value, targets = node.value, node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                value, targets = node.value, [node.target]
            elif isinstance(node, ast.AugAssign):
                value, targets = node.value, [node.target]
            elif isinstance(node, ast.For):
                value, targets = node.iter, [node.target]
            else:
                continue
            names = [n for t in targets for n in _target_names(t)]
            if _is_derived(value, derived):
                derived.update(names)
                for n in names:
                    consts.pop(n, None)
                continue
            folded = _const_eval(value, consts)
            if folded is not None and not isinstance(node, (ast.For, ast.AugAssign)):
                for n in names:
                    if n not in derived:
                        consts[n] = folded
    return derived, consts


def _shape_arg(call):
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "shape":
            return kw.value
    return None


def _is_ndarray_call(call):
    f = call.func
    return isinstance(f, ast.Attribute) and f.attr == "ndarray"


def _buffer_of(call):
    for kw in call.keywords:
        if kw.arg == "buffer":
            return ast.unparse(kw.value)
    return ""


def lint_hardcoded_dims(code, input_shapes):
    """Return a list of {line, src, kind, axis, value, severity, msg}.

    input_shapes: the shapes of the checker's input tensors, e.g. [(32, 12), (128, 64)]. A literal is
    'likely' wrong when it exceeds the smallest input extent on its axis.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    fn = _find_kernel(tree)
    if fn is None:
        return []
    module_consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            v = _const_eval(node.value, module_consts)
            if v is not None:
                module_consts[node.targets[0].id] = v
    derived, consts = _classify_names(fn, module_consts)
    lines = code.splitlines()

    # smallest extent per axis over all checker inputs
    axes = {}
    for shp in input_shapes or []:
        for ax, ext in enumerate(shp):
            axes[ax] = min(axes.get(ax, ext), ext)
    smallest_any = min(axes.values()) if axes else None

    findings, seen = [], set()

    def add(node, kind, axis, value, where):
        lineno = getattr(node, "lineno", 0)
        key = (lineno, kind, axis, value)
        if key in seen:
            return
        seen.add(key)
        limit = axes.get(axis, smallest_any)
        likely = limit is not None and value > limit
        src = lines[lineno - 1].strip()[:110] if 0 < lineno <= len(lines) else ""
        small = ", ".join(str(tuple(s)) for s in (input_shapes or [])[:4])
        msg = (f"{where} uses the literal {value}, which is not derived from any tensor. "
               + (f"The checker's inputs have shapes {small}; the smallest extent on axis {axis} "
                  f"is {limit}, so {value} does not fit every case. " if likely and small else "")
               + "Take the size from the tensor (`P, F = x.shape`) or bound it with "
                 "min(limit, size); a tile limit is a MAXIMUM, never a size to allocate.")
        findings.append({"line": lineno, "src": src, "kind": kind, "axis": axis, "value": value,
                         "severity": "likely" if likely else "possible", "msg": msg})

    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and _is_ndarray_call(node):
            if "shared_hbm" in _buffer_of(node):
                continue                                   # output tensors take the input's shape
            shape = _shape_arg(node)
            elts = shape.elts if isinstance(shape, (ast.Tuple, ast.List)) else []
            for axis, el in enumerate(elts):
                if _is_derived(el, derived):
                    continue
                v = _const_eval(el, consts)
                if v is not None and v >= MIN_LITERAL:
                    add(node, "alloc", axis, v, f"`nl.ndarray` dimension {axis}")
        if isinstance(node, ast.Subscript):
            base = node.value
            if not (isinstance(base, ast.Name) and base.id in derived):
                continue
            idx = node.slice
            dims = idx.elts if isinstance(idx, ast.Tuple) else [idx]
            for axis, d in enumerate(dims):
                if isinstance(d, ast.Slice) and d.upper is not None and not _is_derived(d.upper,
                                                                                         derived):
                    v = _const_eval(d.upper, consts)
                    if v is not None and v >= MIN_LITERAL:
                        add(node, "slice", axis, v, f"the slice upper bound on axis {axis}")
    findings.sort(key=lambda f: (f["severity"] != "likely", f["line"]))
    return findings


def format_hardcoded(findings, limit=2, only_likely=True):
    """One compact paragraph for a prompt or a verifier message."""
    use = [f for f in findings if (f["severity"] == "likely" or not only_likely)]
    if not use:
        return ""
    return "HARD-CODED SIZE (static check): " + " ".join(
        f"[line {f['line']}: `{f['src']}`] {f['msg']}" for f in use[:limit])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        sys.exit("usage: python kernel_lint.py kernel.py  [rows,cols ...]")
    shapes = [tuple(int(x) for x in a.split(",")) for a in sys.argv[2:]] or [(32, 12), (128, 64)]
    found = lint_hardcoded_dims(open(path, encoding="utf-8").read(), shapes)
    print(format_hardcoded(found, limit=10, only_likely=False) or "no hard-coded sizes found")