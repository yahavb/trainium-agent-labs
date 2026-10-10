"""Static check of an NKI kernel BEFORE it is simulated: finds every memory-placement mistake at once.

Why: the simulator stops at the FIRST error. On level 8 every round failed on a new placement mistake
("stationary must be in ['sbuf'], got psum", then "dst must be in ['psum'], got sbuf", ...), so each
round fixed one mistake and revealed the next -- about two minutes per mistake. This reads the kernel's
syntax tree, tracks which memory each tile lives in, and reports ALL violations, each with its line.

Only rules the chip's own tools were seen enforcing in our logs are encoded, so a correct kernel is never
rejected: a false alarm costs more than a missed mistake (the simulator still catches the rest).

    from lint import lint_kernel
    for issue in lint_kernel(source): print(issue)
"""
import ast

BUFFERS = {"sbuf": "sbuf", "psum": "psum", "shared_hbm": "hbm", "hbm": "hbm", "private_hbm": "hbm"}
# op -> {argument: allowed memories}. Argument order is the positional order of the real signature.
RULES = {
    "nc_matmul": (("dst", {"psum"}), ("stationary", {"sbuf"}), ("moving", {"sbuf"})),
    "nc_transpose": (("dst", {"psum"}), ("data", {"sbuf"})),
    "tensor_copy": (("dst", {"sbuf", "psum"}), ("src", {"sbuf", "psum"})),
    "activation": (("dst", {"sbuf", "psum"}), ("op", None), ("data", {"sbuf", "psum"})),
    "tensor_scalar": (("dst", {"sbuf", "psum"}), ("data", {"sbuf", "psum"})),
    "reciprocal": (("dst", {"sbuf", "psum"}), ("data", {"sbuf", "psum"})),
}
WHY = {
    "nc_matmul": "nc_matmul writes into psum and reads both operands from sbuf",
    "nc_transpose": "nc_transpose (the Tensor engine, up to 128x128) writes into psum and reads from sbuf; "
                    "an sbuf destination would send it to the Vector engine, limited to 32x32",
    "tensor_copy": "tensor_copy only moves data between sbuf and psum; HBM is reached with dma_copy",
    "activation": "compute instructions read and write on-chip tiles (sbuf/psum), never HBM",
    "tensor_scalar": "compute instructions read and write on-chip tiles (sbuf/psum), never HBM",
    "reciprocal": "compute instructions read and write on-chip tiles (sbuf/psum), never HBM",
}
FIX = {"psum": "nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)",
       "sbuf": "nl.ndarray(shape, dtype=..., buffer=nl.sbuf)"}
RETURNS_SBUF = {"max", "sum", "min", "mean", "exp", "log", "maximum", "minimum", "multiply", "add",
                "subtract", "divide", "zeros", "ones", "full"}


def _buffer_of_call(call):
    """nl.ndarray(shape, dtype, buffer=nl.X) / nl.zeros(..., buffer=nl.X) -> 'sbuf' | 'psum' | 'hbm'."""
    for kw in call.keywords:
        if kw.arg == "buffer" and isinstance(kw.value, ast.Attribute):
            return BUFFERS.get(kw.value.attr)
    if len(call.args) >= 3 and isinstance(call.args[2], ast.Attribute):
        return BUFFERS.get(call.args[2].attr)
    return None


def _name(node):
    """The tile a node refers to: `x` or `x[...]` -> 'x'; lists and anything else -> None."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _dims(node):
    """(seq, d) / (128, 64) -> ('seq', 'd') / ('128', '64'); anything not written plainly -> None."""
    if isinstance(node, ast.Tuple) and all(isinstance(e, (ast.Name, ast.Constant)) for e in node.elts):
        return tuple(e.id if isinstance(e, ast.Name) else str(e.value) for e in node.elts)
    return None


def lint_kernel(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []                      # the parse check reports this one
    issues, where = [], {}             # where: tile name -> memory
    shape = {}                         # tile name -> tuple of simple dimension names, when written plainly
    produced = {}                      # tile name -> line where a psum-producing instruction wrote it
    lines = source.splitlines()

    def at(node):
        text = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
        return f"line {node.lineno}: `{text[:90]}`"

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for arg in fn.args.args:       # kernel inputs arrive in HBM
            where.setdefault(arg.arg, "hbm")
    for node in ast.walk(tree):        # first pass: where every tile lives
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            mem = None
            if isinstance(f, ast.Attribute) and f.attr in ("ndarray", "zeros", "ones", "full", "empty"):
                mem = _buffer_of_call(node.value) or ("sbuf" if f.attr != "ndarray" else None)
            elif isinstance(f, ast.Attribute) and f.attr in RETURNS_SBUF and isinstance(f.value, ast.Name) \
                    and f.value.id == "nl":
                mem = "sbuf"
            if mem:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        where[t.id] = mem
                        if node.value.args and _dims(node.value.args[0]):
                            shape[t.id] = _dims(node.value.args[0])
        # `seq, d = q.shape` gives the input q the shape (seq, d)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape" \
                and isinstance(node.value.value, ast.Name) and isinstance(node.targets[0], ast.Tuple):
            dims = tuple(e.id for e in node.targets[0].elts if isinstance(e, ast.Name))
            if len(dims) == len(node.targets[0].elts):
                shape.setdefault(node.value.value.id, dims)
        # tiles that nc_matmul / nc_transpose write into must stay in psum
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("nc_matmul", "nc_transpose"):
            dst = next((kw.value for kw in node.keywords if kw.arg == "dst"), node.args[0] if node.args else None)
            if _name(dst):
                produced.setdefault(_name(dst), node.lineno)

    for node in ast.walk(tree):        # second pass: check every instruction and operator
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            op = node.func.attr
            if op in RULES and isinstance(node.func.value, ast.Name) and node.func.value.id == "nisa":
                given = {kw.arg: kw.value for kw in node.keywords}
                for (arg, _), val in zip(RULES[op], node.args):   # positional arguments, in order
                    given.setdefault(arg, val)
                for arg, allowed in RULES[op]:
                    if allowed is None or arg not in given:
                        continue
                    tile = _name(given[arg])
                    mem = where.get(tile)
                    if mem and mem not in allowed:
                        need = sorted(allowed)[0] if len(allowed) == 1 else " or ".join(sorted(allowed))
                        fix = f" Allocate it with {FIX[need]}." if need in FIX else ""
                        if arg != "dst" and mem == "psum" and need == "sbuf" and tile in produced:
                            dims = ", ".join(shape.get(tile, ("...",)))
                            fix = (f" {tile} must stay in psum (line {produced[tile]} writes it there), so do NOT "
                                   f"re-allocate it: copy it into a NEW sbuf tile first -- {tile}_sb = "
                                   f"nl.ndarray(({dims}), dtype=nl.float32, buffer=nl.sbuf); "
                                   f"nisa.tensor_copy(dst={tile}_sb, src={tile}) -- and pass {tile}_sb here.")
                        issues.append(f"{at(node)} -- `{arg}={tile}` is in {mem}, must be {need}: "
                                      f"{WHY[op]}.{fix}")
            if op in ("nc_matmul", "nc_transpose") and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "nisa":
                given = {kw.arg: kw.value for kw in node.keywords}
                for (arg, _), val in zip(RULES[op], node.args):
                    given.setdefault(arg, val)
                sh = {k: shape.get(_name(v)) for k, v in given.items() if k in ("dst", "stationary", "moving", "data")}
                nm = {k: _name(v) for k, v in given.items()}
                if op == "nc_matmul" and sh.get("stationary") and sh.get("moving") \
                        and len(sh["stationary"]) == 2 == len(sh["moving"]):
                    (k1, m), (k2, n) = sh["stationary"], sh["moving"]
                    if k1 != k2:
                        issues.append(f"{at(node)} -- nc_matmul sums over the FIRST (partition) axis of both "
                                      f"operands, so they must match: stationary={nm['stationary']} is ({k1}, {m}), "
                                      f"moving={nm['moving']} is ({k2}, {n}). To get A times B-transposed for A "
                                      f"(r, c) and B (r2, c), pass the TRANSPOSES: stationary=A^T (c, r), moving=B^T (c, r2).")
                    elif sh.get("dst") and tuple(sh["dst"]) == (m, n):
                        pass
                    elif sh.get("dst") and tuple(sh["dst"]) == (k1, k2) and m == n:
                        # dst declared (rows of stationary, rows of moving): the intent is stationary x moving^T
                        # (e.g. scores = q k^T). Name the recipe with the kernel's own tiles -- measured on levels
                        # 8 and 11, the generic "pass the transposes" was not acted on round after round.
                        A, B = nm["stationary"], nm["moving"]
                        issues.append(
                            f"{at(node)} -- you want {A} x {B}^T, shape ({k1}, {k2}), but nc_matmul computes "
                            f"stationary^T x moving and sums over the FIRST axis, so with {A} ({k1}, {m}) and {B} "
                            f"({k2}, {n}) it gives ({m}, {n}). Transpose both first so {m} is the first axis: "
                            f"{A}T_ps = nl.ndarray(({m}, {k1}), dtype=nl.float32, buffer=nl.psum); "
                            f"nisa.nc_transpose(dst={A}T_ps, data={A}); {A}T = nl.ndarray(({m}, {k1}), "
                            f"dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst={A}T, src={A}T_ps); the same "
                            f"for {B} -> {B}T ({n}, {k2}); then nisa.nc_matmul(dst={nm['dst']}, stationary={A}T, "
                            f"moving={B}T).")
                    elif sh.get("dst") and tuple(sh["dst"]) != (m, n):
                        issues.append(f"{at(node)} -- nc_matmul computes stationary^T x moving: ({k1}, {m}) and "
                                      f"({k2}, {n}) give a ({m}, {n}) result, but dst={nm['dst']} is declared "
                                      f"{tuple(sh['dst'])}. If you wanted A times B-transposed, pass the transposes "
                                      f"of A and B as stationary and moving.")
                if op == "nc_transpose" and sh.get("data") and sh.get("dst") and len(sh["data"]) == 2 \
                        and tuple(sh["dst"]) != tuple(reversed(sh["data"])):
                    a, b = sh["data"]
                    issues.append(f"{at(node)} -- transposing data={nm['data']} of shape ({a}, {b}) gives ({b}, {a}), "
                                  f"but dst={nm['dst']} is declared {tuple(sh['dst'])}.")
            if op in ("dma_copy", "tensor_copy") and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "nisa":
                # A copy moves data, it never transposes: copying a (seq, d) tensor into a (d, seq) tile has
                # the same number of elements, so it "succeeds" and silently scrambles the data (measured on
                # level 11: the model assumed k arrives transposed and allocated k_sb as (d, seq)).
                given = {kw.arg: kw.value for kw in node.keywords}
                for k_, v in zip(("dst", "src"), node.args):
                    given.setdefault(k_, v)
                dst, src = given.get("dst"), given.get("src")
                # only whole tiles (no slicing), so a slice of a bigger tile is never compared to the tile
                if isinstance(dst, ast.Name) and isinstance(src, ast.Name):
                    ds, ss = shape.get(dst.id), shape.get(src.id)
                    if ds and ss and len(ds) == len(ss) and tuple(ds) != tuple(ss):
                        hint = (" It is the same data in a different layout: a copy does not transpose. "
                                f"Allocate {dst.id} with shape {tuple(ss)}; if you need the transpose, use "
                                f"nisa.nc_transpose into a psum tile." if sorted(ds) == sorted(ss) else "")
                        issues.append(f"{at(node)} -- copying {src.id} {tuple(ss)} into {dst.id} declared "
                                      f"{tuple(ds)}: shapes must match.{hint}")
            if op == "reshape" and _name(node.func.value) in where:
                issues.append(f"{at(node)} -- tiles cannot be reshaped; allocate a tile with the shape you "
                              f"need and copy into it, or slice with ranges.")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.Div)):
            l, r = _name(node.left), _name(node.right)
            if (l in where and where[l] != "hbm") or (r in where and where[r] != "hbm"):
                issues.append(f"{at(node)} -- Python operators do not work on tiles; use "
                              f"nisa.tensor_scalar(dst=, data=, op0=nl.multiply / nl.subtract, operand0=) "
                              f"or nisa.tensor_tensor.")
    seen, out = set(), []
    for i in issues:                   # one report per line and problem
        if i not in seen:
            seen.add(i); out.append(i)
    return out
