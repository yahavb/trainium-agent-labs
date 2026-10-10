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
import difflib
import inspect

try:                                   # the installed NKI is the authority on what exists
    import nki.isa as _nisa
except Exception:                      # (not installed: this check is skipped)
    _nisa = None

BUFFERS = {"sbuf": "sbuf", "psum": "psum", "shared_hbm": "hbm", "hbm": "hbm", "private_hbm": "hbm"}
# op -> {argument: allowed memories}. Argument order is the positional order of the real signature.
RULES = {
    "nc_matmul": (("dst", {"psum"}), ("stationary", {"sbuf"}), ("moving", {"sbuf"})),
    "nc_transpose": (("dst", {"psum"}), ("data", {"sbuf"})),
    "tensor_copy": (("dst", {"sbuf", "psum"}), ("src", {"sbuf", "psum"})),
    "dma_copy": (("dst", {"hbm", "sbuf"}), ("src", {"hbm", "sbuf"})),
    "activation": (("dst", {"sbuf", "psum"}), ("op", None), ("data", {"sbuf", "psum"})),
    "tensor_scalar": (("dst", {"sbuf", "psum"}), ("data", {"sbuf", "psum"})),
    "reciprocal": (("dst", {"sbuf", "psum"}), ("data", {"sbuf", "psum"})),
}
WHY = {
    "nc_matmul": "nc_matmul writes into psum and reads both operands from sbuf",
    "nc_transpose": "nc_transpose (the Tensor engine, up to 128x128) writes into psum and reads from sbuf; "
                    "an sbuf destination would send it to the Vector engine, limited to 32x32",
    "tensor_copy": "tensor_copy only moves data between sbuf and psum; HBM is reached with dma_copy",
    "dma_copy": "dma_copy moves data between HBM and sbuf only and cannot touch psum",
    "activation": "compute instructions read and write on-chip tiles (sbuf/psum), never HBM",
    "tensor_scalar": "compute instructions read and write on-chip tiles (sbuf/psum), never HBM",
    "reciprocal": "compute instructions read and write on-chip tiles (sbuf/psum), never HBM",
}
FIX = {"psum": "nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)",
       "sbuf": "nl.ndarray(shape, dtype=..., buffer=nl.sbuf)"}
RETURNS_SBUF = {"max", "sum", "min", "mean", "exp", "log", "maximum", "minimum", "multiply", "add",
                "subtract", "divide", "zeros", "ones", "full"}


# Short excerpts from the AWS NKI docs (awsdocs-neuron.readthedocs-hosted.com, nki/api/generated/nki.isa.*),
# attached once per kind of problem so the model sees the rule in the docs' own words.
DOCS = (
    ("nc_matmul", "psum", "nki.isa.nc_matmul docs: \"must read inputs from SBUF and write outputs to PSUM. Therefore, "
     "the stationary and moving must be SBUF tiles, and dst tile must be a PSUM tile.\""),
    ("sums over the FIRST", "", "nki.isa.nc_matmul docs: \"computes dst = stationary.T @ moving... to multiply [M, K] "
     "and [K, N] you need to pass shapes [K, M] and [K, N], as the partition dimension is always the left-most "
     "dimension.\" K <= 128, stationary free <= 128, moving free <= 512."),
    ("stationary^T x moving", "", "nki.isa.nc_matmul docs: \"computes dst = stationary.T @ moving... the partition "
     "dimension is always the left-most dimension.\" For scores = q k^T, pass stationary=q^T (d, seq) and "
     "moving=k^T (d, seq)."),
    ("nc_transpose", "", "nki.isa.nc_transpose docs: \"Tensor Engine nc_transpose must read the input tile from SBUF "
     "and write the transposed result to PSUM... [128, 128] or smaller, while Vector Engine can handle [32, 32] or "
     "smaller\"; dst must have the same dtype as data."),
    ("dma_copy", "psum", "nki.isa.dma_copy docs: \"Both src and dst tiles can be in HBM or SBUF\"; reading PSUM was "
     "removed in NKI 0.3 -- tensor_copy a psum result to sbuf first."),
    ("copying", "", "Copy docs: tensor_copy needs \"the same partition axis size and also the same number of elements "
     "per partition\"; dma_copy needs \"the total number of data elements in src [to] match that of dst\". "
     "Neither transposes."),
    ("must stay in psum", "", "NKI memory pattern: \"Transfer to SBUF for element-wise ops: "
     "nisa.tensor_copy(dst=sbuf_result, src=psum_result)\"."),
    ("128 rows", "", "NKI tiling docs: an on-chip tile's first (partition) axis is at most 128 (nl.tile_size.pmax)."),
    ("Python operators", "", "NKI ISA: every compute instruction takes an explicit dst= tile; element-wise math "
     "is nisa.tensor_scalar / nisa.tensor_tensor / nisa.activation with op=nl.multiply, nl.add, nl.exp..."),
)


def doc_notes(issues):
    """The doc excerpts that match these issues, each once."""
    out = []
    for key, also, text in DOCS:
        if text not in out and any(key in i and also in i for i in issues):
            out.append(text)
    return out


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


def _is_kernel(fn):
    """@nki.jit / @jit / @nki.jit(...) -- the only function whose arguments arrive in HBM."""
    for d in fn.decorator_list:
        d = d.func if isinstance(d, ast.Call) else d
        if (isinstance(d, ast.Attribute) and d.attr == "jit") or (isinstance(d, ast.Name) and d.id == "jit"):
            return True
    return False


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

    consts = {}                        # NAME = 128 -> '128', so P and 128 compare as the same size
    sibling = {}                       # names unpacked from ONE .shape are different axes of one tensor
    fixed = set()                      # kernel inputs and the returned output: their shapes are the task's
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    kernels = [f for f in fns if _is_kernel(f)] or fns[:1]
    for fn in kernels:                 # only the kernel's own inputs arrive in HBM; a helper's arguments
        for arg in fn.args.args:       # can be any tile, so their memory stays unknown
            where.setdefault(arg.arg, "hbm")
            fixed.add(arg.arg)
        for r in ast.walk(fn):
            if isinstance(r, ast.Return) and _name(r.value):
                fixed.add(_name(r.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            consts[node.targets[0].id] = str(node.value.value)

    def size(d):
        return consts.get(d, d)

    def differ(a, b):
        """True only when two dimensions are PROVABLY different: two different numbers, or two axes of
        one tensor (`seq, d = q.shape`), or 1 against an axis. Differently spelled names may be equal (P vs 128, n vs seq)."""
        a, b = size(a), size(b)
        if a == b:
            return False
        if a.isdigit() and b.isdigit():
            return True
        if "1" in (a, b):              # a (seq, 1) reduction tile is never a whole (seq, d) tile
            return True
        return sibling.get(a) is not None and sibling.get(a) == sibling.get(b)

    def differ_shape(x, y):
        return len(x) == len(y) and any(differ(i, j) for i, j in zip(x, y))

    def big(dims):
        """A dimension proven larger than 32 (the Vector engine's transpose limit)."""
        return any(size(d).isdigit() and int(size(d)) > 32 for d in dims or ())
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
                        decl = node.value.args[0] if node.value.args else next(
                            (kw.value for kw in node.value.keywords if kw.arg == "shape"), None)
                        if _dims(decl):        # nl.ndarray((d, seq), ...) or nl.ndarray(shape=(d, seq), ...)
                            shape[t.id] = _dims(decl)
        # `seq, d = q.shape` gives the input q the shape (seq, d)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape" \
                and isinstance(node.value.value, ast.Name) and isinstance(node.targets[0], ast.Tuple):
            dims = tuple(e.id for e in node.targets[0].elts if isinstance(e, ast.Name))
            if len(dims) == len(node.targets[0].elts):
                shape.setdefault(node.value.value.id, dims)
                for d in dims:
                    sibling.setdefault(d, node.lineno)
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
                rules = RULES[op]
                if op == "nc_transpose" and where.get(_name(given.get("dst"))) == "sbuf" \
                        and not big(shape.get(_name(given.get("dst")))) and not big(shape.get(_name(given.get("data")))):
                    rules = (("dst", {"sbuf", "psum"}), ("data", {"sbuf", "psum"}))   # Vector-engine transpose
                for arg, allowed in rules:
                    if allowed is None or arg not in given:
                        continue
                    tile = _name(given[arg])
                    mem = where.get(tile)
                    if mem and mem not in allowed:
                        need = sorted(allowed)[0] if len(allowed) == 1 else " or ".join(sorted(allowed))
                        fix = f" Allocate it with {FIX[need]}." if need in FIX else ""
                        if op == "dma_copy" and mem == "psum" and arg == "dst":
                            fix = (f" dma_copy into an sbuf tile instead; if an instruction needs the data in "
                                   f"psum, it writes there itself (nc_matmul / nc_transpose dst=).")
                        elif op == "dma_copy" and mem == "psum":
                            dims = ", ".join(shape.get(tile, ("...",)))
                            fix = (f" Copy it to sbuf first: {tile}_sb = nl.ndarray(({dims}), dtype=..., "
                                   f"buffer=nl.sbuf); nisa.tensor_copy(dst={tile}_sb, src={tile}); then "
                                   f"dma_copy {tile}_sb.")
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
                    if differ(k1, k2):
                        issues.append(f"{at(node)} -- nc_matmul sums over the FIRST (partition) axis of both "
                                      f"operands, so they must match: stationary={nm['stationary']} is ({k1}, {m}), "
                                      f"moving={nm['moving']} is ({k2}, {n}). To get A times B-transposed for A "
                                      f"(r, c) and B (r2, c), pass the TRANSPOSES: stationary=A^T (c, r), moving=B^T (c, r2).")
                    elif sh.get("dst") and not differ_shape(sh["dst"], (m, n)):
                        pass
                    elif sh.get("dst") and not differ_shape(sh["dst"], (k1, k2)) and not differ(m, n):
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
                    elif sh.get("dst") and differ_shape(sh["dst"], (m, n)):
                        issues.append(f"{at(node)} -- nc_matmul computes stationary^T x moving: ({k1}, {m}) and "
                                      f"({k2}, {n}) give a ({m}, {n}) result, but dst={nm['dst']} is declared "
                                      f"{tuple(sh['dst'])}. If you wanted A times B-transposed, pass the transposes "
                                      f"of A and B as stationary and moving.")
                if op == "nc_transpose" and sh.get("data") and sh.get("dst") and len(sh["data"]) == 2 \
                        and differ_shape(sh["dst"], tuple(reversed(sh["data"]))):
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
                    if ds and ss and differ_shape(ds, ss):
                        same = sorted(map(size, ds)) == sorted(map(size, ss))
                        if dst.id in fixed:
                            # the output's shape is the task's; advice to re-allocate it breaks the kernel
                            hint = (f" {dst.id} is the kernel's {'output' if dst.id not in where or where[dst.id] == 'hbm' else 'tile'} "
                                    f"and its shape {tuple(ds)} is fixed: do NOT change it. Fix {src.id} instead"
                                    + (f": it holds the transpose of what {dst.id} needs, so transpose it with "
                                       f"nisa.nc_transpose (into psum, then tensor_copy to sbuf) before this copy."
                                       if same else f": compute it with shape {tuple(ds)}."))
                        elif src.id in fixed:
                            hint = (f" {src.id} is a kernel input with shape {tuple(ss)}; allocate {dst.id} with "
                                    f"shape {tuple(ss)}" + ("; a copy does not transpose -- if you need the "
                                    "transpose, use nisa.nc_transpose into a psum tile." if same else "."))
                        else:
                            hint = (" Same elements, different layout: a copy keeps the element order, so the data "
                                    "ends up scrambled, not transposed; use nisa.nc_transpose into a psum tile "
                                    "if you need the transpose." if same else "")
                        issues.append(f"{at(node)} -- copying {src.id} {tuple(ss)} into {dst.id} declared "
                                      f"{tuple(ds)}: shapes must match.{hint}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.Div)):
            l, r = _name(node.left), _name(node.right)
            if (l in where and where[l] != "hbm") or (r in where and where[r] != "hbm"):
                issues.append(f"{at(node)} -- Python operators do not work on tiles; use "
                              f"nisa.tensor_scalar(dst=, data=, op0=nl.multiply / nl.subtract, operand0=) "
                              f"or nisa.tensor_tensor.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            t = node.targets[0].id
            dims = shape.get(t) if where.get(t) in ("sbuf", "psum") else None
            if dims and len(dims) == 1:
                issues.append(f"{at(node)} -- {t} is 1-D; an sbuf/psum tile needs at least 2 dimensions "
                              f"(partition, free). For one value per row use shape (rows, 1); for a plain "
                              f"constant such as 1/sqrt(d), use a Python float (operand0=scale), not a tile.")
            if dims and size(dims[0]).isdigit() and int(size(dims[0])) > 128:
                issues.append(f"{at(node)} -- {t} has {size(dims[0])} rows; an on-chip tile has at most 128 "
                              f"(the first axis is the partition axis). Split it into 128-row tiles in a loop.")
            if dims and where.get(t) == "psum" and len(dims) == 2 and size(dims[1]).isdigit() \
                    and int(size(dims[1])) > 512:
                issues.append(f"{at(node)} -- psum tile {t} has {size(dims[1])} columns; one psum bank holds "
                              f"at most 512 fp32 per row. Split the free axis into blocks of 512.")

    # Invented instructions and keywords. Measured: ~15% of failures were names the model made up
    # (nisa.sum, transpose_stationary=True), and the simulator reports only the first one per round.
    if _nisa is not None:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "nisa"):
                continue
            op, fn_ = node.func.attr, getattr(_nisa, node.func.attr, None)
            if fn_ is None:
                close = difflib.get_close_matches(op, [n for n in dir(_nisa) if not n.startswith("_")], n=3)
                extra = (" For a sum or max along an axis use nisa.tensor_reduce(dst=, op=nl.add / nl.maximum, "
                         "data=, axis=, keepdims=True)." if op in ("sum", "max", "reduce", "mean") else "")
                issues.append(f"{at(node)} -- nisa.{op} does not exist"
                              + (f"; did you mean {', '.join('nisa.' + c for c in close)}?" if close else ".") + extra)
                continue
            try:
                params = inspect.signature(fn_).parameters
            except (TypeError, ValueError):
                continue
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue
            bad = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in params]
            if bad:
                issues.append(f"{at(node)} -- nisa.{op} has no argument {', '.join(bad)}. Its real signature: "
                              f"nisa.{op}({', '.join(n for n in params if n != 'name')}).")

    # Read before write: a tile from nl.ndarray holds garbage until something writes it. Measured on level
    # 8: a repair moved an activation to dst=y while the next line still read e, which nothing wrote --
    # the simulator then fails far away, on numbers, not on the line that broke.
    for fn in kernels:
        fresh = {}                     # tile -> line of an nl.ndarray allocation (uninitialised)
        writes, reads = {}, {}         # tile -> first line written / read
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "ndarray":
                for t in node.targets:
                    if isinstance(t, ast.Name) and where.get(t.id) in ("sbuf", "psum"):
                        fresh.setdefault(t.id, node.lineno)
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    if _name(t):       # e[...] = ... or e = nl.sum(...)
                        writes[_name(t)] = min(writes.get(_name(t), 10**9), node.lineno)
            if isinstance(node, ast.Call):
                f = node.func
                lib = isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("nisa", "nl")
                for i, arg in enumerate(node.args):
                    t = _name(arg)
                    if not t:
                        continue
                    if not lib:        # a helper may write any tile it is given
                        writes[t] = min(writes.get(t, 10**9), node.lineno)
                    elif i == 0 and (f.value.id == "nisa" or f.attr == "store"):
                        writes[t] = min(writes.get(t, 10**9), node.lineno)
                    else:
                        reads.setdefault(t, []).append(node)
                for kw in node.keywords:
                    t = _name(kw.value)
                    if not t:
                        continue
                    if kw.arg == "dst" or not lib:
                        writes[t] = min(writes.get(t, 10**9), node.lineno)
                    else:
                        reads.setdefault(t, []).append(node)
        for t, line in fresh.items():
            first = min(reads.get(t, []), key=lambda n: n.lineno, default=None)
            if first is not None and first.lineno < writes.get(t, 10**9):
                later = writes.get(t)
                issues.append(f"{at(first)} -- reads {t}, but nothing has written {t} yet (allocated on line "
                              f"{line}" + (f", first written on line {later}" if later else ", never written")
                              + f"): it holds garbage. Write {t} first (dst={t}), or read the tile that "
                              f"actually holds this value.")

    seen, out = set(), []
    for i in issues:                   # one report per line and problem
        if i not in seen:
            seen.add(i); out.append(i)
    return out
