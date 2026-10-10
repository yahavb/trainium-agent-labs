"""gate_nki.py: a kernel the simulator accepts but the trn2 compiler rejects is not solved (GATE=static or
GATE=compile; off by default). feedback_v7 wraps agent.grade with it.

Task 08: the simulator accepted 39 distinct solves of new operations; 4 compiled and matched in birsim.
34 of the 35 failures used one of two forms, and the model kept writing them even when the prompt said
not to (nl.divide in 37 of 80 samples). So on a kernel that passes every shape:
  * GATE=static  finds those forms in the source and rewrites the line as code, the way the NumPy
                 messages rewrite a banned call (pasted literally in 210 of 212 samples, task 10):
                   nl.divide(A, B)            -> nisa.reciprocal into `inv`, then a multiply
                   tensor_tensor with a (rows, 1) column operand -> nisa.tensor_scalar with it as operand0
  * GATE=compile also lowers a kernel with no such form for trn2 (when neuronx-cc is on PATH), and
                 passes on the compiler's reason.
The kernel then scores FULL - 0.05: not solved, still the best so far, so the next round repairs it.
"""
import ast
import os
import re
import shutil

import feedback_v6 as v6

MARK = "the trn2 compiler rejects"


def _col(code, name):
    return bool(re.match(r"^[A-Za-z_]\w*$", name or "")) and v6.is_column(code, name)


def _is(call, mod, name):
    f = call.func
    return isinstance(f, ast.Attribute) and f.attr == name and isinstance(f.value, ast.Name) and f.value.id == mod


def fixes(code):
    """[(first line, last line, the original lines, replacement lines or None, why)] for the forms the
    trn2 compiler rejects. Parsed with ast, so arguments like `a[i, j]` and calls over several lines work."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    seg = lambda n: ast.get_source_segment(code, n)    # noqa: E731
    out, done = [], set()
    for st in ast.walk(tree):
        if not isinstance(st, (ast.Assign, ast.Expr)) or not isinstance(st.value, ast.Call):
            continue
        call, a0, a1 = st.value, st.lineno, st.end_lineno
        ind = re.match(r"\s*", lines[a0 - 1]).group(0)
        old = "\n".join(lines[a0 - 1:a1])
        if _is(call, "nl", "divide") and len(call.args) == 2 and isinstance(st, ast.Assign) \
                and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            x, a, b = st.targets[0].id, seg(call.args[0]), seg(call.args[1])
            simple_b = isinstance(call.args[1], ast.Name)
            if isinstance(call.args[0], ast.Constant) and call.args[0].value == 1:
                new = [f"{x} = nl.ndarray({b}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                       f"nisa.reciprocal(dst={x}, data={b})"] if simple_b else None
            elif simple_b and _col(code, b):
                new = [f"inv = nl.ndarray({b}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                       f"nisa.reciprocal(dst=inv, data={b})",
                       f"{x} = nl.ndarray({a}.shape, dtype={a}.dtype, buffer=nl.sbuf)",
                       f"nisa.tensor_scalar(dst={x}, data={a}, op0=nl.multiply, operand0=inv)"]
            elif simple_b:
                new = [f"inv = nl.ndarray({b}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                       f"nisa.reciprocal(dst=inv, data={b})",
                       f"{x} = nl.multiply({a}, inv)"]
            else:
                new = None
            out.append((a0, a1, old, [ind + l for l in new] if new else None,
                        "nl.divide: compute the reciprocal with nisa.reciprocal, then multiply"))
            done.add(a0)
        elif (_is(call, "nisa", "tensor_scalar") or _is(call, "nisa", "tensor_tensor")) and any(
                k.arg in ("op", "op0") and seg(k.value).replace(" ", "") == "nl.divide" for k in call.keywords):
            kw = {k.arg: k.value for k in call.keywords if k.arg}
            opk, dk = ("op0", "operand0") if "op0" in kw else ("op", "data2")
            den = kw.get(dk)
            if isinstance(den, ast.Constant) and isinstance(den.value, (int, float)) and opk == "op0":
                body = re.sub(r"op0\s*=\s*nl\.divide", "op0=nl.multiply", old, count=1)
                body = re.sub(rf"operand0\s*=\s*{re.escape(seg(den))}", f"operand0={1.0 / den.value!r}", body, count=1)
                out.append((a0, a1, old, body.splitlines(), "nl.divide as an operation: multiply by the reciprocal"))
            elif isinstance(den, ast.Name) and not any(k.arg == "reverse0" for k in call.keywords):
                body = re.sub(rf"{opk}\s*=\s*nl\.divide", f"{opk}=nl.multiply", old, count=1)
                body = re.sub(rf"{dk}\s*=\s*{den.id}\b", f"{dk}=inv", body, count=1)
                out.append((a0, a1, old, [f"{ind}inv = nl.ndarray({den.id}.shape, dtype=nl.float32, buffer=nl.sbuf)",
                                          f"{ind}nisa.reciprocal(dst=inv, data={den.id})"] + body.splitlines(),
                            "nl.divide as an operation: compute the reciprocal with nisa.reciprocal, then multiply"))
            else:
                out.append((a0, a1, old, None, "nl.divide as an operation: compute the reciprocal with "
                                               "nisa.reciprocal, then use nl.multiply"))
            done.add(a0)
        elif _is(call, "nisa", "tensor_tensor"):
            kw = {k.arg: k.value for k in call.keywords if k.arg}
            d1, d2 = kw.get("data1"), kw.get("data2")
            if "dst" not in kw or "op" not in kw:
                continue
            dst, op = seg(kw["dst"]), seg(kw["op"])
            n1 = d1.id if isinstance(d1, ast.Name) else None
            n2 = d2.id if isinstance(d2, ast.Name) else None
            if n2 and _col(code, n2):
                new = [f"{ind}nisa.tensor_scalar(dst={dst}, data={seg(d1)}, op0={op}, operand0={n2})"]
                why = f"`{n2}` is a (rows, 1) column; tensor_tensor with a broadcast column"
            elif n1 and _col(code, n1):
                new = [f"{ind}nisa.tensor_scalar(dst={dst}, data={seg(d2)}, op0={op}, operand0={n1}, reverse0=True)"]
                why = f"`{n1}` is a (rows, 1) column; tensor_tensor with a broadcast column"
            else:
                continue
            out.append((a0, a1, old, new, why))
            done.add(a0)
    for node in ast.walk(tree):                        # any other nl.divide: name it, no code
        if isinstance(node, ast.Call) and _is(node, "nl", "divide") and node.lineno not in done:
            out.append((node.lineno, node.lineno, lines[node.lineno - 1], None,
                        "nl.divide: compute the reciprocal with nisa.reciprocal, then multiply "
                        "(nisa.tensor_scalar with it as operand0 when it has one value per row)"))
            done.add(node.lineno)
    out += partition_fixes(code) + dst_shape_fixes(code)
    return sorted(out)


# --- l1fix (task 15): two more forms the simulator accepts and the trn2 compiler rejects ---------------
# Measured on the 17 distinct level-1 simulator solves in the 4090 rehearsal logs, full build on seat-115:
# 6 loop over channels and give a compute instruction ONE partition starting at partition c ("Invalid
# access of 1 partitions starting at partition 1 (TensorReduce)", BIR verification); 3 write a dst that
# holds more elements than the result (tensor_scalar dst (C, p, p) for a (C, 1, 1) sum; tensor_reduce).

LOOPS = {"affine_range", "sequential_range", "static_range", "range"}
HBM = {"shared_hbm", "hbm", "private_hbm"}


def _loop_var(st):
    it = st.iter
    if isinstance(st, ast.For) and isinstance(st.target, ast.Name) and isinstance(it, ast.Call):
        f = it.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
        if name in LOOPS and len(it.args) == 1:
            return st.target.id
    return None


def _hbm_names(tree):
    names = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            names |= {a.arg for a in fn.args.args}
    for st in ast.walk(tree):
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                and isinstance(st.value, ast.Call) and _is(st.value, "nl", "ndarray"):
            buf = {k.arg: k.value for k in st.value.keywords}.get("buffer")
            if isinstance(buf, ast.Attribute) and buf.attr in HBM:
                names.add(st.targets[0].id)
    return names


def _uses(node, var):
    return any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(node))


def _one_partition(sub, var):
    """Whether subscript `sub` puts loop variable `var` on the partition axis as ONE partition:
    t[c, ...] or t[c:c + 1, ...]."""
    idx = sub.slice.elts[0] if isinstance(sub.slice, ast.Tuple) and sub.slice.elts else sub.slice
    if isinstance(idx, ast.Slice):
        lo, hi = idx.lower, idx.upper
        return (lo is not None and hi is not None and _uses(lo, var) and isinstance(hi, ast.BinOp)
                and isinstance(hi.op, ast.Add) and ast.dump(hi.left) == ast.dump(lo)
                and isinstance(hi.right, ast.Constant) and hi.right.value == 1)
    return _uses(idx, var)


def _base(e):
    while isinstance(e, ast.Subscript):
        e = e.value
    return e.id if isinstance(e, ast.Name) else None


def _dma_only(tree, sub, parents, hbm):
    """A view that only DMA reads or writes: an argument of nisa.dma_copy, one side of an assignment
    whose other side is in HBM (`tile[p, i] = x[p, i]` is a DMA), or a name used only in nisa.dma_copy."""
    par = parents.get(id(sub))
    if isinstance(par, ast.keyword):
        par = parents.get(id(par))
    if isinstance(par, ast.Call) and _is(par, "nisa", "dma_copy"):
        return True
    if isinstance(par, ast.Assign) and len(par.targets) == 1 and isinstance(par.targets[0], ast.Subscript) \
            and (par.targets[0] is sub or par.value is sub) \
            and (_base(par.targets[0]) in hbm or _base(par.value) in hbm):
        return True
    if isinstance(par, ast.Assign) and par.value is sub and len(par.targets) == 1 \
            and isinstance(par.targets[0], ast.Name):
        v, uses = par.targets[0].id, []
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id == v and isinstance(n.ctx, ast.Load):
                p = parents.get(id(n))
                p = parents.get(id(p)) if isinstance(p, ast.keyword) else p
                uses.append(isinstance(p, ast.Call) and _is(p, "nisa", "dma_copy"))
        return bool(uses) and all(uses)
    return False


def _one_cell(sub, first, scalar_to_slice):
    """sub with its first index replaced by `first`, and each other plain index i made i:i+1."""
    elts = list(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else [sub.slice]
    new = [first]
    for e in elts[1:]:
        if scalar_to_slice and isinstance(e, ast.Constant) and isinstance(e.value, int):
            e = ast.Slice(lower=e, upper=ast.Constant(e.value + 1))
        elif scalar_to_slice and not isinstance(e, ast.Slice):
            e = ast.Slice(lower=e, upper=ast.BinOp(left=e, op=ast.Add(), right=ast.Constant(1)))
        new.append(e)
    return ast.Subscript(value=sub.value, slice=ast.Tuple(elts=new, ctx=ast.Load()), ctx=sub.ctx)


class _AllChannels(ast.NodeTransformer):
    """The body of `for c in nl.affine_range(C):` with every channel at once: t[c, ...] -> t[:, ...],
    a (1, ...) allocation -> (C, ...), and `a[c, i, j] = b` -> a copy instruction into a[:, i:i+1, j:j+1]."""

    def __init__(self, var, count, hbm):
        self.var, self.count, self.hbm, self.ok = var, count, hbm, True

    def visit_Subscript(self, node):
        self.generic_visit(node)
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if elts and ((isinstance(elts[0], ast.Name) and elts[0].id == self.var) or (
                isinstance(elts[0], ast.Slice) and isinstance(elts[0].lower, ast.Name)
                and elts[0].lower.id == self.var and _one_partition(node, self.var))):
            return _one_cell(node, ast.Slice(), False)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if _is(node, "nl", "ndarray") and node.args and isinstance(node.args[0], ast.Tuple) \
                and node.args[0].elts and isinstance(node.args[0].elts[0], ast.Constant) \
                and node.args[0].elts[0].value == 1:
            node.args[0].elts[0] = self.count
        return node

    def visit_Assign(self, node):
        tgt = node.targets[0] if len(node.targets) == 1 else None
        if isinstance(tgt, ast.Subscript) and _uses(tgt, self.var):
            elts = tgt.slice.elts if isinstance(tgt.slice, ast.Tuple) else [tgt.slice]
            if not (elts and isinstance(elts[0], ast.Name) and elts[0].id == self.var):
                self.ok = False
                return node
            dst = _one_cell(tgt, ast.Slice(), True)
            src = node.value
            if isinstance(src, ast.Subscript):
                s0 = src.slice.elts[0] if isinstance(src.slice, ast.Tuple) else src.slice
                if (isinstance(s0, ast.Name) and s0.id == self.var) or \
                        (isinstance(s0, ast.Constant) and s0.value == 0):
                    s_elts = src.slice.elts if isinstance(src.slice, ast.Tuple) else [src.slice]
                    if all(isinstance(e, ast.Constant) for e in s_elts[1:]):
                        src = _one_cell(src, ast.Slice(), True)
                    else:
                        src = _one_cell(src, ast.Slice(), False)
            src = self.visit(src)
            dst.value = self.visit(dst.value)
            hbm = isinstance(tgt.value, ast.Name) and tgt.value.id in self.hbm
            call = ast.Call(func=ast.Attribute(value=ast.Name("nisa", ast.Load()),
                                               attr="dma_copy" if hbm else "tensor_copy", ctx=ast.Load()),
                            args=[], keywords=[ast.keyword("dst", dst), ast.keyword("src", src)])
            return ast.copy_location(ast.Expr(call), node)
        self.generic_visit(node)
        return node


def partition_fixes(code):
    """A loop variable as the partition index of an on-chip tile (t[c, ...] for c in a loop) gives a
    compute instruction ONE partition starting at partition c; the trn2 compiler rejects that (BIR
    verification), the simulator does not. A view that only DMA moves is fine (x[c] from HBM is too)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    parents = {}
    for n in ast.walk(tree):
        for ch in ast.iter_child_nodes(n):
            parents[id(ch)] = n
    hbm = _hbm_names(tree)
    out = []
    for loop in ast.walk(tree):
        var = _loop_var(loop) if isinstance(loop, ast.For) else None
        if not var:
            continue
        bad = [n for st in loop.body for n in ast.walk(st)
               if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id not in hbm
               and _one_partition(n, var) and not _dma_only(tree, n, parents, hbm)]
        if not bad:
            continue
        tile = bad[0].value.id
        ind = re.match(r"\s*", lines[loop.lineno - 1]).group(0)
        old = "\n".join(lines[loop.lineno - 1:loop.end_lineno])
        fix = _AllChannels(var, loop.iter.args[0], hbm)
        body = [fix.visit(ast.parse(ast.unparse(st)).body[0]) for st in loop.body]
        uses_left = any(_uses(st, var) for st in body)
        new = None
        if fix.ok and not uses_left and not loop.orelse:
            new = []
            for st in body:
                ast.fix_missing_locations(st)
                new += [ind + l for l in ast.unparse(st).splitlines()]
        cnt = ast.get_source_segment(code, loop.iter.args[0])
        why = (f"`{tile}[{var}, ...]` gives an instruction one partition starting at partition `{var}`; on "
               f"the chip every compute instruction must start at partition 0. Do all {cnt} rows at once "
               f"with `{var}` gone: `{tile}[:, ...]`, and (1, ...) tiles become ({cnt}, ...)")
        out.append((loop.lineno, loop.end_lineno, old, new, why))
    return out


def _shape(tree, name, upto):
    """The shape of `name` as a tuple of source strings, from its last assignment before line `upto`:
    nl.ndarray((a, b, ..)) or a keepdims reduction nl.sum/max/min/mean(t, axis=[..], keepdims=True)."""
    best = None
    for st in ast.walk(tree):
        if isinstance(st, ast.Assign) and st.lineno < upto and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name) and st.targets[0].id == name \
                and isinstance(st.value, ast.Call) and (best is None or st.lineno > best.lineno):
            best = st
    if best is None:
        return None
    c = best.value
    kw = {k.arg: k.value for k in c.keywords}
    if _is(c, "nl", "ndarray") and c.args and isinstance(c.args[0], ast.Tuple):
        return tuple(ast.unparse(e) for e in c.args[0].elts)
    if any(_is(c, "nl", f) for f in ("sum", "max", "min", "mean")) and c.args and isinstance(c.args[0], ast.Name) \
            and isinstance(kw.get("keepdims"), ast.Constant) and kw["keepdims"].value is True \
            and isinstance(kw.get("axis"), (ast.List, ast.Tuple)):
        src = _shape(tree, c.args[0].id, best.lineno)
        axes = [e.value for e in kw["axis"].elts if isinstance(e, ast.Constant)]
        if src and len(axes) == len(kw["axis"].elts) and all(0 <= a < len(src) for a in axes):
            return tuple("1" if i in axes else d for i, d in enumerate(src))
    return None


def dst_shape_fixes(code):
    """nisa.tensor_scalar / activation / tensor_copy whose dst has a different shape from its input: the
    simulator writes the first elements, the trn2 compiler rejects it ('dst' free total elements 9 !=
    'src' free total elements 1)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = code.splitlines()
    out = []
    for st in ast.walk(tree):
        if not isinstance(st, ast.Expr) or not isinstance(st.value, ast.Call):
            continue
        call = st.value
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        src_k = ("data" if any(_is(call, "nisa", f) for f in ("tensor_scalar", "activation")) else
                 "src" if _is(call, "nisa", "tensor_copy") else None)
        if not src_k or not isinstance(kw.get("dst"), ast.Name) or not isinstance(kw.get(src_k), ast.Name):
            continue
        d, s = kw["dst"].id, kw[src_k].id
        ds, ss = _shape(tree, d, st.lineno), _shape(tree, s, st.lineno)
        if not ds or not ss or len(ds) != len(ss) or ds == ss:
            continue
        if not any((a == "1") != (b == "1") for a, b in zip(ds, ss)):
            continue                                  # can't tell the sizes apart from the source
        out.append((st.lineno, st.end_lineno, "\n".join(lines[st.lineno - 1:st.end_lineno]), None,
                    f"dst `{d}` has shape ({', '.join(ds)}) but `{s}` has shape ({', '.join(ss)}); the chip "
                    f"needs dst to hold exactly as many elements as `{s}`. Write into a new tile, "
                    f"`r = nl.ndarray({s}.shape, dtype={s}.dtype, buffer=nl.sbuf)`, and read the result "
                    f"from `r` afterwards"))
    return out


def lower_error(kernel_src, level, agent):
    """The compiler's reason if this kernel fails to lower for trn2, else None (also None if no compiler)."""
    if not shutil.which("neuronx-cc"):
        return None
    try:
        import nkibench
        from nkitool import analyze
        path = f"/tmp/_gate_{os.getpid()}.py"
        open(path, "w").write(kernel_src)
        k = nkibench.load_kernel(path, nkibench.LEVELS[level]["entry"])
        args, _ = nkibench.make_inputs(nkibench.LEVELS[level]["shapes"][0], level)
        analyze(k, *args, neff=False)
        return None
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")
        j = msg.find("error:")
        return msg[j:j + 200] if j >= 0 else msg[:200]


def message(code, found):
    parts = [f"Correct in the simulator on every shape, but {MARK} it: the chip has no such instruction, so "
             f"this kernel cannot run on the device. Fix exactly these lines:"]
    for a0, a1, old, new, why in found:
        where = f"Line {a0}" if a0 == a1 else f"Lines {a0}-{a1}"
        if new:
            parts.append(f"{where} ({why}). Replace:\n{old}\nwith:\n" + "\n".join(new))
        else:
            parts.append(f"{where}: {why}.\n{old.strip()}")
    parts.append("Keep everything else identical.")
    return "\n\n".join(parts)


def install(agent, mode):
    their = agent.grade
    full = sum(agent.WEIGHTS.values())

    def gated(source, level):
        reward, parts, note = their(source, level)
        if reward < full - 1e-9:
            return reward, parts, note
        found = fixes(source)
        if found:
            return full - 0.05, dict(parts, correct=False), message(source, found)
        if mode == "compile":
            why = lower_error(source, level, agent)
            if why:
                return full - 0.05, dict(parts, correct=False), (
                    f"Correct in the simulator on every shape, but {MARK} it for trn2: {why}\n"
                    f"Change the line the error points at; keep everything else identical.")
        return reward, parts, note

    agent.grade = gated
