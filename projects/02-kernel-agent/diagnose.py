#!/usr/bin/env python3
"""
diagnose.py -- say WHERE in the kernel a failure happened, not only what the simulator said.

The harness reports a crash as, for example,

    raised ValueError: cannot reshape array of size 32768 into shape (1,64)

That sentence comes from inside the simulator. It describes the simulator's symptom, and whoever
wrote the kernel has to guess which of their own lines caused it. A kernel that never calls
reshape can still produce it, and "do not reshape" is then an instruction about code that is not
there. The missing half is cheap and exact: the traceback already knows which line of the
candidate was executing. This module reads it out, so the feedback can say

    line 23, `nisa.nc_matmul(dst=res, stationary=lhs_tile, moving=rhs_tile)`, raised ValueError: ...

It names a line of the model's OWN code. It never says what the line should be.

The second half is what was true on that line when it failed. The kernel's frame is still in the
traceback with every variable in it, so the shapes are not guesses:

    At that line: `result_psum` is (1, 64) in psum; `lhs_tile` is (128, 64) in sbuf; `rhs_tile`
    is (128, 512) in sbuf. nc_matmul's result has stationary's second dimension as its first ...

Shapes and values are the kernel's own, read at the moment of failure. A rule is added only when
those shapes break it, and it is stated as the rule -- which dimension must match which -- never
as the shape to type in. Nothing here can reject a kernel: it speaks only after the simulator
has already raised.

    python diagnose.py --selftest      # plant known bugs in the reference kernels, check both
    python diagnose.py --explain FILE --level N    # the diagnosis for your own kernel, as the
                                                   # model would get it. For people writing NKI.
    python diagnose.py --replay LOG    # what the model WOULD have been told, for every failing
                                       # kernel in an attempt log. Calls no model.

WHAT IS VERIFIED AND WHAT IS NOT. --selftest needs `nki`, so it proves something only where the
Neuron SDK is installed: run it in the seat pod before trusting a located line. devtools/fake_nki
is a stand-in for exercising the plumbing on a laptop and says nothing about the real simulator.
"""

import argparse
import ast
import math
import os
import re
import sys
import types

# Marks the located text inside a feedback string, so it can be shown to the model and also
# removed again. See shown() and key().
MARK = "\x00"

# where() and state() run inside the grader's error handler, in the middle of a measured run that
# takes the better part of an hour. A bug in them must cost one sentence of feedback, never the
# run. So they swallow their own failures and note them here, where --replay reports them.
ERRORS = []


def statement_at(source, lineno, limit=200):
    """The statement on `lineno`, on one line. A call split across several lines comes back whole;
    for a loop or an `if`, only its header line, since its body is not what failed."""
    lines = source.splitlines()
    plain = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.stmt):
                end = node.end_lineno or node.lineno
                if node.lineno <= lineno <= end and (best is None
                                                    or end - node.lineno <= best[0]):
                    best = (end - node.lineno, node)
    except SyntaxError:
        pass
    text = plain
    if best is not None and not hasattr(best[1], "body"):
        text = ast.get_source_segment(source, best[1]) or plain
    text = " ".join(text.split())
    return text[:limit] + (" ..." if len(text) > limit else "")


def _function_spans(source):
    try:
        return {n.name: (n.lineno, n.end_lineno or n.lineno) for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return {}


def _hit(exc, path, source=""):
    """(line, frame) of the candidate file that was executing when `exc` was raised, or None.

    Takes the INNERMOST frame that belongs to the candidate: if the kernel calls a helper of its
    own, the helper's line is the specific one. Walks the exception chain as well, because a
    simulator that catches an error and raises its own from above the kernel leaves the kernel's
    frame only on the original.
    """
    real = os.path.realpath(path)
    chain, seen, e = [], set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or e.__context__
    spans = None
    for e in reversed(chain):
        hit, tb = None, e.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            if code.co_filename == path or os.path.realpath(code.co_filename) == real:
                hit = (tb.tb_lineno, tb.tb_frame)
            elif source:
                # A decorator may recompile the function under another file name while keeping
                # its line numbers. Accept that only when the line really is inside a function
                # of that name in the candidate, so a simulator frame can never be mistaken for it.
                if spans is None:
                    spans = _function_spans(source)
                lo, hi = spans.get(code.co_name, (0, -1))
                if lo <= tb.tb_lineno <= hi and not os.path.exists(code.co_filename):
                    hit = (tb.tb_lineno, tb.tb_frame)
            tb = tb.tb_next
        if hit:
            return hit
    return None


def locate(exc, path, source=""):
    """The line of the candidate file that was executing when `exc` was raised, or None."""
    hit = _hit(exc, path, source)
    return hit[0] if hit else None


def where(exc, path, source):
    """'line N, `statement`, ' wrapped in MARK, or '' when the line cannot be found. Empty rather
    than a guess: a wrong line is worse than no line."""
    try:
        n = locate(exc, path, source)
        if not n:
            return ""
        return f"{MARK}line {n}, `{statement_at(source, n)}`, {MARK}"
    except Exception as e:
        ERRORS.append(f"where: {type(e).__name__}: {e}")
        return ""


def shown(text):
    """The feedback as the model sees it."""
    return text.replace(MARK, "")


def key(text):
    """The feedback with the located part removed: what an unmodified run would have said."""
    return re.sub(f"{MARK}.*?{MARK}", "", text, flags=re.S)


# ---------------------------------------------------------------- what was true on that line

REGIONS = ("sbuf", "psum", "shared_hbm", "private_hbm", "hbm")

# Positional order of the operands, for the calls whose rules are worth stating.
OPERANDS = {"dma_copy": ("dst", "src"), "tensor_copy": ("dst", "src"),
            "nc_matmul": ("dst", "stationary", "moving"), "ndarray": ("shape", "dtype", "buffer")}


def _region_name(buf):
    text = f"{getattr(buf, 'name', '')} {buf!r}".lower()
    return next((r for r in REGIONS if r in text), None) if buf is not None else None


class watching:
    """For the length of one simulation, remember which buffer each tile was allocated in.

    A tile's shape can be read off the tile. Where it lives is known for certain only at the
    moment it is allocated, from the buffer= it was given, so note it there. Same technique as
    the harness counting bytes by wrapping nisa.dma_copy, and the same caveat: it assumes the
    kernel calls nl.ndarray through the module. If it does not, nothing is recorded and the
    report simply leaves the buffer out.
    """

    def __init__(self, path=None):
        self.buffers, self._keep, self._nl = {}, [], None
        # (line, shape, buffer) of every on-chip tile allocated with more rows than a tile can
        # have. See tall().
        self.tall, self._path = [], path

    def __enter__(self):
        try:
            import nki.language as nl
        except ImportError:
            return self
        self._nl, self._real = nl, nl.ndarray
        known = {id(getattr(nl, r)): r for r in REGIONS if hasattr(nl, r)}
        real = self._real

        def ndarray(*args, **kw):
            out = real(*args, **kw)
            try:
                buf = kw.get("buffer", args[2] if len(args) > 2 else None)
                name = known.get(id(buf)) or _region_name(buf)
                if name:
                    self.buffers[id(out)] = name
                    self._keep.append(out)      # so the id cannot be reused while we look
                rows = _shape(out)
                where = name or ("sbuf" if buf is None else None)   # sbuf is the default buffer
                if rows and where in ("sbuf", "psum") and rows[0] > _pmax():
                    caller = sys._getframe(1)
                    mine = self._path is None or \
                        os.path.realpath(caller.f_code.co_filename) == os.path.realpath(self._path)
                    self.tall.append((caller.f_lineno if mine else None, rows, where))
            except Exception:
                pass
            return out

        nl.ndarray = ndarray
        return self

    def __exit__(self, *exc):
        if self._nl is not None:
            self._nl.ndarray = self._real
        return False


def _pmax():
    try:
        import nkibench
        return nkibench.PMAX
    except Exception:
        return 128


def output_on_chip(source, entry):
    """The message for a kernel that returns an on-chip tile as its output, or ''.

    Found in the logs: three level 3 runs scored 1.0 with a kernel that returns its PSUM tile
    directly, with no copy out to HBM. The simulator hands back the numbers either way. Every
    reference kernel returns a tensor allocated in shared_hbm, which is where a kernel's output
    has to be on the device. Read from the source: the returned name and the buffer it was
    allocated with. Anything less plain than `return name` is left alone rather than guessed at.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == entry),
              None)
    if fn is None:
        return ""
    returned = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.Name)]
    if len(returned) != 1:
        return ""
    name, where = returned[0].value.id, None
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == name:
            where = None
            if isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "ndarray":
                buf = next((k.value for k in n.value.keywords if k.arg == "buffer"),
                           n.value.args[2] if len(n.value.args) > 2 else None)
                where = "sbuf" if buf is None else _region_name(ast.unparse(buf))
                line = n.lineno
    if where not in ("sbuf", "psum"):
        return ""
    return (f"RUNS ON THE SIMULATOR, NOT ON THE CHIP: the kernel returns `{name}`, which line "
            f"{line} allocates in {where}. A kernel's output has to be in HBM: allocate it with "
            f"buffer=nl.shared_hbm and copy the result into it -- from psum to sbuf with "
            f"nisa.tensor_copy, then from sbuf to the output with nisa.dma_copy.")


def tall(watch, source):
    """The message for an on-chip tile allocated with more rows than a tile can have, or ''.

    Found in the logs, not predicted. Every kernel but one that scored 0.75 on level 4 allocates
    an SBUF or PSUM tile with 256 or 512 rows and then only ever hands 128-row slices of it to an
    instruction. The simulator checks the piece an instruction is given, never the tile it was
    cut from, so those kernels run and their numbers are right. The harness's rule scan does
    forbid a partition dimension over 128, but it reads the source and can only see a literal
    number: `nl.ndarray((256, 512), ...)` is caught, `nl.ndarray((K, N), ...)` is not. So the
    score rewarded a tile the chip does not have. This sees the allocation as it happens, with
    the real shape, and says so.
    """
    seen = getattr(watch, "tall", None)
    if not seen:
        return ""
    line, shape, where = seen[0]
    at = (f"line {line}, `{statement_at(source, line)}`, allocates" if line
          else "the kernel allocates")
    return (f"RUNS ON THE SIMULATOR, NOT ON THE CHIP: {at} a tile in {where} with {shape[0]} "
            f"rows. A tile on the chip has at most {_pmax()} rows, and slicing it afterwards "
            f"does not change that: the simulator checks the piece an instruction is given, not "
            f"the tile it was cut from. Allocate the tile inside the loop with one chunk's rows, "
            f"and load one chunk into it at a time.")


def _shape(val):
    try:
        shape = tuple(int(d) for d in val.shape)
    except Exception:
        return None
    return shape if not isinstance(val, type) else None


def _buffer(val, name, watch, params):
    for attr in ("buffer", "_buffer", "region", "memory"):
        try:
            found = _region_name(getattr(val, attr, None))
        except Exception:
            found = None
        if found:
            return found
    if watch is not None and id(val) in watch.buffers:
        return watch.buffers[id(val)]
    return "hbm" if name in params else None     # an argument of the kernel is the caller's tensor


def _where_text(buf, name, params):
    if buf is None:
        return ""
    if "hbm" in buf:
        return " in HBM (an input)" if name in params else " in HBM"
    return f" in {buf}"


_SAFE_NODES = (ast.Expression, ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add,
               ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.Subscript, ast.Slice,
               ast.Tuple, ast.Attribute, ast.Call)
_SAFE_CALLS = {"min", "max", "int", "len", "abs"}


def _evaluate(node, frame):
    """Evaluate one operand expression in the kernel's own frame, or return None.

    Only expressions that cannot do anything are evaluated: names, numbers, arithmetic, slices,
    attribute reads and min/max. `a[i*128:(i+1)*128, :]` is; a call that allocates is not.
    """
    for n in ast.walk(node):
        if not isinstance(n, _SAFE_NODES):
            return None
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name)
                                            and n.func.id in _SAFE_CALLS and not n.keywords):
            return None
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            return None
    try:
        expr = ast.fix_missing_locations(ast.Expression(body=node))
        return eval(compile(expr, "<operand>", "eval"), frame.f_globals, dict(frame.f_locals))
    except Exception:
        return None


def _statement_node(source, lineno):
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.stmt):
                end = node.end_lineno or node.lineno
                if node.lineno <= lineno <= end and (best is None
                                                    or end - node.lineno <= best[0]):
                    best = (end - node.lineno, node)
    except SyntaxError:
        return None
    return best[1] if best else None


def _own_expressions(stmt):
    """The parts of a statement that are its own: for a loop or an `if`, the header and not the
    body, because the body is other lines."""
    if not hasattr(stmt, "body"):
        return [stmt]
    return [v for k, v in ast.iter_fields(stmt)
            if k not in ("body", "orelse", "finalbody", "handlers") and isinstance(v, ast.AST)]


def _rules(func, ops, limits):
    """Sentences for the rules these operands actually break. Empty when none is broken, so a
    rule is never recited at a kernel that obeys it. Each names which dimension must match
    which; none gives the shape to write."""
    pmax, smax, mmax = limits
    out = []
    shape = {k: v[1] for k, v in ops.items()}
    buf = {k: v[2] for k, v in ops.items()}
    if func in ("dma_copy", "tensor_copy") and shape.get("dst") and shape.get("src"):
        nd, ns = math.prod(shape["dst"]), math.prod(shape["src"])
        if nd != ns:
            out.append(f"The two sides of a {func} must hold the same number of elements, "
                       f"because it does not slice, reshape or broadcast: dst holds {nd} "
                       f"elements and src holds {ns}.")
    if func in ("dma_copy", "tensor_copy"):
        for k in ("dst", "src"):
            if buf.get(k) in ("sbuf", "psum") and shape.get(k) and shape[k][0] > pmax:
                out.append(f"A tile on the chip may have at most {pmax} rows, since its first "
                           f"dimension is the partition dimension: {k} has {shape[k][0]}.")
    if func == "nc_matmul" and shape.get("stationary") and shape.get("moving"):
        st, mv, dst = shape["stationary"], shape["moving"], shape.get("dst")
        if st[0] != mv[0]:
            out.append(f"The first dimension of stationary and of moving is the one nc_matmul "
                       f"contracts, so the two must be equal: stationary's is {st[0]} and "
                       f"moving's is {mv[0]}.")
        elif st[0] > pmax:
            out.append(f"One nc_matmul can contract at most {pmax}: the contraction (first) "
                       f"dimension here is {st[0]}.")
        if len(st) > 1 and st[1] > smax:
            out.append(f"The second dimension of stationary may be at most {smax}: it is "
                       f"{st[1]}.")
        if len(mv) > 1 and mv[1] > mmax:
            out.append(f"The second dimension of moving may be at most {mmax}: it is {mv[1]}.")
        if dst and len(st) == 2 and len(mv) == 2 and tuple(dst) != (st[1], mv[1]):
            out.append("The result of nc_matmul has stationary's second dimension as its first "
                       "and moving's second dimension as its second, and dst must be exactly "
                       f"that shape: dst is {dst}.")
    if func == "ndarray":
        dims = ops.get("shape", (None,) * 4)[3]
        try:
            dims = tuple(int(d) for d in dims)
        except Exception:
            dims = None
        # nl.ndarray(shape, dtype, buffer=nl.sbuf): leaving buffer out means a tile on the chip.
        where = _region_name(ops["buffer"][3]) if "buffer" in ops else "sbuf"
        if dims is not None and where in ("sbuf", "psum"):
            if len(dims) < 2:
                out.append(f"A tile on the chip needs at least 2 dimensions, the partition "
                           f"dimension first: this shape has {len(dims)}.")
            elif dims[0] > pmax:
                out.append(f"A tile on the chip may have at most {pmax} rows: this one would "
                           f"have {dims[0]}.")
    return out


def _allocation(source, name, before, inside):
    """(line, statement) of the last assignment to `name` above line `before`, in the function
    spanning `inside`. That is where a tile got its shape."""
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name \
                    and inside[0] <= node.lineno < before and (best is None
                                                               or node.lineno > best):
                best = node.lineno
    except SyntaxError:
        return None
    return (best, statement_at(source, best, limit=140)) if best else None


def _earlier_use(source, name, before, inside):
    """(line, statement) of the first instruction above line `before` that used tile `name` as an
    operand. Execution got past it, so the shape the tile has is the shape that line needed."""
    best = None
    try:
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") in OPERANDS
                    and node.func.attr != "ndarray" and inside[0] <= node.lineno < before):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                root = arg
                while isinstance(root, ast.Subscript):
                    root = root.value
                if isinstance(root, ast.Name) and root.id == name \
                        and (best is None or node.lineno < best):
                    best = node.lineno
    except SyntaxError:
        return None
    return (best, statement_at(source, best, limit=140)) if best else None


def _later_calls(source, name, after, inside):
    """Instructions below line `after` that use tile `name` as an operand, in order:
    [(line, function, {role: operand text})]. What a tile is used for is what its shape is for."""
    found = []
    try:
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") in OPERANDS
                    and node.func.attr != "ndarray" and after < node.lineno <= inside[1]):
                continue
            given = dict(zip(OPERANDS[node.func.attr], node.args))
            given.update({k.arg: k.value for k in node.keywords if k.arg})
            roles = {}
            for role, arg in given.items():
                root = arg
                while isinstance(root, ast.Subscript):
                    root = root.value
                if isinstance(root, ast.Name):
                    roles[role] = (root.id, " ".join((ast.get_source_segment(source, arg) or "").split()))
            if any(r[0] == name for r in roles.values()):
                found.append((node.lineno, node.func.attr, roles))
    except SyntaxError:
        return []
    return sorted(found, key=lambda f: f[0])


def _has_loop(source, inside):
    try:
        return any(isinstance(n, (ast.For, ast.While)) and inside[0] <= n.lineno <= inside[1]
                   for n in ast.walk(ast.parse(source)))
    except SyntaxError:
        return False


def _in_loop(source, lineno):
    try:
        return any(isinstance(n, (ast.For, ast.While)) and n.lineno < lineno <= (n.end_lineno or 0)
                   for n in ast.walk(ast.parse(source)))
    except SyntaxError:
        return False


def _loaded_askew(func, ops, source, lineno, inside, local):
    """Sentences for a matmul operand that was LOADED with the wrong layout, or [].

    Measured, level 3, two runs out of three on one seat: the model allocates the left operand's
    tile as (64, 128) and copies `lhsT`, which is (128, 64), into it. Same number of elements, so
    the copy is accepted. The matmul then says the contraction dimensions differ, 64 against 128,
    which is true and points at the matmul; the model resends the kernel unchanged. The line
    that is wrong is the allocation, and what is wrong with it only shows next to the tensor it
    was loaded from. A level whose only passing shape is square can never show this at all.
    """
    if func != "nc_matmul":
        return []
    shape = {k: v[1] for k, v in ops.items()}
    st, mv = shape.get("stationary"), shape.get("moving")
    if not (st and mv and st[0] != mv[0]):
        return []
    out = []
    for role in ("stationary", "moving"):
        name = ops[role][0].split("[", 1)[0].strip()
        if not name.isidentifier():
            continue
        loads = [c for c in _later_calls(source, name, inside[0] - 1, (inside[0], lineno - 1))
                 if c[1] in ("dma_copy", "tensor_copy") and c[2].get("dst", ("",))[0] == name
                 and "src" in c[2]]
        for line, _, roles in loads[-1:]:
            src = roles["src"][0]
            have, came = _shape(local.get(name)), _shape(local.get(src))
            if have and came and have != came and math.prod(have) == math.prod(came):
                out.append(f"`{name}` is {have}, but it was loaded on line {line} from `{src}`, "
                           f"which is {came}: the same number of elements in a different layout. "
                           f"A copy does not transpose or rearrange, so the tile holds `{src}`'s "
                           f"values cut into rows at the wrong places. Allocate `{name}` with "
                           f"the shape of `{src}`.")
    return out


def _origin(func, ops, limits, source, lineno, inside, params, pieces=False, error="",
            ahead=False, assigned=None):
    """Sentences naming the tile that has to change and the line that gave it its shape.

    Measured on level 3, two runs out of two: told the failing line, the shapes on it and the
    rule, the model sent back the same tile. The line that FAILS is the nc_matmul; the line that
    is WRONG is ten lines above it, where the result tile was allocated, and "change exactly what
    the checker names" points at the first. So follow the tile back to where it was made, and say
    which of its dimensions is off, in which direction, and which operand it has to match. That
    is which thing is wrong and which way -- not the number to type.
    """
    pmax, smax, mmax = limits
    shape = {k: v[1] for k, v in ops.items()}
    text = {k: v[0] for k, v in ops.items()}
    blame = []                                   # (role, what is wrong with it)
    direct = []                                  # sentences that need no tracing back
    replaces = False                             # does this say what the old hint tried to?

    def size(s):
        return math.prod(s)

    if func == "nc_matmul" and shape.get("stationary") and shape.get("moving") and shape.get("dst"):
        st, mv, dst = shape["stationary"], shape["moving"], shape["dst"]
        # An operand over the per-call limit is the thing to fix. While it is, the result tile
        # only looks wrong, and blaming it sends the model to resize the one tile that is right.
        over = pieces and len(st) == 2 and len(mv) == 2 and (st[1] > smax or mv[1] > mmax)
        if len(st) == 2 and len(mv) == 2 and tuple(dst) != (st[1], mv[1]) and not over:
            if len(dst) != 2:
                blame.append(("dst", f"it has {len(dst)} dimension(s) and needs 2"))
            else:
                parts = []
                for i, (have, want, src, word) in enumerate(
                        ((dst[0], st[1], text["stationary"], "first"),
                         (dst[1], mv[1], text["moving"], "second"))):
                    if have != want:
                        parts.append(f"its {word} dimension is {have}, too "
                                     f"{'small' if have < want else 'large'}: it must equal the "
                                     f"second dimension of `{src}`")
                blame.append(("dst", "; ".join(parts)))
    if pieces and func == "nc_matmul" and "free dimension" in error \
            and shape.get("stationary") and shape.get("moving") and shape.get("dst"):
        st, mv, dst = shape["stationary"], shape["moving"], shape["dst"]
        if len(st) == 2 and len(mv) == 2 and st[1] <= smax and mv[1] <= mmax \
                and tuple(dst) != (st[1], mv[1]):
            # Measured: with both operands cut to size, the simulator still said "moving free
            # dimension 1024 exceeds 512". The 1024 was the RESULT tile's. The model, told an
            # operand was too wide when neither was, sent the same code back three times.
            direct.append("Both operands are within the limits here, so the simulator's message "
                          "is about the result tile, which it measures against the same limits.")
    if pieces and func == "nc_matmul" and "contraction" not in error:
        # Measured on level 4, every run: with one dimension tiled the kernel reached 2 of 4
        # shapes and stopped here, resending the same code. The simulator's message for this has
        # no advice written for it at all. The limit is per call, so the only way through is
        # more pieces: say which dimension, the limit, and that it is cut and looped like the
        # one already cut.
        # Not while the simulator is complaining about the contraction. Measured: said alongside
        # a contraction error, with three limits in one message, it made the kernel worse (0.62
        # where the shorter message reached 0.75). One named change at a time.
        # The test is "not a contraction error" rather than "a free-dimension error" because
        # the simulator has no check on the moving operand's width at all: an operand of 1024
        # with a result tile in range comes back as a numpy reshape error. The self-test caught
        # that on the real simulator before any run used the narrower test.
        for role, lim in (("stationary", smax), ("moving", mmax)):
            s = shape.get(role)
            if s and len(s) == 2 and s[1] > lim:
                direct.append(f"`{text[role]}` is {s}: its second dimension is {s[1]}, too "
                              f"large, since one nc_matmul takes at most {lim} there. Cut that "
                              f"dimension too, at most {lim} at a time, and loop over the pieces.")
    if func in ("dma_copy", "tensor_copy") and shape.get("dst") and shape.get("src"):
        big, small = (("dst", "src") if size(shape["dst"]) > size(shape["src"])
                      else ("src", "dst"))
        fixed_output = ops[big][2] == "shared_hbm" and big == "dst" and _has_loop(source, inside)
        if pieces and size(shape["dst"]) != size(shape["src"]) and fixed_output \
                and not _in_loop(source, lineno):
            # The output's shape is set by the task. A piece copied into all of it, after the
            # loops, means the copy is in the wrong place, not that the output is the wrong size.
            name = text[big].split("[", 1)[0].strip()
            direct.append(f"`{name}` is the kernel's output, so its shape is fixed and it stays. "
                          f"It is larger than what is copied into it here: "
                          f"{size(shape[big])} elements against {size(shape[small])}. This copy "
                          f"moves one piece, so it belongs inside the loop that makes the "
                          f"pieces, writing each one into its own slice of `{name}`.")
        elif pieces and size(shape["dst"]) != size(shape["src"]) and "hbm" in (ops[big][2] or "") \
                and _in_loop(source, lineno):
            # A tensor in HBM is an input or the output: its shape is given. When it is the
            # larger side AND the copy sits in a loop, the kernel is moving one piece of it, and
            # resizing the tile to fit the whole tensor is exactly the mistake tiling exists to
            # avoid. Outside a loop the same sizes mean something else -- a tile sized for the
            # wrong operand -- and "read a piece of it" would be wrong advice, so this stays out.
            name = text[big].split("[", 1)[0].strip()
            direct.append(f"`{name}` is in HBM and is larger than the tile on the other side: "
                          f"{size(shape[big])} elements against {size(shape[small])}. That is "
                          f"the tensor's own shape and it stays. "
                          f"{'Write into' if big == 'dst' else 'Read'} only the piece of `{name}` "
                          f"this tile is for: a slice of it with the same shape as "
                          f"`{text[small]}`.")
        elif size(shape["dst"]) != size(shape["src"]):
            # The tile the kernel allocated is the one it can reshape; an input is what it is.
            sides = [(r, o) for r, o in (("dst", "src"), ("src", "dst"))
                     if text[r].split("[", 1)[0].strip() not in params] or [("dst", "src")]
            role, other = sides[0]
            way = "small" if size(shape[role]) < size(shape[other]) else "large"
            blame.append((role, f"it holds {size(shape[role])} elements, too {way} for "
                                f"the {size(shape[other])} on the other side"))
        for role in ("dst", "src"):
            if shape[role][0] > pmax and ops[role][2] in ("sbuf", "psum"):
                blame.append((role, f"it has {shape[role][0]} rows, too many: at most {pmax}"))
    out, done = [], set()
    for role, what in blame:
        root = text[role].split("[", 1)[0].strip()
        if not root.isidentifier() or root in done:
            continue
        done.add(root)
        made = _allocation(source, root, lineno, inside) if root not in params else None
        # Only a use AFTER the tile last got its shape says anything about that shape: a name
        # allocated, used, then allocated again is a new tile.
        since = (made[0] + 1, inside[1]) if made else inside
        used = _earlier_use(source, root, lineno, since) if func != "nc_matmul" else None
        # Later WRITES only. A tile that is loaded here and read by a matmul further down is one
        # tile doing one job. Measured on level 4: counting that read as "other data" told the
        # model to split a tile that was fine, and dropped the advice it needed (0.62).
        later = ([c for c in _later_calls(source, root, lineno, inside)
                  if c[2].get("dst", ("",))[0] == root]
                 if ahead and func in ("dma_copy", "tensor_copy") and not used else [])
        if "too many: at most" in what:
            # No number on the allocation line fixes a tile that is too tall, and a second tile
            # of the same height is no better. Say so before either of the remedies below.
            where = f" It got its shape on line {made[0]}, `{made[1]}`." if made else ""
            out.append(f"`{root}` is too tall: {what}.{where} No single tile can have that "
                       f"many rows, so a different number there will not do: the data has to "
                       f"be handled in pieces of at most that many rows, in a loop.")
        elif used:
            # Measured on level 3: told to resize a tile that an earlier copy had filled, the
            # model resized it, broke that copy, was told to resize it back, and went round.
            # A tile an earlier instruction already used has the shape THAT instruction needed.
            # It is being used for two things, and the second needs a tile of its own.
            out.append(f"`{root}` is the wrong size here: {what}. But do not resize it: line "
                       f"{used[0]}, `{used[1]}`, already ran with the shape it has, so it is "
                       f"being used for two different things. Leave `{root}` as it is and "
                       f"allocate a separate tile for this line, sized to match the other side.")
        elif later:
            # Measured, the one level 3 run in five that did not solve: a single tile meant to
            # hold the left operand, the right operand and the result. Told to resize it, the
            # model resized it for one of the three and failed on the next, four times over.
            # The old hint's example, a (128, 512) tile, was the very tile it then allocated.
            lines = ", ".join(str(c[0]) for c in later[:4])
            out.append(f"`{root}` is the wrong size here: {what}. It is also written to on line"
                       f"{'s' if len(later) > 1 else ''} {lines} with other data, and one tile "
                       f"cannot hold different things. Allocate a separate tile for each thing "
                       f"it holds, each shaped like what is copied into it.")
            replaces = True
        elif made:
            # The old hint for a copy mismatch explains how to take a piece of a BIGGER tensor.
            # That is the answer when the source is the larger side, and it stays. When the
            # tile is the larger side there is nothing to take a piece of, and the hint's
            # example -- a (128, 512) tile -- is the tile the model then allocates.
            if ahead and func in ("dma_copy", "tensor_copy") and role == "dst" \
                    and size(shape["dst"]) > size(shape["src"]):
                replaces = True
            piece = (" Each pass of the loop makes one piece of the result, so that tile holds "
                     "one piece: fill it, copy it to its place in the output, then reuse it."
                     if pieces and func == "nc_matmul" and role == "dst"
                     and _in_loop(source, lineno) else "")
            out.append(f"The tile to change is `{root}`: {what}. It got its shape on line "
                       f"{made[0]}, `{made[1]}`, so that is the line to change, not this one."
                       f"{piece}")
        else:
            out.append(f"The tile to change is `{root}`: {what}.")
    if ahead and func == "ndarray" and assigned:
        # A tile that could not be allocated has no shape to report, but the kernel already says
        # what it is for. The old hint for this error -- "give a vector the shape (1, N)" --
        # answered a question nobody asked and produced the (1, 64) result tile on level 3.
        for line, called, roles in _later_calls(source, assigned, lineno, inside)[:1]:
            mine = next((r for r, v in roles.items() if v[0] == assigned), None)
            if called == "nc_matmul" and mine == "dst" and "stationary" in roles and "moving" in roles:
                direct.append(f"`{assigned}` is used on line {line} as the result of nc_matmul. "
                              f"A result has two dimensions: the second dimension of "
                              f"`{roles['stationary'][1]}`, then the second dimension of "
                              f"`{roles['moving'][1]}`.")
                replaces = True
            elif called in ("dma_copy", "tensor_copy") and mine in ("dst", "src"):
                other = roles.get("src" if mine == "dst" else "dst")
                if other:
                    direct.append(f"`{assigned}` is used on line {line} in a copy with "
                                  f"`{other[1]}`, so it needs the same shape as `{other[1]}`.")
                    replaces = True
    return direct + out, replaces


def _missing_name(exc, stmt, source, local, watch, params):
    """A sentence for a NameError, or ''.

    Measured: the only level 3 runs still failing with `ahead` on. The model splits one tile into
    three, renames them, and leaves the old name on one line. Told "name 'sbuf_tile' is not
    defined", it sends the same code back until the loop gives up. The harness already does this
    for an invented NKI function -- it lists the real ones. This does it for a variable: say
    which tensors do exist at that line, and what the missing one is needed for.
    """
    m = re.search(r"name '(\w+)' is not defined", str(exc))
    if not (isinstance(exc, NameError) and m):
        return ""
    missing = m.group(1)
    have = []
    for name, val in local.items():
        shape = _shape(val)
        if shape is not None and not isinstance(val, (types.ModuleType, type)):
            have.append(f"`{name}` {shape}" + _where_text(_buffer(val, name, watch, params),
                                                          name, params))
    out = f"`{missing}` does not exist at this line: nothing above it assigns that name."
    if have:
        out += " The tensors that do exist here are " + ", ".join(have[:8]) + "."
    for call in (c for part in _own_expressions(stmt) for c in ast.walk(part)
                 if isinstance(c, ast.Call) and getattr(c.func, "attr", "") in OPERANDS):
        given = dict(zip(OPERANDS[call.func.attr], call.args))
        given.update({k.arg: k.value for k in call.keywords if k.arg})
        text = {r: " ".join((ast.get_source_segment(source, n) or "").split())
                for r, n in given.items()}
        role = next((r for r, n in given.items() if isinstance(n, ast.Name) and n.id == missing),
                    None)
        func = call.func.attr
        if func in ("dma_copy", "tensor_copy") and role in ("dst", "src"):
            other = text.get("src" if role == "dst" else "dst")
            if other:
                out += (f" It is the {role} of this {func}, so allocate it above this line with "
                        f"the same shape as `{other}`.")
        elif func == "nc_matmul" and role == "dst" and "stationary" in text and "moving" in text:
            out += (f" It is the result of this nc_matmul, so allocate it above this line in "
                    f"psum, with the second dimension of `{text['stationary']}` then the second "
                    f"dimension of `{text['moving']}`.")
        break
    return out


def state(exc, path, source, watch=None, limit=700):
    """' At that line: ...' wrapped in MARK, or '' when there is nothing certain to say."""
    return state_and_rules(exc, path, source, watch, limit)[0]


def state_and_rules(exc, path, source, watch=None, limit=700, origin=False, pieces=False,
                    ahead=False, names=False, layout=False):
    """(text, number of rules stated, whether it says what the old hint for this error tried
    to). Shapes alone say what was there; a rule says what was wrong with it. Only the second
    can stand in for advice."""
    try:
        return _state(exc, path, source, watch,
                      limit + (350 if origin else 0) + (250 if pieces else 0), origin, pieces,
                      ahead, names, layout)
    except Exception as e:
        ERRORS.append(f"state: {type(e).__name__}: {e}")
        return "", 0, False


def _state(exc, path, source, watch, limit, origin=False, pieces=False, ahead=False,
           undefined=False, askew=False):
    hit = _hit(exc, path, source)
    stmt = _statement_node(source, hit[0]) if hit else None
    if stmt is None:
        return "", 0, False
    frame = hit[1]
    replaces = False
    assigned = (stmt.targets[0].id if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) else None)
    local = dict(frame.f_locals)
    params = set()
    try:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == frame.f_code.co_name:
                params = {a.arg for a in node.args.args}
    except SyntaxError:
        pass

    # Every variable the statement mentions, in the order it mentions them.
    facts, seen = [], set()
    names = sorted((n for part in _own_expressions(stmt) for n in ast.walk(part)
                    if isinstance(n, ast.Name)), key=lambda n: (n.lineno, n.col_offset))
    for n in names:
        if n.id in seen or n.id not in local:
            continue
        seen.add(n.id)
        val = local[n.id]
        if isinstance(val, (types.ModuleType, types.FunctionType, type)) or isinstance(val, bool):
            continue
        shape = _shape(val)
        if shape is not None:
            facts.append(f"`{n.id}` is {shape}"
                         + _where_text(_buffer(val, n.id, watch, params), n.id, params))
        elif isinstance(val, (int, float)):
            facts.append(f"`{n.id}` = {val}")
        elif isinstance(val, tuple) and val and all(isinstance(v, int) for v in val):
            facts.append(f"`{n.id}` = {val}")

    # The operands of the instruction on that line, including ones that are slices.
    rules = []
    for call in sorted((c for part in _own_expressions(stmt) for c in ast.walk(part)
                        if isinstance(c, ast.Call) and getattr(c.func, "attr", "") in OPERANDS),
                       key=lambda c: (c.lineno, c.col_offset)):
        func = call.func.attr
        given = dict(zip(OPERANDS[func], call.args))
        given.update({k.arg: k.value for k in call.keywords if k.arg})
        ops = {}
        for role, node in given.items():
            if role not in OPERANDS[func]:
                continue
            text = " ".join((ast.get_source_segment(source, node) or "").split())
            val = _evaluate(node, frame)
            shape = _shape(val) if val is not None else None
            root = node
            while isinstance(root, ast.Subscript):
                root = root.value
            rname = root.id if isinstance(root, ast.Name) else ""
            # A slice lives wherever the tensor it was cut from lives.
            base = local.get(rname) if isinstance(node, ast.Subscript) else val
            buf = _buffer(base, rname, watch, params) if base is not None else None
            ops[role] = (text, shape, buf, val)
            if shape is not None and not isinstance(node, ast.Name) and func != "ndarray":
                facts.append(f"{role} `{text[:80]}` is {shape}" + _where_text(buf, rname, params))
        try:
            import nkibench
            limits = (nkibench.PMAX, nkibench.GEMM_STATIONARY_FMAX, nkibench.GEMM_MOVING_FMAX)
        except Exception:
            limits = (128, 128, 512)
        rules = _rules(func, ops, limits)
        if origin:
            inside = _function_spans(source).get(frame.f_code.co_name, (1, hit[0]))
            more, replaces = _origin(func, ops, limits, source, hit[0], inside, params, pieces,
                                     str(exc), ahead, assigned)
            rules += more
            if askew:
                skew = _loaded_askew(func, ops, source, hit[0], inside, local)
                if skew:
                    # While an operand has the wrong layout, the result tile only LOOKS wrong:
                    # it is being compared with an operand that is itself the mistake. Measured
                    # on that same kernel: told the result tile was off, the model resized a
                    # tile that was right. One named change, and it is the operand.
                    rules = [r for r in rules if not r.startswith(("The result of nc_matmul has",
                                                                   "The tile to change is"))]
                    rules += skew
        break
    # `undefined`, not `names`: this function already has a local called names, the Name nodes of
    # the statement, and a parameter of that name was silently always true. The replay against
    # the previous commit caught it before it reached a seat.
    if undefined:
        lost = _missing_name(exc, stmt, source, local, watch, params)
        if lost:
            rules.append(lost)
    if not facts and not rules:
        return "", 0, False
    text = (" At that line: " + "; ".join(facts) + "." if facts else "") \
        + "".join(" " + r for r in rules)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return f"{MARK}{text}{MARK}", len(rules), replaces


# ---------------------------------------------------------------- selftest
#
# A located line is only worth sending if it is the RIGHT line. So plant a bug whose failing
# statement is known in a kernel that is otherwise correct, and check that statement is the one
# reported. `expect` is a fragment of the statement that must be named, and `facts` are things
# that were true on that line and must be reported as they were.

PLANTS = [
    dict(name="tile too small for the copy", level=3, case=0,
         old="rhs_tile = nl.ndarray(rhs.shape,",
         new="rhs_tile = nl.ndarray((rhs.shape[0], rhs.shape[1] // 2),",
         expect="nisa.dma_copy(dst=rhs_tile, src=rhs)",
         facts=["`rhs_tile` is (128, 256)", "`rhs` is (128, 512)",
                "dst holds 32768 elements and src holds 65536"],
         origin=["The tile to change is `rhs_tile`", "too small",
                 "`rhs_tile = nl.ndarray((rhs.shape[0], rhs.shape[1] // 2),"]),
    dict(name="1-D tile", level=3, case=0,
         old="lhs_tile = nl.ndarray(lhsT.shape,",
         new="lhs_tile = nl.ndarray((lhsT.shape[0] * lhsT.shape[1],),",
         expect="lhs_tile = nl.ndarray(",
         facts=["`lhsT` is (128, 64)", "needs at least 2"]),
    dict(name="whole tensor in one tile", level=4, case=1,
         old="  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n",
         new="  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n"
             "  whole = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)\n"
             "  nisa.dma_copy(dst=whole, src=lhsT)\n",
         expect="nisa.dma_copy(dst=whole, src=lhsT)",
         facts=["`whole` is (256, 256)", "`lhsT` is (256, 256)"],
         origin=["`whole` is too tall", "too many", "`whole = nl.ndarray(lhsT.shape,",
                 "in pieces"]),
    dict(name="tile too small, call split over lines, inside three loops", level=4, case=1,
         old="rhs_tile = nl.ndarray((TILE_K, TILE_N),",
         new="rhs_tile = nl.ndarray((TILE_K, TILE_N // 2),",
         expect="nisa.dma_copy(dst=rhs_tile,",
         facts=["`rhs_tile` is (128, 256)", "`k` = 0", "`TILE_K` = 128",
                "dst holds 32768 elements and src holds 65536"],
         origin=["The tile to change is `rhs_tile`", "too small",
                 "`rhs_tile = nl.ndarray((TILE_K, TILE_N // 2),"]),
    dict(name="one tile used for two things", level=3, case=0,
         old="nisa.tensor_copy(dst=result_sbuf, src=result_psum)",
         new="nisa.tensor_copy(dst=rhs_tile, src=result_psum)",
         expect="nisa.tensor_copy(dst=rhs_tile, src=result_psum)",
         facts=["`rhs_tile` is (128, 512)", "`result_psum` is (64, 512)"],
         origin=["do not resize it", "`nisa.dma_copy(dst=rhs_tile, src=rhs)`",
                 "allocate a separate tile"]),
    dict(name="whole input copied into the tile for one piece", level=4, case=1,
         old="src=lhsT[k * TILE_K:(k + 1) * TILE_K,\n"
             "                               m * TILE_M:(m + 1) * TILE_M])",
         new="src=lhsT)",
         expect="nisa.dma_copy(dst=lhsT_tile,",
         pieces=["`lhsT` is in HBM and is larger", "Read only the piece of `lhsT`",
                 "same shape as `lhsT_tile`"]),
    dict(name="one piece written over the whole output", level=4, case=1,
         old="nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,\n"
             "                               n * TILE_N:(n + 1) * TILE_N],\n"
             "                    src=res_sb)",
         new="nisa.dma_copy(dst=result, src=res_sb)",
         expect="nisa.dma_copy(dst=result, src=res_sb)",
         pieces=["`result` is in HBM and is larger", "Write into only the piece of `result`"]),
    dict(name="operand too wide for one matmul", level=4, case=1,
         old="rhs_tile = nl.ndarray((TILE_K, TILE_N),",
         new="rhs_tile = nl.ndarray((TILE_K, N),",
         more=[("src=rhs[k * TILE_K:(k + 1) * TILE_K,\n"
                "                              n * TILE_N:(n + 1) * TILE_N])",
                "src=rhs[k * TILE_K:(k + 1) * TILE_K, :])")],
         expect="nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)",
         pieces=["`rhs_tile` is (128, 1024)", "too large", "at most 512", "loop over the pieces"]),
    dict(name="result tile with one dimension", level=3, case=0,
         old="result_psum = nl.ndarray(result.shape,",
         new="result_psum = nl.ndarray((M,),",
         expect="result_psum = nl.ndarray((M,),",
         ahead=["`result_psum` is used on line", "as the result of nc_matmul",
                "second dimension of `lhs_tile`", "second dimension of `rhs_tile`"]),
    dict(name="one tile for both operands", level=3, case=0,
         old="lhs_tile = nl.ndarray(lhsT.shape,",
         new="lhs_tile = nl.ndarray(rhs.shape,",
         more=[("nisa.dma_copy(dst=rhs_tile, src=rhs)", "nisa.dma_copy(dst=lhs_tile, src=rhs)")],
         expect="nisa.dma_copy(dst=lhs_tile, src=lhsT)",
         ahead=["`lhs_tile` is the wrong size here", "It is also written to on line 57 with",
                "Allocate a separate tile for each thing"]),
    dict(name="a tile used under a name nothing assigns", level=3, case=0,
         old="nisa.tensor_copy(dst=result_sbuf, src=result_psum)",
         new="nisa.tensor_copy(dst=result_sb, src=result_psum)",
         expect="nisa.tensor_copy(dst=result_sb, src=result_psum)",
         names=["`result_sb` does not exist at this line", "`result_psum` (64, 512) in psum",
                "It is the dst of this tensor_copy", "same shape as `result_psum`"]),
    dict(name="operand tile allocated transposed, then loaded", level=3, case=0,
         old="lhs_tile = nl.ndarray(lhsT.shape,",
         new="lhs_tile = nl.ndarray((lhsT.shape[1], lhsT.shape[0]),",
         expect="nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)",
         layout=["`lhs_tile` is (64, 128), but it was loaded on line", "from `lhsT`, which is (128, 64)",
                 "does not transpose", "Allocate `lhs_tile` with the shape of `lhsT`"]),
    # Not pass/fail. This asks the simulator a question: what does a result tile of the wrong
    # shape look like from outside? If the answer is a reshape error, then a kernel with no
    # reshape in it can produce one, and the line is the only thing that says where.
    dict(name="result tile of the wrong shape", level=3, case=0, question=True,
         old="result_psum = nl.ndarray(result.shape,",
         new="result_psum = nl.ndarray((1, M),",
         expect="nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)",
         facts=["`result_psum` is (1, 64)", "`lhs_tile` is (128, 64)", "`rhs_tile` is (128, 512)",
                "dst must be exactly that shape"],
         origin=["The tile to change is `result_psum`", "its first dimension is 1, too small",
                 "second dimension of `rhs_tile`", "`result_psum = nl.ndarray((1, M),"]),
]


def _run(source, level, case, tag, watch=True):
    """Simulate `source` on one shape of `level`. Returns (path, exception or None, watch, out)."""
    import nkibench
    spec = nkibench.LEVELS[level]
    path = f"/tmp/_diagnose_{tag}.py"
    with open(path, "w") as f:
        f.write(source)
    args, _ = nkibench.make_inputs(spec["shapes"][case], level)
    w = watching(path) if watch else None
    try:
        if w is None:
            out, _ = nkibench.simulate_and_count(nkibench.load_kernel(path, spec["entry"]), args)
        else:
            with w:
                out, _ = nkibench.simulate_and_count(nkibench.load_kernel(path, spec["entry"]),
                                                     args)
    except nkibench.NkiMissing:
        raise
    except Exception as e:
        return path, e, w, None
    return path, None, w, out


def selftest():
    import nkibench
    here = os.path.dirname(os.path.abspath(__file__))
    print("Planting known bugs in the reference kernels and checking the located line.\n")

    # statement_at needs no simulator, so prove it everywhere.
    src = "a = 1\nfor i in range(3):\n    f(a,\n      i)\n"
    ok = (statement_at(src, 3) == "f(a, i)" and statement_at(src, 4) == "f(a, i)"
          and statement_at(src, 2) == "for i in range(3):" and statement_at(src, 1) == "a = 1")
    print(f"  statement_at: split call whole, loop as its header -> {'ok' if ok else 'FAIL'}")
    marked = f"On K=1: {MARK}line 3, `x`, {MARK}raised E: m"
    ok2 = shown(marked) == "On K=1: line 3, `x`, raised E: m" and key(marked) == "On K=1: raised E: m"
    print(f"  shown / key: located text shown, then removed     -> {'ok' if ok2 else 'FAIL'}")
    rc = 0 if ok and ok2 else 1
    state_rc = origin_rc = pieces_rc = ahead_rc = names_rc = layout_rc = 0

    try:
        import nki
    except ImportError:
        print("\n  NEEDS THE NEURON SDK: the plants below cannot run here. Run this in the seat pod.")
        return rc
    print(f"\n  simulator: nki {getattr(nki, '__version__', '?')} from {os.path.dirname(nki.__file__)}\n")

    # A tile taller than the chip has, allocated by a symbolic size and never passed whole to an
    # instruction, so nothing raises. Only the allocation watcher can see it.
    ref4 = open(os.path.join(here, "reference_level4.py")).read()
    mark = "  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n"
    hidden = ref4.replace(mark, mark + "  spare = nl.ndarray((K, TILE_M), dtype=lhsT.dtype, "
                                      "buffer=nl.sbuf)\n")
    path, err, w, _ = _run(hidden, 4, 1, "tall")
    said = tall(w, hidden)
    ok = ref4.count(mark) == 1 and err is None and "`spare = nl.ndarray((K, TILE_M)" in said \
        and "256 rows" in said
    path, err0, w0, _ = _run(ref4, 4, 1, "tall0")
    ok0 = err0 is None and tall(w0, ref4) == ""
    print(f"  tile taller than the chip, never used whole: the kernel still runs -> "
          f"{'yes' if err is None else 'NO, it raised ' + type(err).__name__}")
    print(f"      caught, with its line and real height -> {'ok' if ok else 'FAIL'}")
    print(f"      {said[:150]}")
    print(f"      and the untouched reference is not accused -> {'ok' if ok0 else 'FAIL'}")
    ref3 = open(os.path.join(here, "reference_level3.py")).read()
    psum_out = ref3.replace("  return result\n", "  return result_psum\n")
    said3 = output_on_chip(psum_out, "nki_matmul_basic_")
    ok3 = psum_out != ref3 and "returns `result_psum`" in said3 and "psum" in said3 \
        and all(output_on_chip(open(os.path.join(here, f"reference_level{n}.py")).read(),
                               nkibench.LEVELS[n]["entry"]) == "" for n in (1, 2, 3, 4))
    print(f"  an output returned from psum is caught, and no reference is -> {'ok' if ok3 else 'FAIL'}")
    tall_rc = 0 if ok and ok0 and ok3 else 1
    print()

    # The worked example that --features example shows the model. It has to be right.
    import numpy as _np
    import agent as _agent
    ex_path = "/tmp/_diagnose_example.py"
    open(ex_path, "w").write(_agent.TILING_EXAMPLE)
    a = _np.arange(256 * 1024, dtype=_np.float32).reshape(256, 1024)
    ex_ok, ex_note = False, ""
    try:
        with watching(ex_path) as ew:
            got, _ = nkibench.simulate_and_count(nkibench.load_kernel(ex_path, "copy_in_pieces"), [a])
        ex_ok = _np.array_equal(_np.asarray(got), a) and not ew.tall
        ex_note = "" if ex_ok else f"wrong output or a tall tile: {ew.tall[:1]}"
    except Exception as e:
        ex_note = f"raised {type(e).__name__}: {str(e)[:120]}"
    print(f"  the worked tiling example: runs, copies exactly, no tile over 128 rows -> "
          f"{'ok' if ex_ok else 'FAIL  ' + ex_note}")
    example_rc = 0 if ex_ok else 1
    print()

    # The control. If the untouched references do not run, a plant's failure proves nothing --
    # and watching a simulation must not change what it computes, or the watcher is the bug.
    import numpy as np
    refs = {}
    for level in sorted({p["level"] for p in PLANTS}):
        refs[level] = open(os.path.join(here, f"reference_level{level}.py")).read()
        for case in sorted({p["case"] for p in PLANTS if p["level"] == level}):
            _, err, _, plain = _run(refs[level], level, case, f"ref{level}", watch=False)
            _, err2, w, watched = _run(refs[level], level, case, f"ref{level}w", watch=True)
            lbl = nkibench.label(nkibench.LEVELS[level]["shapes"][case], level)
            print(f"  control: reference level {level} on {lbl} -> "
                  f"{'runs' if err is None else f'FAIL, raised {type(err).__name__}: {err}'}")
            same = err is None and err2 is None and np.array_equal(np.asarray(plain),
                                                                   np.asarray(watched))
            print(f"           and gives the same output while watched -> "
                  f"{'ok' if same else 'FAIL'}  ({len(w.buffers)} tile allocations seen)")
            rc |= 0 if err is None and same else 1
    print()

    for i, p in enumerate(PLANTS):
        ref = refs[p["level"]]
        if ref.count(p["old"]) != 1:
            print(f"  {p['name']}: FAIL, the reference no longer contains exactly one "
                  f"`{p['old'].strip()}` -- the plant is stale")
            rc |= 1
            continue
        source = ref.replace(p["old"], p["new"])
        for old, new in p.get("more", []):
            if source.count(old) != 1:
                print(f"  {p['name']}: FAIL, a second edit of this plant is stale")
                rc |= 1
            source = source.replace(old, new)
        path, err, w, _ = _run(source, p["level"], p["case"], f"plant{i}")
        if err is None:
            verdict = "the simulator accepted it" if p.get("question") else "FAIL, nothing was raised"
            print(f"  {p['name']}: {verdict}")
            rc |= 0 if p.get("question") else 1
            continue
        n = locate(err, path, source)
        stmt = statement_at(source, n) if n else ""
        right = bool(n) and p["expect"] in stmt
        if p.get("question"):
            print(f"  QUESTION, {p['name']}:")
        else:
            print(f"  {p['name']}: {'ok' if right else 'FAIL'}")
            rc |= 0 if right else 1
        print(f"      raised   {type(err).__name__}: {str(err)[:150]}")
        print(f"      located  {f'line {n}, `{stmt}`' if n else 'NO LINE FOUND'}")
        if not right and not p.get("question"):
            print(f"      expected a statement containing `{p['expect']}`")
        said = shown(state(err, path, source, w))
        missing = [f for f in p.get("facts", []) if f not in said]
        print(f"      state   {said.strip() or 'NOTHING REPORTED'}")
        if p.get("question"):
            print(f"      (facts not reported: {missing or 'none'})")
        else:
            print(f"      state reports every planted fact -> {'ok' if not missing else 'FAIL'}"
                  + (f"  missing: {missing}" if missing else ""))
            state_rc |= 0 if not missing else 1
        traced = shown(state_and_rules(err, path, source, w, origin=True)[0])
        lost = [f for f in p.get("origin", []) if f not in traced]
        if p.get("origin"):
            tail = traced.split("elements.", 1)[-1].strip() if "elements." in traced else traced
            print(f"      origin  {tail[-330:] if tail else 'NOTHING TRACED'}")
            print(f"      origin names the planted tile and line -> {'ok' if not lost else 'FAIL'}"
                  + (f"  missing: {lost}" if lost else ""))
            origin_rc |= 0 if not lost else 1
        if p.get("pieces"):
            cut = shown(state_and_rules(err, path, source, w, origin=True, pieces=True)[0])
            gone = [f for f in p["pieces"] if f not in cut]
            print(f"      pieces  {cut.strip()[-300:] if cut else 'NOTHING SAID'}")
            print(f"      pieces says to cut, not to resize -> {'ok' if not gone else 'FAIL'}"
                  + (f"  missing: {gone}" if gone else ""))
            pieces_rc |= 0 if not gone else 1
        if p.get("layout"):
            skew = shown(state_and_rules(err, path, source, w, origin=True, layout=True)[0])
            gone = [f for f in p["layout"] if f not in skew]
            print(f"      layout  {skew.strip()[-330:] if skew else 'NOTHING SAID'}")
            print(f"      layout traces the operand to the tensor it was loaded from -> "
                  f"{'ok' if not gone else 'FAIL'}" + (f"  missing: {gone}" if gone else ""))
            layout_rc |= 0 if not gone else 1
        if p.get("names"):
            lost = shown(state_and_rules(err, path, source, w, origin=True, names=True)[0])
            gone = [f for f in p["names"] if f not in lost]
            print(f"      names   {lost.strip()[-330:] if lost else 'NOTHING SAID'}")
            print(f"      names says what exists and what the missing one is for -> "
                  f"{'ok' if not gone else 'FAIL'}" + (f"  missing: {gone}" if gone else ""))
            names_rc |= 0 if not gone else 1
        if p.get("ahead"):
            look = shown(state_and_rules(err, path, source, w, origin=True, ahead=True)[0])
            gone = [f for f in p["ahead"] if f not in look]
            print(f"      ahead   {look.strip()[-300:] if look else 'NOTHING SAID'}")
            print(f"      ahead explains the tile by its later use -> {'ok' if not gone else 'FAIL'}"
                  + (f"  missing: {gone}" if gone else ""))
            ahead_rc |= 0 if not gone else 1

    print()
    print("Located lines: " + ("all are the planted ones." if rc == 0 else
                               "SOMETHING FAILED. Do not use --features locate until this passes."))
    print("State on the line: " + ("every planted fact was reported." if state_rc == 0 else
                                   "SOMETHING FAILED. Do not use --features state until this passes."))
    print("Origin of the tile: " + ("every planted line was traced." if origin_rc == 0 else
                                    "SOMETHING FAILED. Do not use --features origin until this passes."))
    print("In pieces: " + ("every planted case was told to cut." if pieces_rc == 0 else
                           "SOMETHING FAILED. Do not use --features pieces until this passes."))
    print("Too tall for the chip: " + ("the planted tile was caught and the reference was not."
                                       if tall_rc == 0 else
                                       "SOMETHING FAILED. Do not use --features strict until this passes."))
    print("Worked example: " + ("it runs, it is correct, and no tile in it is too tall."
                                if example_rc == 0 else
                                "SOMETHING FAILED. Do not use --features example until this passes."))
    print("Loaded askew: " + ("the planted operand was traced to its load." if layout_rc == 0 else
                              "SOMETHING FAILED. Do not use --features layout until this passes."))
    print("Missing names: " + ("the planted name was explained." if names_rc == 0 else
                               "SOMETHING FAILED. Do not use --features names until this passes."))
    print("Looking ahead: " + ("every planted case was explained by its later use."
                               if ahead_rc == 0 else
                               "SOMETHING FAILED. Do not use --features ahead until this passes."))
    return rc | (state_rc << 1) | (origin_rc << 2) | (pieces_rc << 3) | (ahead_rc << 4) \
        | (tall_rc << 5) | (names_rc << 6) | (example_rc << 7) | (layout_rc << 8)


def replay(log_path, level=None, features=("locate", "state")):
    """Re-grade every distinct failing kernel in an attempt log with the features on.

    Two uses. Before a measured run: the features meet kernels a model really wrote, which are
    stranger than any plant, and this shows that they cope -- for nothing, since no model is
    called. After one: the "was told / would add" pairs are the evidence for what changed.
    """
    import json
    import agent
    agent.CANDIDATE_PATH = "/tmp/_replay_level{level}.py"   # never the path a live run is using
    groups = {}
    with open(log_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (r.get("code") or "").strip() or r["reward"] >= 1.0 - 1e-9:
                continue
            if level is not None and r["level"] != level:
                continue
            groups.setdefault((r["level"], "".join(r["code"].split())), [r, 0])[1] += 1
    n = dict(kernels=0, crashed=0, located=0, state=0, differs=0)
    for (lv, _), (r, count) in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[1][1])):
        n["kernels"] += 1
        agent.FEATURES.clear()
        agent.FEATURES.update(features)
        try:
            _, _, marked = agent._grade(r["code"], lv)
        finally:
            agent.FEATURES.clear()
        added = [seg.strip() for seg in re.findall(f"{MARK}(.*?){MARK}", marked, re.S)]
        crashed = "raised " in key(marked)
        n["crashed"] += crashed
        n["located"] += any(seg.startswith("line ") for seg in added)
        n["state"] += any("At that line" in seg for seg in added)
        print(f"\nlevel {lv}   {count} attempt(s) were this kernel   reward {r['reward']:.2f}")
        print("  was told : " + " ".join((r.get("feedback") or "").split())[:300])
        if not r.get("features") and key(marked) != r.get("feedback"):
            n["differs"] += 1
            print("  NOTE     : grading it again gave a different message: "
                  + " ".join(key(marked).split())[:200])
        if added:
            for seg in added:
                print("  would add: " + seg)
        else:
            print("  would add: nothing" + ("" if crashed else
                                            " (it did not crash inside the kernel)"))
    print(f"\n{n['kernels']} distinct failing kernels; {n['crashed']} crashed in the simulator; "
          f"a line was named for {n['located']} of those and the state for {n['state']}.")
    if n["differs"]:
        print(f"{n['differs']} graded differently from the log. Grading should be repeatable: "
              f"look at those before trusting either.")
    print(f"errors inside the diagnosis itself: {len(ERRORS)}" + ("" if not ERRORS else
          "\n  " + "\n  ".join(sorted(set(ERRORS)))))
    return 1 if ERRORS else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay", metavar="LOG.jsonl")
    ap.add_argument("--level", type=int)
    ap.add_argument("--features", default="locate,state", help="for --replay")
    ap.add_argument("--explain", metavar="KERNEL.py",
                    help="grade one kernel file and print what the checker says about it")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.explain:
        import agent
        level = a.level or 3
        flags = {f for f in (a.features if a.features != "locate,state" else
                             "internal,origin,ahead,names,layout,strict").split(",") if f}
        for f in list(flags):
            if f in ("facts", "origin", "internal", "strict", "names", "pieces", "ahead", "layout"):
                flags |= {"state", "locate"}
            if f in ("pieces", "ahead", "layout"):
                flags.add("origin")
        agent.FEATURES.update(flags)
        agent.CANDIDATE_PATH = "/tmp/_explain_level{level}.py"
        reward, parts, told, _ = agent.grade(open(a.explain).read(), level)
        print(f"level {level}, flags {sorted(flags)}\nscore {reward:.2f}  {parts}\n")
        print(told)
        sys.exit(0 if reward >= 1.0 - 1e-9 else 1)
    if a.replay:
        sys.exit(replay(a.replay, a.level, {f for f in a.features.split(",") if f}))
    ap.print_help()


if __name__ == "__main__":
    main()
