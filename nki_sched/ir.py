"""The ns-level loop IR: explicit loops, allocations with memory spaces, scalar statements and
engine-explicit instruction calls (`ns.<engine>.<inst>`).

Everything is an immutable dataclass; schedule primitives rebuild the parts they touch. Names
(loop variables, buffers) are unique within a Proc and double as v0 handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Optional

from .expr import Aff

HBM, SBUF, PSUM = "HBM", "SBUF", "PSUM"
MEMS = (HBM, SBUF, PSUM)


# ---------------------------------------------------------------- buffers / windows
@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple  # tuple[Aff]
    dtype: str  # "f32" | "bf16" | "f16" | "like:<argname>"
    mem: str = HBM
    role: str = "temp"  # "arg" | "out" | "temp"

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Window:
    """buf[lo0:lo0+size0, lo1:lo1+size1, ...] (sizes are Aff, normally constants)."""

    buf: str
    lo: tuple
    size: tuple
    points: tuple = ()  # dims indexed by an integer (size 1) that are dropped from the window's rank

    def is_full(self, buffers: dict) -> bool:
        b = buffers[self.buf]
        return not self.points and all(l == Aff(0) for l in self.lo) and tuple(self.size) == tuple(b.shape)

    def shape(self) -> tuple:
        """sizes of the dims that remain after dropping `points`"""
        return tuple(z for i, z in enumerate(self.size) if i not in self.points)


# ---------------------------------------------------------------- data expressions
@dataclass(frozen=True)
class Read:
    buf: str
    idx: tuple


@dataclass(frozen=True)
class Mul:
    a: Any
    b: Any


@dataclass(frozen=True)
class Add:
    a: Any
    b: Any


@dataclass(frozen=True)
class CastE:
    dtype: str
    e: Any


@dataclass(frozen=True)
class Lit:
    value: float


# ---------------------------------------------------------------- statements
@dataclass(frozen=True)
class Alloc:
    buf: Buffer


@dataclass(frozen=True)
class For:
    var: str
    extent: Aff
    body: tuple
    stage: str = ""  # which algorithm stage this loop belongs to ("acc", "acc.init", "C")
    kind: str = "auto"  # auto | affine | sequential | static | spmd  (auto => inferred at emission; spmd => bound to nl.program_id)


@dataclass(frozen=True)
class Assign:
    buf: str
    idx: tuple
    rhs: Any


@dataclass(frozen=True)
class Reduce:
    """buf[idx] += rhs"""

    buf: str
    idx: tuple
    rhs: Any


@dataclass(frozen=True)
class Call:
    instr: str  # "ns.tensor.matmul", "ns.vector.tensor_copy", "ns.sync.dma_copy", ...
    args: tuple  # ((role, Window), ...)
    attrs: tuple = ()  # ((key, value), ...)

    def arg(self, role: str) -> Window:
        for r, w in self.args:
            if r == role:
                return w
        raise KeyError(role)

    def attr(self, key, default=None):
        return dict(self.attrs).get(key, default)


@dataclass(frozen=True)
class Assumption:
    """`expr % mod == 0`, required by a `split(perfect=True)`; the emitter turns it into an assert."""

    expr: Aff
    mod: int

    def __str__(self):
        return f"{self.expr} % {self.mod} == 0"


@dataclass(frozen=True)
class Proc:
    name: str
    args: tuple  # tuple[Buffer], role arg/out
    sizes: tuple  # ((sym, argname, dim), ...)  how to read each size symbol from an argument
    body: tuple
    assumptions: tuple = ()
    params: tuple = ()  # ((name, Aff), ...) derived Python-level parameters (shape-dependent knobs)

    def with_(self, **kw):
        return replace(self, **kw)


# ---------------------------------------------------------------- traversal
def walk(stmts) -> Iterator:
    for s in stmts:
        yield s
        if isinstance(s, For):
            yield from walk(s.body)


def buffers_of(proc: Proc) -> dict:
    """name -> Buffer for args and every Alloc in the program (names are unique)."""
    out = {b.name: b for b in proc.args}
    for s in walk(proc.body):
        if isinstance(s, Alloc):
            if s.buf.name in out and s.buf.name not in {b.name for b in proc.args}:
                raise ValueError(f"duplicate allocation of {s.buf.name}")
            out[s.buf.name] = s.buf
    return out


def find_loop(stmts, var: str) -> Optional[For]:
    for s in walk(stmts):
        if isinstance(s, For) and s.var == var:
            return s
    return None


def loop_vars(stmts) -> list:
    return [s.var for s in walk(stmts) if isinstance(s, For)]


def _map_expr(e, f):
    if isinstance(e, Read):
        return Read(e.buf, tuple(f(i) for i in e.idx))
    if isinstance(e, (Mul, Add)):
        return type(e)(_map_expr(e.a, f), _map_expr(e.b, f))
    if isinstance(e, CastE):
        return CastE(e.dtype, _map_expr(e.e, f))
    return e


def map_aff(s, f):
    """Apply f (Aff -> Aff) to every index/extent expression inside statement s (recursively)."""
    if isinstance(s, For):
        return replace(s, extent=f(s.extent), body=tuple(map_aff(x, f) for x in s.body))
    if isinstance(s, Assign):
        return Assign(s.buf, tuple(f(i) for i in s.idx), _map_expr(s.rhs, f))
    if isinstance(s, Reduce):
        return Reduce(s.buf, tuple(f(i) for i in s.idx), _map_expr(s.rhs, f))
    if isinstance(s, Alloc):
        return Alloc(s.buf.with_(shape=tuple(f(d) for d in s.buf.shape)))
    if isinstance(s, Call):
        return Call(
            s.instr,
            tuple((r, Window(w.buf, tuple(f(l) for l in w.lo), tuple(f(z) for z in w.size), w.points)) for r, w in s.args),
            s.attrs,
        )
    raise TypeError(s)


def subst_vars(s, m: dict):
    return map_aff(s, lambda a: a.subs(m))


def _remap_expr(e, buf, new, f):
    if isinstance(e, Read):
        return Read(new, f(e.idx)) if e.buf == buf else e
    if isinstance(e, (Mul, Add)):
        return type(e)(_remap_expr(e.a, buf, new, f), _remap_expr(e.b, buf, new, f))
    if isinstance(e, CastE):
        return CastE(e.dtype, _remap_expr(e.e, buf, new, f))
    return e


def remap_buf(s, buf: str, new: str, f):
    """Redirect every access to `buf` (scalar reads/writes and window bases) to buffer `new`,
    transforming the index tuple with f(idx) -> idx. Window sizes are unchanged."""
    if isinstance(s, For):
        return replace(s, body=tuple(remap_buf(x, buf, new, f) for x in s.body))
    if isinstance(s, Assign):
        rhs = _remap_expr(s.rhs, buf, new, f)
        return Assign(new, f(s.idx), rhs) if s.buf == buf else Assign(s.buf, s.idx, rhs)
    if isinstance(s, Reduce):
        rhs = _remap_expr(s.rhs, buf, new, f)
        return Reduce(new, f(s.idx), rhs) if s.buf == buf else Reduce(s.buf, s.idx, rhs)
    if isinstance(s, Call):
        return Call(s.instr, tuple((r, Window(new, f(w.lo), w.size, w.points) if w.buf == buf else w) for r, w in s.args), s.attrs)
    return s


def map_stmts(stmts, f):
    """Bottom-up rewrite: f(stmt) -> stmt | tuple[stmt] | None (None keeps; tuple splices)."""
    out = []
    for s in stmts:
        if isinstance(s, For):
            s = replace(s, body=tuple(map_stmts(s.body, f)))
        r = f(s)
        if r is None:
            out.append(s)
        elif isinstance(r, tuple):
            out.extend(r)
        else:
            out.append(r)
    return out


def expr_reads(e) -> Iterator[Read]:
    if isinstance(e, Read):
        yield e
    elif isinstance(e, (Mul, Add)):
        yield from expr_reads(e.a)
        yield from expr_reads(e.b)
    elif isinstance(e, CastE):
        yield from expr_reads(e.e)


def accesses(s):
    """Yield (mode, bufname, idx_or_window) for scalar and window accesses in a statement
    (recursing through loops). mode in {"r","w","rw"}; Reduce is "rw"."""
    if isinstance(s, For):
        for x in s.body:
            yield from accesses(x)
    elif isinstance(s, Assign):
        for r in expr_reads(s.rhs):
            yield ("r", r.buf, r.idx)
        yield ("w", s.buf, s.idx)
    elif isinstance(s, Reduce):
        for r in expr_reads(s.rhs):
            yield ("r", r.buf, r.idx)
        yield ("rw", s.buf, s.idx)
    elif isinstance(s, Call):
        for role, w in s.args:
            mode = "w" if role == "dst" else "r"
            if role == "dst" and s.instr.endswith("matmul") and s.attr("accumulate", None) is not False:
                mode = "rw"
            yield (mode, w.buf, w)


# ---------------------------------------------------------------- printing
def _e(e) -> str:
    if isinstance(e, Read):
        return f"{e.buf}[{', '.join(str(i) for i in e.idx)}]"
    if isinstance(e, Mul):
        return f"{_e(e.a)} * {_e(e.b)}"
    if isinstance(e, Add):
        return f"{_e(e.a)} + {_e(e.b)}"
    if isinstance(e, CastE):
        return f"cast_{e.dtype}({_e(e.e)})"
    if isinstance(e, Lit):
        return repr(e.value)
    raise TypeError(e)


def _w(w: Window) -> str:
    def dim(i, l, z):
        if i in w.points:
            return str(l)
        return f"{l}:+{z}"

    return f"{w.buf}[{', '.join(dim(i, l, z) for i, (l, z) in enumerate(zip(w.lo, w.size)))}]"


def fmt(stmts, ind=0) -> str:
    pad = "  " * ind
    lines = []
    for s in stmts:
        if isinstance(s, Alloc):
            b = s.buf
            lines.append(f"{pad}alloc {b.name} : {b.dtype}[{', '.join(str(d) for d in b.shape)}] @ {b.mem}")
        elif isinstance(s, For):
            kind = "" if s.kind == "auto" else f" {s.kind}"
            lines.append(f"{pad}for {s.var} in 0..{s.extent}{kind}:")
            lines.append(fmt(s.body, ind + 1))
        elif isinstance(s, Assign):
            lines.append(f"{pad}{s.buf}[{', '.join(str(i) for i in s.idx)}] = {_e(s.rhs)}")
        elif isinstance(s, Reduce):
            lines.append(f"{pad}{s.buf}[{', '.join(str(i) for i in s.idx)}] += {_e(s.rhs)}")
        elif isinstance(s, Call):
            args = ", ".join(f"{r}={_w(w)}" for r, w in s.args)
            attrs = "".join(f", {k}={v}" for k, v in s.attrs)
            lines.append(f"{pad}{s.instr}({args}{attrs})")
        else:
            raise TypeError(s)
    return "\n".join(l for l in lines if l != "")


def fmt_proc(p: Proc) -> str:
    head = f"proc {p.name}({', '.join(b.name + ': ' + b.dtype + '[' + ', '.join(str(d) for d in b.shape) + ']' for b in p.args)})"
    if p.assumptions:
        head += "\n  assume " + "; ".join(str(a) for a in p.assumptions)
    return head + "\n" + fmt(p.body, 1)


Proc.__str__ = fmt_proc
