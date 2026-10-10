"""numpy interpreter for the ns-level IR. It executes any intermediate program a schedule produces,
which is how every primitive is differential-tested against eager torch.

Storage is float32 everywhere; stores to bf16/f16 buffers are rounded to that precision so numerics
match the declared dtypes. Matmul accumulate semantics follow nc_matmul: with accumulate=None the
first write to a freshly-allocated PSUM region overwrites and later writes add.
"""

from __future__ import annotations

import numpy as np

from . import ir
from .expr import Aff


def quantize(dtype: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if dtype == "f32":
        return x
    if dtype == "f16":
        with np.errstate(over="ignore"):  # overflow to inf is the intended f16 behaviour
            return x.astype(np.float16).astype(np.float32)
    if dtype == "bf16":
        # round-to-nearest-even to the top 16 bits of the float32 pattern
        u = np.array(x, dtype=np.float32).view(np.uint32)
        lsb = (u >> np.uint32(16)) & np.uint32(1)
        u = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
        return u.view(np.float32)
    raise ValueError(dtype)


class InterpError(Exception):
    pass


class _State:
    def __init__(self, proc: ir.Proc, inputs: dict):
        self.proc = proc
        self.reverse_affine = False
        self.bufs = {}
        self.fresh = {}  # name -> bool mask: True where not yet written since allocation
        self.dtypes = {}
        self.env = {}
        arg_dtype = {}
        for b in proc.args:
            if b.role == "arg":
                if b.name not in inputs:
                    raise InterpError(f"missing input {b.name}")
                arr = np.asarray(inputs[b.name])
                arg_dtype[b.name] = _dtype_of(arr, inputs.get("__dtypes__", {}).get(b.name))
        self.arg_dtype = arg_dtype
        # size symbols
        for sym, argname, dim in proc.sizes:
            self.env[sym] = int(np.asarray(inputs[argname]).shape[dim])
        for name, expr in proc.params:
            self.env[name] = expr.eval(self.env)
        for a in proc.assumptions:
            if a.expr.eval(self.env) % a.mod != 0:
                raise InterpError(f"assumption violated: {a}")
        for b in proc.args:
            if b.role == "arg":
                self.bufs[b.name] = np.asarray(inputs[b.name], dtype=np.float32).copy()
                self.dtypes[b.name] = self.arg_dtype[b.name]
            else:
                self._alloc(b)

    def resolve_dtype(self, d: str) -> str:
        if d.startswith("like:"):
            return self.arg_dtype[d[5:]]
        return d

    def _alloc(self, b: ir.Buffer):
        shape = tuple(int(d.eval(self.env)) for d in b.shape)
        self.bufs[b.name] = np.full(shape, np.nan, dtype=np.float32)
        self.fresh[b.name] = np.ones(shape, dtype=bool)
        self.dtypes[b.name] = self.resolve_dtype(b.dtype)

    def win(self, w: ir.Window):
        arr = self.bufs[w.buf]
        sl = []
        for i, (l, z) in enumerate(zip(w.lo, w.size)):
            lo, n = int(l.eval(self.env)), arr.shape[i]
            hi = lo + int(z.eval(self.env))
            if lo < 0 or hi > n:
                raise InterpError(f"window out of bounds on {w.buf}: dim {i} [{lo}:{hi}] vs shape {arr.shape}")
            sl.append(lo if i in w.points else slice(lo, hi))
        return tuple(sl)

    def idx(self, buf, idx):
        t = tuple(int(i.eval(self.env)) for i in idx)
        shape = self.bufs[buf].shape
        for x, n in zip(t, shape):
            if x < 0 or x >= n:
                raise InterpError(f"index out of bounds on {buf}: {t} vs shape {shape}")
        return t

    def ev(self, e):
        if isinstance(e, ir.Read):
            v = self.bufs[e.buf][self.idx(e.buf, e.idx)]
            if np.isnan(v):
                raise InterpError(f"read of uninitialised {e.buf}[{e.idx}]")
            return v
        if isinstance(e, ir.Mul):
            return np.float32(self.ev(e.a) * self.ev(e.b))
        if isinstance(e, ir.Add):
            return np.float32(self.ev(e.a) + self.ev(e.b))
        if isinstance(e, ir.CastE):
            return quantize(self.resolve_dtype(e.dtype), np.float32(self.ev(e.e)))
        if isinstance(e, ir.Lit):
            return np.float32(e.value)
        raise TypeError(e)

    def store(self, buf, t, v, accumulate=False):
        dt = self.dtypes[buf]
        arr = self.bufs[buf]
        if accumulate:
            if np.isnan(arr[t]):
                raise InterpError(f"accumulate into uninitialised {buf}{t}")
            v = arr[t] + v
        arr[t] = quantize(dt, np.float32(v))
        if buf in self.fresh:
            self.fresh[buf][t] = False


def run(proc: ir.Proc, inputs: dict, reverse_affine: bool = False):
    """Execute proc; returns {output name: array}. `inputs` maps arg name -> numpy array (any
    float dtype; bf16 inputs should be given as float32 arrays holding bf16-representable values
    with inputs['__dtypes__'] = {'name': 'bf16'}). With reverse_affine, loops whose kind is
    "affine" run their iterations in reverse order (used to test loop-kind claims)."""
    st = _State(proc, inputs)
    st.reverse_affine = reverse_affine
    _exec_block(st, proc.body)
    return {b.name: st.bufs[b.name].copy() for b in proc.args if b.role == "out"}


def _dtype_of(arr, override):
    if override:
        return override
    d = np.asarray(arr).dtype
    if d == np.float32:
        return "f32"
    if d == np.float16:
        return "f16"
    return "f32"


# ---------------------------------------------------------------- statements
def _exec_alloc(st: _State, s: ir.Alloc):
    st._alloc(s.buf)


def _exec_for(st: _State, s: ir.For):
    n = int(s.extent.eval(st.env))
    order = range(n - 1, -1, -1) if (st.reverse_affine and s.kind in ("affine", "spmd")) else range(n)
    for i in order:
        st.env[s.var] = i
        _exec_block(st, s.body)
    st.env.pop(s.var, None)


def _exec_assign(st: _State, s: ir.Assign):
    st.store(s.buf, st.idx(s.buf, s.idx), st.ev(s.rhs))


def _exec_reduce(st: _State, s: ir.Reduce):
    st.store(s.buf, st.idx(s.buf, s.idx), st.ev(s.rhs), accumulate=True)


def _exec_call(st: _State, c: ir.Call):
    try:
        handler = CALLS[c.instr]
    except KeyError:
        raise InterpError(f"unknown instruction {c.instr}") from None
    handler(st, c)


STMTS = {
    ir.Alloc: _exec_alloc,
    ir.For: _exec_for,
    ir.Assign: _exec_assign,
    ir.Reduce: _exec_reduce,
    ir.Call: _exec_call,
}


def _exec_block(st: _State, stmts):
    for s in stmts:
        STMTS[type(s)](st, s)


# ---------------------------------------------------------------- instructions
def _copy(st: _State, c: ir.Call):
    """dma_copy / tensor_copy on any engine: dst window = src window (cast to dst dtype)."""
    dst, src = c.arg("dst"), c.arg("src")
    ds, ss = st.win(dst), st.win(src)
    v = st.bufs[src.buf][ss]
    if v.shape != st.bufs[dst.buf][ds].shape:
        raise InterpError(f"{c.instr}: shape mismatch {v.shape} vs {st.bufs[dst.buf][ds].shape}")
    if np.isnan(v).any():
        raise InterpError(f"{c.instr}: copying uninitialised data from {src.buf}")
    st.bufs[dst.buf][ds] = quantize(st.dtypes[dst.buf], v)
    if dst.buf in st.fresh:
        st.fresh[dst.buf][ds] = False


def _matmul(st: _State, c: ir.Call):
    """dst (+)= stationary.T @ moving, with nc_matmul's accumulate semantics:
    None -> first write into a freshly allocated region overwrites, later writes add;
    True -> add onto existing contents (must be initialised); False -> overwrite."""
    dst, sta, mov = c.arg("dst"), c.arg("stationary"), c.arg("moving")
    ds = st.win(dst)
    a = st.bufs[sta.buf][st.win(sta)]
    b = st.bufs[mov.buf][st.win(mov)]
    if np.isnan(a).any() or np.isnan(b).any():
        raise InterpError("matmul operand reads uninitialised data")
    prod = a.astype(np.float32).T @ b.astype(np.float32)
    acc = c.attr("accumulate", None)
    cur_dst = st.bufs[dst.buf][ds]
    if prod.shape != cur_dst.shape:
        raise InterpError(f"matmul: result {prod.shape} vs dst {cur_dst.shape}")
    if acc is None:
        fresh = st.fresh[dst.buf][ds]
        if fresh.any() and not fresh.all():
            raise InterpError("matmul accumulate=None onto a partially-written region")
        st.bufs[dst.buf][ds] = (np.where(fresh, 0.0, cur_dst) + prod).astype(np.float32)
    elif acc is True:
        if np.isnan(cur_dst).any():
            raise InterpError("matmul accumulate=True onto uninitialised PSUM")
        st.bufs[dst.buf][ds] = cur_dst + prod
    else:
        st.bufs[dst.buf][ds] = prod
    st.fresh[dst.buf][ds] = False


CALLS = {
    "ns.sync.dma_copy": _copy,
    "ns.vector.tensor_copy": _copy,
    "ns.scalar.tensor_copy": _copy,
    "ns.gpsimd.tensor_copy": _copy,
    "ns.tensor.matmul": _matmul,
}
