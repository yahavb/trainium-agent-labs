"""Runtime tile guard.

Kernel source runs with `numpy` swapped for a proxy, and every array it can reach is a `TArray`.
The AST check can't tell `x[i]` (int) from `x[idx]` (index array) or see how big a slice is; the
runtime can, so the two checks together give line-accurate violations with few false negatives.
Violations are recorded, not raised, so one run reports every broken rule plus the numerical result.
"""
import builtins as _builtins
import contextlib
import signal
import sys
import types

import numpy as _np

from . import rules
from .rules import TILE_C, TILE_R

KERNEL_FILENAME = "kernel.py"

_REC = None  # the active Recorder; TArray consults it so violations attribute to the kernel line


class Recorder:
    def __init__(self):
        self.violations = {}  # (line, kind) -> what
        self._quiet = 0
        self.reset_cost()

    def reset_cost(self):
        # A Stage-A stand-in for an NKI trace: inputs live in "HBM"; every elementwise op is one
        # instruction whose FLOPs are its output elements; slicing an input is a load and
        # writing into the returned array is a store.
        self.ops = self.flops = self.hbm_read = 0
        self.writes = {}

    @contextlib.contextmanager
    def quiet(self):
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    def add(self, kind, what):
        if self._quiet:
            return
        f, line = sys._getframe(1), None
        while f is not None:
            if f.f_code.co_filename == KERNEL_FILENAME:
                line = f.f_lineno
                break
            f = f.f_back
        self.violations.setdefault((line, kind), what)


def _rec():
    r = _REC
    return r if r is not None and not r._quiet else None


def _oversize(shape):
    if len(shape) == 0:
        return False
    if len(shape) == 1:
        return shape[0] > TILE_C
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows > TILE_R or shape[-1] > TILE_C


_IN_SETITEM = [0]  # numpy re-enters __getitem__ while assigning; that read is the write target


def _column_rows(arr_ndim, key, result_shape):
    """Rows spanned by a column slice: x[a:b, j] on a 2-D array is a 1-D result running down the ROW
    axis, so it is bounded by the 128-row tile, not by the 512-column one that _oversize applies to a
    1-D shape. Red-team finding on live v3-rewrite L9: k[lo:hi+1, c] read 259 rows with 0 violations,
    and x[:, j] over 131 rows passed L2. Applied to reads only: a transpose legally writes a 128x512
    input tile as a 512x128 output region (hand L6 writes out[j0:j1, i])."""
    if arr_ndim != 2 or len(result_shape) != 1:
        return 0
    k = key if isinstance(key, tuple) else (key,)
    if len(k) == 2 and isinstance(k[1], (int, _np.integer)) and not isinstance(k[1], bool):
        return result_shape[0]
    return 0


def _check_key(key):
    rec = _rec()
    if rec is None:
        return
    for k in key if isinstance(key, tuple) else (key,):
        if k is None:
            rec.add("newaxis", "indexing with None / np.newaxis")
        elif isinstance(k, (bool, _np.bool_)):
            rec.add("fancy-index", "boolean scalar index")
        elif isinstance(k, (int, _np.integer)) or k is Ellipsis:
            continue
        elif isinstance(k, slice):
            for part in (k.start, k.stop, k.step):
                if part is not None and not isinstance(part, (int, _np.integer)):
                    rec.add("fancy-index", f"slice bound of type {type(part).__name__}")
        elif isinstance(k, _np.ndarray) and k.dtype == bool:
            rec.add("fancy-index", "boolean mask used as an index")
        else:
            rec.add("fancy-index", f"index of type {type(k).__name__} (index arrays/lists)")


def _wrap(res):
    if isinstance(res, _np.ndarray) and not isinstance(res, TArray):
        return res.view(TArray)
    if isinstance(res, tuple):
        return tuple(_wrap(r) for r in res)
    return res


class TArray(_np.ndarray):
    def __array_finalize__(self, obj):
        rec = _rec()
        if rec is not None and _oversize(self.shape):
            rec.add("whole-array", f"an operation produced shape {self.shape}, larger than one "
                                   f"{TILE_R}x{TILE_C} tile")

    def __getitem__(self, key):
        _check_key(key)
        res = super().__getitem__(key)
        rec = _rec()
        if rec is not None and not _IN_SETITEM[0] and isinstance(res, _np.ndarray):
            n = _column_rows(self.ndim, key, res.shape)
            if n > TILE_R:
                rec.add("whole-array", f"a column slice of {n} rows, more than one {TILE_R}-row tile")
        r = _REC
        if r is not None and not r._quiet and self.__dict__.get("_hbm"):
            r.hbm_read += res.nbytes if isinstance(res, _np.ndarray) else self.itemsize
        return res

    def __setitem__(self, key, value):
        _check_key(key)
        rec = _rec()
        if rec is not None:
            with rec.quiet():
                target = _np.ndarray.__getitem__(self.view(_np.ndarray), key)
            tshape = getattr(target, "shape", ())
            if _oversize(tshape):
                rec.add("whole-array", f"assignment writes a region of shape {tshape}")
            rec.writes[id(self)] = rec.writes.get(id(self), 0) + int(_np.prod(tshape)) * self.itemsize
            vshape = getattr(value, "shape", ())
            squeeze = lambda s: tuple(d for d in s if d != 1)
            if vshape != () and squeeze(vshape) != squeeze(tshape):
                rec.add("broadcast", f"assigning shape {vshape} into a region of shape {tshape}")
        _IN_SETITEM[0] += 1
        try:
            super().__setitem__(key, value)
        finally:
            _IN_SETITEM[0] -= 1

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        rec = _rec()
        if rec is not None:
            if ufunc is _np.matmul:
                rec.add("matmul-op", "@ / np.matmul")
            elif method != "__call__":
                rec.add("reduction", f"np.{ufunc.__name__}.{method}")
            shapes = {a.shape for a in inputs if isinstance(a, _np.ndarray) and a.ndim > 0}
            if len(shapes) > 1 and method == "__call__" and ufunc is not _np.matmul:
                rec.add("broadcast", f"np.{ufunc.__name__} on operands of shapes "
                                     f"{' and '.join(map(str, sorted(shapes)))}")
        args = [a.view(_np.ndarray) if isinstance(a, TArray) else a for a in inputs]
        if out is not None:
            kwargs["out"] = tuple(o.view(_np.ndarray) if isinstance(o, TArray) else o for o in out)
        res = getattr(ufunc, method)(*args, **kwargs)
        r = _REC
        if r is not None and not r._quiet:
            r.ops += 1
            first = kwargs["out"][0] if out is not None else (res[0] if isinstance(res, tuple) else res)
            r.flops += getattr(first, "size", 1)
        if out is not None:
            return out[0] if len(out) == 1 else out
        return _wrap(res)

    def __array_function__(self, func, types_, args, kwargs):
        rec = _rec()
        if rec is not None:
            if func.__name__ in rules.BANNED_NP:
                rec.add(rules.np_kind(func.__name__), f"np.{func.__name__}")
            if "keepdims" in kwargs:
                rec.add("keepdims", "keepdims=")
        return super().__array_function__(func, types_, args, kwargs)


def _banned_method(name):
    base = getattr(_np.ndarray, name)

    def method(self, *args, **kwargs):
        rec = _rec()
        if rec is not None:
            kind = rules.np_kind(name) if name in rules.BANNED_NP else "layout"
            rec.add(kind, f".{name}()")
        return base(self, *args, **kwargs)

    method.__name__ = name
    return method


def _banned_attr(name):
    base = getattr(_np.ndarray, name)

    def getter(self):
        rec = _rec()
        if rec is not None:
            rec.add("layout", f".{name}")
        return base.__get__(self)

    return property(getter)


for _m in rules.BANNED_METHODS:
    # the guard itself needs .view(); the static check bans it in kernels
    if _m != "view" and hasattr(_np.ndarray, _m):
        setattr(TArray, _m, _banned_method(_m))
for _a in rules.BANNED_ATTRS:
    setattr(TArray, _a, _banned_attr(_a))

_ALLOCATORS = {"empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like",
               "full_like", "arange", "linspace"}


def _allocator(fn):
    def alloc(*args, **kwargs):
        rec = _REC
        with rec.quiet() if rec is not None else contextlib.nullcontext():
            res = fn(*args, **kwargs)
            return res.view(TArray) if isinstance(res, _np.ndarray) else res

    return alloc


class _NumpyProxy(types.ModuleType):
    def __getattr__(self, name):
        rec = _rec()
        if rec is not None and name in rules.BANNED_NP:
            rec.add(rules.np_kind(name), f"np.{name}")
        val = getattr(_np, name)
        return _allocator(val) if name in _ALLOCATORS else val


_NP_PROXY = _NumpyProxy("numpy")


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root == "numpy":
        if name != "numpy":
            rec = _rec()
            if rec is not None:
                rec.add(rules.np_kind(name.split(".")[1]), f"import {name}")
        return _NP_PROXY
    if root == "math":
        return __import__(name, globals, locals, fromlist, level)
    rec = _rec()
    if rec is not None:
        rec.add("import", f"import {name}")
    raise ImportError(f"import of {name!r} is not allowed in a kernel")


def _reducing_builtin(name, fn):
    def wrapper(*args, **kwargs):
        rec = _rec()
        if rec is not None and len(args) == 1:
            rec.add("builtin-reduce", f"{name}(...)")
        return fn(*args, **kwargs)

    return wrapper


def _kernel_builtins():
    b = dict(vars(_builtins))
    for name in rules.BANNED_BUILTINS:
        b.pop(name, None)
    b["__import__"] = _guarded_import
    for name in rules.REDUCING_BUILTINS:
        b[name] = _reducing_builtin(name, getattr(_builtins, name))
    return b


def load_kernel(source, fn_name="kernel"):
    """Compile and exec kernel source in a restricted namespace; returns (fn, recorder)."""
    global _REC
    rec = Recorder()
    ns = {"__builtins__": _kernel_builtins(), "__name__": "kernel"}
    code = compile(source, KERNEL_FILENAME, "exec")
    _REC = rec
    try:
        exec(code, ns)
    finally:
        _REC = None
    return ns.get(fn_name), rec


class KernelTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise KernelTimeout()


def call_kernel(fn, rec, args, timeout_s):
    """Run one case. Inputs are copied (a kernel that mutates its inputs can't poison later cases)."""
    global _REC
    guarded = [a.copy().view(TArray) if isinstance(a, _np.ndarray) else a for a in args]
    for g in guarded:
        if isinstance(g, TArray):
            g._hbm = True
    rec.reset_cost()
    _REC = rec
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        with _np.errstate(all="ignore"):  # overflow is the kernel's bug to report, not noise
            out = fn(*guarded)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        _REC = None
    rec.hbm_write = rec.writes.get(id(out), getattr(out, "nbytes", 0))
    if isinstance(out, _np.ndarray):
        out = out.view(_np.ndarray)
    return out
