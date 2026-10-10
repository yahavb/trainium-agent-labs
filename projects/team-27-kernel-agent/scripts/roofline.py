"""Roofline time model shared by the optimiser and the analysis.

t = max(bytes moved to/from HBM, FLOPs / RIDGE), in units of HBM bytes. RIDGE is the Trainium
figure nkibench uses (222 FLOP/byte, published for bf16, indicative for float32). time_x = t / t_min,
where t_min uses the op's minimal traffic (each input read once, output written once) and minimal
FLOPs: 1.00 means at the roofline.
"""
import contextlib

import numpy as np

from kagent import guard
from kagent.harness import verify

RIDGE = 222.0
_plain_getitem = guard.TArray.__getitem__


def _getitem_counting_scalars(self, key):
    # x[r, c] normally returns a numpy scalar, and arithmetic on numpy scalars never reaches
    # TArray.__array_ufunc__, so per-element python loops looked free (0 FLOPs, 0 instructions).
    # A 0-d TArray behaves like the scalar but every op on it is counted.
    res = _plain_getitem(self, key)
    if isinstance(res, np.generic):
        rec = guard._REC
        with rec.quiet() if rec is not None else contextlib.nullcontext():
            res = np.asarray(res).view(guard.TArray)
    return res


def measure(level, code):
    """Cost of a kernel with scalar ops counted. Only for efficiency: pass/fail always comes from
    the unpatched harness, which is what the benchmark uses."""
    guard.TArray.__getitem__ = _getitem_counting_scalars
    try:
        return metrics(verify(level, code))
    finally:
        guard.TArray.__getitem__ = _plain_getitem


def metrics(rep):
    c = rep.cost
    moved = c["hbm_read"] + c["hbm_write"]
    t = max(moved, c["flops"] / RIDGE)
    t_min = max(c["min_read"] + c["min_write"], c["min_flops"] / RIDGE)
    return dict(flops_x=c["flops"] / c["min_flops"] if c["min_flops"] else None,
                reads_x=c["hbm_read"] / c["min_read"], writes_x=c["hbm_write"] / c["min_write"],
                intensity=c["flops"] / moved if moved else 0.0, time_x=t / t_min,
                bound="compute" if c["flops"] / RIDGE > moved else "memory",
                ops=c["ops"], elems_per_op=c["flops"] / max(c["ops"], 1), case=c["case"])
