"""Shared helpers for the examples: trace the spec, build a Sched, print IR and NKI code."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nki_sched import NC_DEFAULT, Sched, arg, trace  # noqa: E402,F401
from nki_sched.emit import EmitError  # noqa: E402
from nki_sched.lower import HardwareError  # noqa: E402


def matmul_spec(lhsT, rhs):
    """The algorithm: C = lhsT.T @ rhs. lhsT is [K, M] (contraction on axis 0, as nc_matmul wants)."""
    return lhsT.T @ rhs


def matmul_proc(dtype="f32", name="matmul_kernel"):
    """Trace the spec above with symbolic sizes K, M, N and example shapes only for tracing."""
    return trace(
        matmul_spec,
        {"lhsT": arg(("K", "M"), dtype, example=(8, 4)), "rhs": arg(("K", "N"), dtype, example=(8, 8))},
        name=name,
    )


def new_sched(hw=NC_DEFAULT, **kw) -> Sched:
    return Sched(matmul_proc(**kw), hw)


def banner(example):
    """Print an example function's name and its full docstring."""
    import inspect

    print("\n" + "=" * 78 + f"\n{example.__name__}\n" + inspect.cleandoc(example.__doc__ or "") + "\n" + "=" * 78)


def show(s: Sched, what="", code=True):
    """Print the scheduled ns-level IR and, when it is fully lowered to instructions, the NKI source."""
    if what:
        print(f"-- {what}")
    print(s.show())
    if code:
        try:
            print("\n-- emitted NKI source:\n" + s.source())
        except (EmitError, HardwareError) as e:
            print(f"\n-- (not emittable yet: {e})")
