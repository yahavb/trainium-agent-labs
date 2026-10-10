"""
A STAND-IN for the Neuron SDK's `nki`, for exercising plumbing on a laptop that has no SDK.

It runs the shipped reference kernels for levels 3 and 4 on NumPy and raises on a few obvious
mistakes. It exists so that code which only needs "a kernel ran" or "a kernel raised on line N"
can be tested before it is copied to a seat pod.

IT IS NOT THE SIMULATOR. Its error messages and its rules are imitations. Never report a number
from it, and never treat how it behaves as evidence of how `nki.simulate` behaves.

    PYTHONPATH=devtools/fake_nki python diagnose.py --selftest
"""
import numpy as np

from . import isa, language, typing  # noqa: F401

__version__ = "0.0-fake"


class _Jitted:
    """kernel(*args) runs as Python; kernel[grid](*args) imitates a device launch by simulating."""
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__code__ = fn.__code__

    def __call__(self, *args, **kw):
        return self.fn(*args, **kw)

    def __getitem__(self, grid):
        return lambda *args: simulate(self.fn)(*args)


def jit(fn):
    return _Jitted(fn)


def simulate(kernel):
    def run(*args):
        wrapped = [language._Tensor(a, language.shared_hbm) if isinstance(a, np.ndarray) else a
                   for a in args]
        out = kernel(*wrapped)
        return out.arr if isinstance(out, language._Tensor) else out
    return run
