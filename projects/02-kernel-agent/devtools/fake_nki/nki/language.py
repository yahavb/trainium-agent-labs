"""Imitation of nki.language. See the package docstring: plumbing tests only."""
import numpy as np


class _Region:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


sbuf, psum, shared_hbm = _Region("sbuf"), _Region("psum"), _Region("shared_hbm")
float32 = np.float32
bfloat16 = np.float32
multiply = np.multiply


class tile_size:
    pmax = 128
    gemm_stationary_fmax = 128
    gemm_moving_fmax = 512


class _Tensor:
    def __init__(self, arr, buffer):
        self.arr = arr
        self.buffer = buffer

    shape = property(lambda self: self.arr.shape)
    dtype = property(lambda self: self.arr.dtype)

    def __getitem__(self, idx):
        return _Tensor(self.arr[idx], self.buffer)


def affine_range(*args):
    return range(*args)


def ndarray(shape, dtype=np.float32, buffer=None):
    shape = tuple(shape)
    if buffer in (sbuf, psum) and len(shape) < 2:
        raise AssertionError("SBUF and PSUM tensors must have at least 2 dimensions "
                             "(partition-dim and free-dim)")
    return _Tensor(np.zeros(shape, dtype), buffer)
