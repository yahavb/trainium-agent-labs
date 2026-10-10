"""Three more operations for task 08, registered as levels 12-14 the same way as ops07.py. They are
HELD OUT: task 08's first-prompt change (PROMPT1=v2) was written from levels 9-11's failures in task 07,
and nothing was changed after looking at how the agent does on these.

 12  row L2-normalize   x / sqrt(sum(x^2) + 1e-6) per row; one shape scaled by 10
 13  row log-sum-exp    log(sum(exp(x))) per row, one value per row; one shape scaled by 30, where exp()
                        overflows unless the row max is taken out first
 14  clipped gate       clip(x, -1, 1) * y, elementwise, two inputs
"""
import numpy as np

import nkibench
from ops07 import _args_rows, _args_two, _label


def ref_l2norm(x):
    """Scale each row of x [R, C] to unit length: x / sqrt(sum(x^2) + 1e-6) per row."""
    x32 = x.astype(np.float32)
    return (x32 / np.sqrt((x32 * x32).sum(axis=1, keepdims=True) + 1e-6)).astype(x.dtype)


def ref_logsumexp(x):
    """log(sum(exp(x))) over each row: x [R, C] -> [R, 1]."""
    x32 = x.astype(np.float32)
    m = x32.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x32 - m).sum(axis=1, keepdims=True))).astype(x.dtype)


def ref_clipgate(x, y):
    """clip(x, -1, 1) * y elementwise: x, y [R, C] -> [R, C]."""
    return (np.clip(x.astype(np.float32), -1.0, 1.0) * y.astype(np.float32)).astype(x.dtype)


nkibench.level(
    12, "row L2-normalize", "nki_l2norm_",
    "a row-wise reduction applied back to every element of its row",
    "none; a correctness level", ref_l2norm,
    [dict(R=128, C=64), dict(R=64, C=512), dict(R=256, C=128, scale=10.0)],
    {"normalize", "norm", "l2_normalize"},
    make_args=_args_rows, label=_label)

nkibench.level(
    13, "row log-sum-exp", "nki_logsumexp_",
    "a row-wise reduction to one value per row, kept numerically stable",
    "none; a correctness level", ref_logsumexp,
    [dict(R=128, C=64), dict(R=64, C=512), dict(R=256, C=128, scale=30.0)],
    {"logsumexp", "log_softmax", "softmax"},
    make_args=_args_rows, label=_label)

nkibench.level(
    14, "clipped gate", "nki_clipgate_",
    "elementwise math on two inputs, with a clamp",
    "none; a correctness level", ref_clipgate,
    [dict(R=128, C=64), dict(R=64, C=512), dict(R=256, C=128)],
    {"clip", "clamp", "hardtanh"},
    make_args=_args_two, label=_label)

HELD_OUT = (12, 13, 14)
