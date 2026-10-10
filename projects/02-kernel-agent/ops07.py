"""Three operations added for task 07 through the harness's own extension point, `nkibench.level()`,
the same way their level 8 (attention) is added. Importing this module registers them as levels 9-11.

They are HELD OUT: the v6 messages were designed only from level 8's logged failures, and nothing in
v6 was changed after looking at how the agent does on these. Each has one shape with more than 128
rows (so it needs a loop over row tiles) and one value trap:

  9  row softmax   one shape scaled by 30, so exp() overflows unless the row max is subtracted first
 10  layer norm    one shape shifted by +5 and scaled by 3, so the mean is not 0 and the variance
                   not 1
 11  gated SiLU    silu(a) * b, elementwise, two inputs
"""
import numpy as np

import nkibench


def ref_softmax(x):
    """Softmax over each row: x [R, C] -> [R, C], every row sums to 1."""
    x32 = x.astype(np.float32)
    e = np.exp(x32 - x32.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(x.dtype)


def ref_layernorm(x):
    """Layer normalization over each row, with no learned scale or shift: x [R, C] -> [R, C]."""
    x32 = x.astype(np.float32)
    mean = x32.mean(axis=1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(axis=1, keepdims=True)
    return ((x32 - mean) / np.sqrt(var + 1e-5)).astype(x.dtype)


def ref_swiglu(a, b):
    """Gated SiLU: silu(a) * b elementwise, where silu(a) = a * sigmoid(a). a, b [R, C] -> [R, C]."""
    a32 = a.astype(np.float32)
    return (a32 / (1.0 + np.exp(-a32)) * b.astype(np.float32)).astype(a.dtype)


def _args_rows(spec, r):
    x = r.standard_normal((spec["R"], spec["C"])) * spec.get("scale", 1.0) + spec.get("shift", 0.0)
    return (x.astype(np.float32),)


def _args_two(spec, r):
    return tuple(r.standard_normal((spec["R"], spec["C"])).astype(np.float32) for _ in range(2))


def _label(sp):
    extra = "".join(f" {k}={sp[k]}" for k in ("scale", "shift") if k in sp)
    return f"R={sp['R']} C={sp['C']}{extra}"


nkibench.level(
    9, "row softmax", "nki_softmax_",
    "a row-wise reduction whose result is applied back to every element of its row",
    "none; a correctness level", ref_softmax,
    [dict(R=128, C=64), dict(R=64, C=512), dict(R=256, C=128, scale=30.0)],
    {"softmax", "log_softmax"},
    make_args=_args_rows, label=_label)

nkibench.level(
    10, "layer norm (no scale or shift)", "nki_layernorm_",
    "two row-wise reductions (mean, then variance) applied back to every element",
    "none; a correctness level", ref_layernorm,
    [dict(R=128, C=64), dict(R=64, C=512), dict(R=256, C=128, scale=3.0, shift=5.0)],
    {"layer_norm", "layernorm", "normalize", "batch_norm", "std", "var"},
    make_args=_args_rows, label=_label)

nkibench.level(
    11, "gated SiLU (SwiGLU)", "nki_swiglu_",
    "elementwise math on two inputs, with an activation function",
    "none; a correctness level", ref_swiglu,
    [dict(R=128, C=64), dict(R=64, C=512), dict(R=256, C=128)],
    {"silu", "swish", "swiglu", "sigmoid", "expit"},
    make_args=_args_two, label=_label)

HELD_OUT = (9, 10, 11)
