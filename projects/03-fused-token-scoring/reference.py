"""Frozen host contract and stable FP32 oracle; no Neuron dependency."""

import numpy as np
import ml_dtypes

ATOL = 1e-3
RTOL = 1e-4


def validate_inputs(logits, indices):
    if not isinstance(logits, np.ndarray) or logits.dtype != ml_dtypes.bfloat16:
        raise TypeError("logits must be a NumPy BF16 array")
    if logits.ndim != 2 or min(logits.shape) < 1:
        raise ValueError("logits must have nonempty shape [T,V]")
    if logits.shape[1] > 2**24:
        raise ValueError("vocabulary must be <= 2**24 for exact FP32 index arithmetic")
    if not logits.flags.c_contiguous:
        raise ValueError("logits must be contiguous")
    if not isinstance(indices, np.ndarray) or indices.dtype != np.int32:
        raise TypeError("indices must be a NumPy int32 array")
    if indices.shape != (logits.shape[0],) or not indices.flags.c_contiguous:
        raise ValueError("indices must have contiguous shape [T]")
    promoted = logits.astype(np.float32)
    if not np.isfinite(promoted).all() or (np.abs(promoted) > 10000).any():
        raise ValueError("logits must be finite with absolute value <= 10000")
    if (indices < 0).any() or (indices >= logits.shape[1]).any():
        raise ValueError("selected index outside [0,V)")


def score_reference(logits, indices):
    validate_inputs(logits, indices)
    x = logits.astype(np.float32)
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis=1, dtype=np.float32)
    log_s = np.log(s)
    logprobs = z[np.arange(x.shape[0]), indices] - log_s
    entropy = log_s - np.sum(e * z, axis=1, dtype=np.float32) / s
    return logprobs.astype(np.float32), entropy.astype(np.float32)


def make_inputs(rows, vocab, seed=2026, std=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    logits = (rng.normal(0, std, (rows, vocab)) + offset).astype(ml_dtypes.bfloat16)
    indices = rng.integers(0, vocab, rows, dtype=np.int32)
    if rows >= 3:
        indices[:3] = [0, vocab // 2, vocab - 1]
    return logits, indices


def compare(actual, expected, vocab):
    errors = {}
    for name, got, want in zip(("logprobs", "entropy"), actual, expected):
        got = np.asarray(got)
        if got.shape != want.shape or got.dtype != np.float32:
            raise AssertionError(f"{name}: wrong shape/dtype {got.shape} {got.dtype}")
        if not np.isfinite(got).all():
            raise AssertionError(f"{name}: non-finite output")
        err = np.abs(got - want)
        allowed = ATOL + RTOL * np.abs(want)
        if not (err <= allowed).all():
            i = int(np.argmax(err / allowed))
            raise AssertionError(f"{name}[{i}]: {got[i]} vs {want[i]}, error={err[i]}, allowed={allowed[i]}")
        errors[name] = float(np.max(err))
    logprobs, entropy = actual
    if (logprobs > ATOL).any():
        raise AssertionError("log-probability > 0")
    if (entropy < -ATOL).any() or (entropy > np.log(vocab) + ATOL + RTOL * np.log(vocab)).any():
        raise AssertionError("entropy outside [0, log(V)]")
    return errors
