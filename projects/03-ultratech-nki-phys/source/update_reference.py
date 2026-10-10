"""CPU semantics for a fixed number of projected-gradient updates."""

import numpy as np


def prepare(fixture):
    count = len(fixture["b"])
    capacity = max(16, 1 << (count - 1).bit_length())
    if capacity > 128:
        raise ValueError("Initial NKI update supports at most 128 contacts")
    matrix = np.zeros((capacity, capacity), dtype=np.float32)
    matrix[:count, :count] = fixture["A"]
    bias = np.zeros((capacity, 1), dtype=np.float32)
    bias[:count, 0] = fixture["b"]
    alpha = float(1 / np.linalg.eigvalsh(fixture["A"])[-1])
    return np.ascontiguousarray(matrix.T), bias, np.zeros_like(bias), alpha


def fixed_steps(a_transposed, bias, initial, alpha, steps):
    impulses = initial.copy()
    for _ in range(steps):
        impulses = np.maximum(0, impulses - np.float32(alpha) * (a_transposed.T @ impulses + bias))
    return impulses
