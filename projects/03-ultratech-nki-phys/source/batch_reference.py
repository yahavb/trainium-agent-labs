"""Pack genuinely independent QPs and preserve per-world update semantics."""

import numpy as np
from update_reference import fixed_steps, prepare


def prepare_batch(fixtures):
    if not 1 <= len(fixtures) <= 16:
        raise ValueError("Batch must contain 1 to 16 worlds")
    counts = {len(f["b"]) for f in fixtures}
    if len(counts) != 1:
        raise ValueError("This baseline requires equal active contact counts")
    prepared = [prepare(f) for f in fixtures]
    matrix = np.ascontiguousarray(np.concatenate([p[0] for p in prepared], axis=1))
    bias = np.ascontiguousarray(np.concatenate([p[1] for p in prepared], axis=1))
    alpha = np.empty_like(bias)
    for world, p in enumerate(prepared):
        alpha[:, world] = p[3]
    return matrix, bias, np.zeros_like(bias), alpha


def fixed_batch(a_transposed, bias, initial, alpha, steps):
    capacity, worlds = bias.shape
    output = np.empty_like(bias)
    for world in range(worlds):
        output[:, world:world + 1] = fixed_steps(
            a_transposed[:, world * capacity:(world + 1) * capacity],
            bias[:, world:world + 1], initial[:, world:world + 1],
            float(alpha[0, world]), steps)
    return output
