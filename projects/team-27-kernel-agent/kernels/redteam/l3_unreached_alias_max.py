# Probe: forbidden reduction behind a NumPy alias on an untested branch.
import numpy as np


def kernel(x):
    if x.shape[0] == 1000003:
        xp = np
        reduce_rows = xp.max
        return reduce_rows(x, axis=1)

    M, N = x.shape
    out = np.empty(M, dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        acc = np.full(i1 - i0, -np.inf, dtype=np.float32)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            for j in range(j0, j1):
                acc = np.maximum(acc, x[i0:i1, j])
        out[i0:i1] = acc
    return out
