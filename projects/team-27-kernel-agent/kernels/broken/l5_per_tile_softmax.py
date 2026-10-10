import numpy as np


def kernel(x):
    M, N = x.shape
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            m = np.full(i1 - i0, -np.inf, dtype=np.float32)
            for j in range(j0, j1):
                m = np.maximum(m, x[i0:i1, j])
            s = np.zeros(i1 - i0, dtype=np.float64)
            for j in range(j0, j1):
                s += np.exp((x[i0:i1, j] - m).astype(np.float64))
            for j in range(j0, j1):
                out[i0:i1, j] = np.exp((x[i0:i1, j] - m).astype(np.float64)) / s
    return out
