import numpy as np


def kernel(x, eps=1e-6):
    g = np.ones(x.shape[1], dtype=np.float32)
    M, N = x.shape
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            ss = np.zeros(i1 - i0, dtype=np.float64)
            for j in range(j0, j1):
                col = x[i0:i1, j].astype(np.float64)
                ss += col * col
            rinv = 1.0 / np.sqrt(ss / (j1 - j0) + eps)
            for j in range(j0, j1):
                out[i0:i1, j] = x[i0:i1, j] * rinv * g[j]
    return out
