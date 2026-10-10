import numpy as np


def kernel(x, eps=1e-6):
    g = np.ones(x.shape[1], dtype=np.float32)
    M, N = x.shape
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        ss = np.zeros(i1 - i0, dtype=np.float32)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            for j in range(j0, j1):
                col = x[i0:i1, j]
                ss += col * col
        rinv = 1.0 / np.sqrt(ss / N + eps)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            for j in range(j0, j1):
                out[i0:i1, j] = x[i0:i1, j] * rinv * g[j]
    return out
