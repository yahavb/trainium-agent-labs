import numpy as np


def kernel(x, eps=1e-5):
    g = np.ones(x.shape[1], dtype=np.float32)
    b = np.zeros(x.shape[1], dtype=np.float32)
    M, N = x.shape
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        s = np.zeros(i1 - i0, dtype=np.float32)
        sq = np.zeros(i1 - i0, dtype=np.float32)
        for j0 in range(0, N, 512):
            for j in range(j0, min(j0 + 512, N)):
                col = x[i0:i1, j]
                s += col
                sq += col * col
        mu = s / N
        var = sq / N - mu * mu
        rinv = 1.0 / np.sqrt(var + eps)
        for j0 in range(0, N, 512):
            for j in range(j0, min(j0 + 512, N)):
                out[i0:i1, j] = (x[i0:i1, j] - mu) * rinv * g[j] + b[j]
    return out
