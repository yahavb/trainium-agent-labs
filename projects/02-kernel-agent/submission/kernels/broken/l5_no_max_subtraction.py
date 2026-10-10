import numpy as np


def kernel(x):
    M, N = x.shape
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        s = np.zeros(i1 - i0, dtype=np.float32)
        for j0 in range(0, N, 512):
            for j in range(j0, min(j0 + 512, N)):
                s += np.exp(x[i0:i1, j])
        for j0 in range(0, N, 512):
            for j in range(j0, min(j0 + 512, N)):
                out[i0:i1, j] = np.exp(x[i0:i1, j]) / s
    return out
