import numpy as np


def kernel(x):
    M, N = x.shape
    out = np.empty(M, dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        acc = np.full(i1 - i0, -1e9, dtype=np.float32)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            for j in range(j0, j1):
                acc = np.maximum(acc, x[i0:i1, j])
        out[i0:i1] = acc
    return out
