import numpy as np


def kernel(x, a, b):
    M, N = x.shape
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            t = x[i0:i1, j0:j1] * a + b
            t[t < 0] = 0.0
            out[i0:i1, j0:j1] = t
    return out
