import numpy as np


def kernel(x):
    M, N = x.shape
    out = np.zeros((N, M), dtype=x.dtype)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        for j0 in range(0, N - N % 512, 512):
            j1 = j0 + 512
            for i in range(i0, i1):
                out[j0:j1, i] = x[i, j0:j1]
    return out
