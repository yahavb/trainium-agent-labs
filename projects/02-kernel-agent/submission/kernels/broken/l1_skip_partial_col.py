import numpy as np


def kernel(x, a, b):
    M, N = x.shape
    out = np.zeros((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        for j0 in range(0, N - N % 512, 512):
            j1 = j0 + 512
            out[i0:i1, j0:j1] = np.maximum(x[i0:i1, j0:j1] * a + b, 0.0)
    return out
