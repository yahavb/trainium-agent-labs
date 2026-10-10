# legal float32 two-pass layernorm: must pass (the bound allows float32 arithmetic)
import numpy as np


def kernel(x, eps=1e-5):
    M, N = x.shape
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        s = np.zeros(i1 - i0, dtype=np.float32)
        for j0 in range(0, N, 512):
            for j in range(j0, min(j0 + 512, N)):
                s += x[i0:i1, j].astype(np.float32)
        mu = s / N
        ss = np.zeros(i1 - i0, dtype=np.float32)
        for j0 in range(0, N, 512):
            for j in range(j0, min(j0 + 512, N)):
                d = x[i0:i1, j].astype(np.float32) - mu
                ss += d * d
        rinv = 1.0 / np.sqrt(ss / N + eps)
        for j0 in range(0, N, 512):
            for j in range(j0, min(j0 + 512, N)):
                out[i0:i1, j] = (x[i0:i1, j].astype(np.float32) - mu) * rinv
    return out
