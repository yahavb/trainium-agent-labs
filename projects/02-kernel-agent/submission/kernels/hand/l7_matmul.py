import numpy as np


def kernel(a, b):
    M, K = a.shape
    N = b.shape[1]
    out = np.empty((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            acc = np.zeros((i1 - i0, j1 - j0), dtype=np.float64)
            for k0 in range(0, K, 128):
                k1 = min(k0 + 128, K)
                at = a[i0:i1, k0:k1].astype(np.float64)
                bt = b[k0:k1, j0:j1].astype(np.float64)
                for k in range(k1 - k0):
                    for r in range(i1 - i0):
                        acc[r, :] += at[r, k] * bt[k, :]
            out[i0:i1, j0:j1] = acc
    return out
