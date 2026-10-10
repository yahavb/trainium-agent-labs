import numpy as np


def kernel(a, b):
    M, K = a.shape
    N = b.shape[1]
    out = np.zeros((M, N), dtype=np.float32)
    for i0 in range(0, M, 128):
        i1 = min(i0 + 128, M)
        for j0 in range(0, N, 512):
            j1 = min(j0 + 512, N)
            acc = np.zeros((i1 - i0, j1 - j0), dtype=np.float64)
            for k0 in range(0, K - K % 128, 128):
                at = a[i0:i1, k0:k0 + 128]
                bt = b[k0:k0 + 128, j0:j1]
                for k in range(128):
                    for r in range(i1 - i0):
                        acc[r, :] += at[r, k] * bt[k, :]
            out[i0:i1, j0:j1] = acc
    return out
