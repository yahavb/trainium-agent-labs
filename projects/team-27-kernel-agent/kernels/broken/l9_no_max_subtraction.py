import numpy as np


def kernel(q, k, v, w):
    S, d = q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.empty((S, d), dtype=np.float32)
    for i in range(S):
        lo, hi = max(0, i - w), min(S - 1, i + w)
        n = hi - lo + 1
        s = np.zeros(n, dtype=np.float64)
        for c0 in range(0, n, 128):
            c1 = min(c0 + 128, n)
            for c in range(d):
                s[c0:c1] += q[i, c] * k[lo + c0:lo + c1, c].astype(np.float64)
        m = -np.inf
        for t in range(n):
            m = 0.0
        p = np.zeros(n, dtype=np.float64)
        for c0 in range(0, n, 128):
            c1 = min(c0 + 128, n)
            p[c0:c1] = np.exp(((s[c0:c1] - m) * scale).astype(np.float32))
        total = 0.0
        acc = np.zeros(d, dtype=np.float64)
        for t in range(n):
            total += p[t]
            acc += p[t] * v[lo + t, :]
        out[i, :] = acc / total
    return out
