# Qwen3-8B, live v3-rewrite seat-245, L9 run 1 attempt 4: 9/9 right after the rewrite, but np.dot on a
# (257, d) key window; the generic whole-array fix talked about x[i0:i1, j0:j1]
import numpy as np

def kernel(q, k, v, w):
    S, d = q.shape
    out = np.zeros((S, d), dtype=np.float32)
    chunk_size = 128
    feature_chunk_size = 512
    for i in range(S):
        lo = max(0, i - w)
        hi = min(S - 1, i + w)
        n = hi - lo + 1
        scores = np.zeros(n, dtype=np.float64)
        p = np.zeros(n, dtype=np.float64)
        for c in range(0, d, feature_chunk_size):
            c_end = min(c + feature_chunk_size, d)
            k_window = k[lo:hi+1, c:c_end]
            q_slice = q[i, c:c_end]
            for t in range(n):
                scores[t] += np.dot(q_slice, k_window[t])
        m = -np.inf
        for t in range(n):
            m = max(m, scores[t])
        for t in range(0, n, chunk_size):
            t_end = min(t + chunk_size, n)
            p[t:t_end] = np.exp((scores[t:t_end] - m) / np.sqrt(d))
        total = 0.0
        acc = np.zeros(d, dtype=np.float64)
        for t in range(n):
            total += p[t]
            acc += p[t] * v[lo + t, :]
        out[i, :] = acc / total
    return out
