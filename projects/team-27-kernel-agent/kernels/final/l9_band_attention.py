# L9 band_attention: ours vc5e98d run1 (results/erfu-c5e98d5a75-partial/ours-L89-seat-245.jsonl); organizers' checker: 5/5 cases
import numpy as np

def kernel(q, k, v, w):
    S, d = q.shape
    out = np.zeros((S, d), dtype=np.float32)
    chunk_size = 128  # arbitrary chunk size to avoid full array operations

    for i in range(S):
        lo = max(0, i - w)
        hi = min(S - 1, i + w)
        n = hi - lo + 1

        # Initialize score array
        s = np.zeros(n, dtype=np.float64)

        # Compute scores in chunks
        for c0 in range(0, n, chunk_size):
            c1 = min(c0 + chunk_size, n)
            for c in range(d):
                s[c0:c1] += q[i, c] * k[lo + c0:lo + c1, c].astype(np.float64)

        # Compute max for numerical stability
        m = -np.inf
        for t in range(n):
            if s[t] > m:
                m = s[t]

        # Compute softmax in chunks
        p = np.zeros(n, dtype=np.float64)
        for c0 in range(0, n, chunk_size):
            c1 = min(c0 + chunk_size, n)
            p[c0:c1] = np.exp((s[c0:c1] - m) / np.sqrt(d))

        # Compute weighted sum of v in chunks
        acc = np.zeros(d, dtype=np.float64)
        total = 0.0
        for t in range(n):
            total += p[t]
            acc += p[t] * v[lo + t, :].astype(np.float64)

        # Normalize and assign to output
        out[i, :] = acc / total

    return out
