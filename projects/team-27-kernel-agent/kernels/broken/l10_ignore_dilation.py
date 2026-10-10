import numpy as np


def kernel(x, w, stride, dilation):
    C_in, L = x.shape
    C_out, _, K = w.shape
    L_out = (L - dilation * (K - 1) - 1) // stride + 1
    out = np.empty((C_out, L_out), dtype=np.float32)
    for t0 in range(0, L_out, 512):
        t1 = min(t0 + 512, L_out)
        for o in range(C_out):
            acc = np.zeros(t1 - t0, dtype=np.float64)
            for c in range(C_in):
                for kk in range(K):
                    start = t0 * stride + kk
                    acc += w[o, c, kk] * x[c, start:start + (t1 - t0 - 1) * stride + 1:stride]
            out[o, t0:t1] = acc
    return out
