"""Small hardware diagnostic; no model calls."""
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


def add(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    x = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    y = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    z = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=a)
    nisa.dma_copy(dst=y, src=b)
    nisa.tensor_tensor(dst=z, data1=x, data2=y, op=nl.add)
    nisa.dma_copy(dst=out, src=z)
    return out


if __name__ == "__main__":
    a = np.ones((128, 512), dtype=np.float32)
    b = np.ones_like(a)
    fn = nki.jit(add)
    out = fn[2](a, b)
    print("Direct hardware add:", out.min(), out.max(), "correct:", np.allclose(out, 2))
