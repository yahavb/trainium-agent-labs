"""Our reference for level 11 (gated SiLU). NOT shown to the agent."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_swiglu_(a, b):
    R, C = a.shape
    out = nl.ndarray((R, C), dtype=a.dtype, buffer=nl.shared_hbm)
    P = min(128, R)
    for i in nl.affine_range(R // P):
        ta = nl.ndarray((P, C), dtype=a.dtype, buffer=nl.sbuf)
        tb = nl.ndarray((P, C), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ta, src=a[i * P:(i + 1) * P, :])
        nisa.dma_copy(dst=tb, src=b[i * P:(i + 1) * P, :])
        g = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g, op=nl.silu, data=ta)
        r = nl.ndarray((P, C), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=g, data2=tb, op=nl.multiply)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=r)
    return out
