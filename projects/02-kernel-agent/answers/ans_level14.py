"""Our reference for level 14 (clipped gate). NOT shown to the agent."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_clipgate_(x, y):
    R, C = x.shape
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    P = min(128, R)
    for i in nl.affine_range(R // P):
        tx = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        ty = nl.ndarray((P, C), dtype=y.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tx, src=x[i * P:(i + 1) * P, :])
        nisa.dma_copy(dst=ty, src=y[i * P:(i + 1) * P, :])
        c = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=c, data=tx, op0=nl.maximum, operand0=-1.0, op1=nl.minimum, operand1=1.0)
        r = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=c, data2=ty, op=nl.multiply)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=r)
    return out
