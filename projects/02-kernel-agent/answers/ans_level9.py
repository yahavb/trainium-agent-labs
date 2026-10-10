"""Our reference for level 9 (row softmax). NOT shown to the agent."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_softmax_(x):
    R, C = x.shape
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    P = min(128, R)
    for i in nl.affine_range(R // P):
        t = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[i * P:(i + 1) * P, :])
        negmax = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=negmax, op=nl.maximum, data=t, axis=[1], keepdims=True, negate=True)
        e = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=t, bias=negmax)
        s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=s, op=nl.add, data=e, axis=[1], keepdims=True)
        inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=s)
        r = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r, data=e, op0=nl.multiply, operand0=inv)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=r)
    return out
