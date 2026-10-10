"""Our reference for level 13 (row log-sum-exp). NOT shown to the agent."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_logsumexp_(x):
    R, C = x.shape
    out = nl.ndarray((R, 1), dtype=x.dtype, buffer=nl.shared_hbm)
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
        ls = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ls, op=nl.log, data=s)
        r = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r, data1=ls, data2=negmax, op=nl.subtract)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=r)
    return out
