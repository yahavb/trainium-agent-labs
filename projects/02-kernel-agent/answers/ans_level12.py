"""Our reference for level 12 (row L2-normalize). NOT shown to the agent."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_l2norm_(x):
    R, C = x.shape
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    P = min(128, R)
    for i in nl.affine_range(R // P):
        t = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[i * P:(i + 1) * P, :])
        sq = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=t, data2=t, op=nl.multiply)
        ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=[1], keepdims=True)
        rn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rn, op=nl.rsqrt, data=ss, bias=1e-6)
        r = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r, data=t, op0=nl.multiply, operand0=rn)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=r)
    return out
