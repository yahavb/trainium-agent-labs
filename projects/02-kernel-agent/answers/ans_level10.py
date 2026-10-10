"""Our reference for level 10 (layer norm, no scale or shift). NOT shown to the agent."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_layernorm_(x):
    R, C = x.shape
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    P = min(128, R)
    for i in nl.affine_range(R // P):
        t = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[i * P:(i + 1) * P, :])
        s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=s, op=nl.add, data=t, axis=[1], keepdims=True)
        negmean = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negmean, data=s, op0=nl.multiply, operand0=-1.0 / C)
        xc = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xc, data=t, op0=nl.add, operand0=negmean)
        sq = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=xc, data2=xc, op=nl.multiply)
        ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=[1], keepdims=True)
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, scale=1.0 / C, bias=1e-5)
        r = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=r, data=xc, op0=nl.multiply, operand0=rstd)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=r)
    return out
