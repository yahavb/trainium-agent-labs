import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray((C, p, p), dtype=x.dtype, buffer=nl.sbuf)
    t = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    for i in nl.affine_range(H // p):
        for j in nl.affine_range(W // p):
            t_view = t[:, i*p:(i+1)*p, j*p:(j+1)*p]
            s = nl.sum(t_view, axis=[1, 2], keepdims=True)
            m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            nisa.dma_copy(dst=out[:, i, j], src=m)
    return out