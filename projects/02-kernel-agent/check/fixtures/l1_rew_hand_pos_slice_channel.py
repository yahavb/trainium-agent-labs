import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out = nl.ndarray((C, H // p, W // p), dtype=x.dtype, buffer=nl.shared_hbm)
    # channel loop with a one-partition SLICE t[c:c + 1]: same failure expected as t[c]
    t = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    for h in nl.affine_range(H // p):
        for w in nl.affine_range(W // p):
            s = nl.sum(t[:, h * p:(h + 1) * p, w * p:(w + 1) * p], axis=[1, 2], keepdims=True)
            m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            nisa.dma_copy(dst=out[:, h:h + 1, w:w + 1], src=m)
    return out
