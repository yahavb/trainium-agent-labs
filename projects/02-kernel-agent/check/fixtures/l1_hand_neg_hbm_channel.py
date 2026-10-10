import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out = nl.ndarray((C, H // p, W // p), dtype=x.dtype, buffer=nl.shared_hbm)
    # channel loop, but the channel index is only on HBM; every compute starts at partition 0
    for c in nl.affine_range(C):
        t1 = nl.ndarray((1, H, W), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t1, src=x[c:c + 1, :, :])
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                s = nl.sum(t1[0:1, h * p:(h + 1) * p, w * p:(w + 1) * p], axis=[1, 2], keepdims=True)
                m = nl.ndarray((1, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
                nisa.dma_copy(dst=out[c:c + 1, h:h + 1, w:w + 1], src=m)
    return out
