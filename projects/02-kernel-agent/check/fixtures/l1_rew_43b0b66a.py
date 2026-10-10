import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    # Compute the average over the pool regions
    # Reshape to (C, H//p, p, W//p, p)
    # Use tile views and reductions
    # Loop over each channel
    for h in nl.affine_range(H // p):
        for w in nl.affine_range(W // p):
            tile = t[:, h * p:(h + 1) * p, w * p:(w + 1) * p]
            mean = nl.sum(tile, axis=[1, 2], keepdims=True)
            scaled_mean = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=scaled_mean, data=mean, op0=nl.multiply, operand0=1.0 / (p * p))
            nisa.dma_copy(dst=out[:, h:h + 1, w:w + 1], src=scaled_mean[:, 0:1, 0:1])
    return out
