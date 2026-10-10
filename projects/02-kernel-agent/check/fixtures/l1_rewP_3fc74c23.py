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
    # Compute the average over the pool window
    # Use tile.ap to create a strided view for reduction
    # Partition along channel axis (axis 0)
    tile = nl.ndarray((C, p, p), dtype=x.dtype, buffer=nl.sbuf)
    # Loop over the input tensor
    for h in nl.affine_range(H // p):
        for w in nl.affine_range(W // p):
            nisa.tensor_copy(dst=tile[:, :, :], src=t[:, h * p:(h + 1) * p, w * p:(w + 1) * p])
            sum_tile = nl.sum(tile, axis=[1, 2], keepdims=True)
            nisa.tensor_scalar(dst=tile, data=sum_tile, op0=nl.multiply, operand0=1.0 / (p * p))
            nisa.dma_copy(dst=out[:, h:h + 1, w:w + 1], src=tile[:, 0:1, 0:1])
    return out
