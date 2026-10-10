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
    # Reshape to (C, H//p, p, W//p, p)
    # Use tile views to compute the mean
    tile = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    # Use affine_range to loop over the tiles
    for i in nl.affine_range(H // p):
        for j in nl.affine_range(W // p):
            # Tile view for the current window
            window = t[:, i*p:(i+1)*p, j*p:(j+1)*p]
            # Compute the sum over the window
            s = nl.sum(window, axis=[1, 2], keepdims=True)
            # Compute the mean by scaling
            m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            # Copy to output
            out[:, i, j] = m
    return out