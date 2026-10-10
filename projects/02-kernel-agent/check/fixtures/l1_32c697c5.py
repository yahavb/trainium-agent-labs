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

    # Initialize the output to zero to avoid uninitialized values
    nisa.dma_copy(dst=out, src=nl.zeros(out_shape, dtype=x.dtype))

    for i in nl.affine_range(H // p):
        for j in nl.affine_range(W // p):
            # Copy the tile to the shared buffer
            t_view = t[:, i*p:(i+1)*p, j*p:(j+1)*p]
            nisa.dma_copy(dst=tile, src=t_view)

            # Compute sum over the pool window
            s = nl.sum(tile, axis=[1, 2], keepdims=True)

            # Compute average by multiplying with 1/(p*p)
            # Use nisa.tensor_scalar for scalar operations
            avg = nl.ndarray(s.shape, dtype=s.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=avg, data=s, op0=nl.multiply, operand0=1.0 / (p * p))

            # Copy the average to the output
            nisa.dma_copy(dst=out[:, i, j], src=avg)
    return out