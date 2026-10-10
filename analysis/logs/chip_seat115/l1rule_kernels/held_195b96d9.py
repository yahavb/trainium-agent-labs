import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=x)
    # Compute the average over the pool windows
    # Reshape to (C, H//p, p, W//p, p)
    # Use tile view with strided access
    # Loop over the output dimensions
    for c in nl.affine_range(C):
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                # Create a view of the pool window
                view = tile[c, h*p:(h+1)*p, w*p:(w+1)*p]
                # Compute mean over the pool window
                mean = nl.sum(view, axis=[1, 2], keepdims=True)
                # Scale by 1/p^2
                scaled_mean = nl.ndarray((1, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=scaled_mean, data=mean, op0=nl.multiply, operand0=1.0 / (p * p))
                # Write to output
                out[c, h, w] = scaled_mean[0, 0, 0]
    return out