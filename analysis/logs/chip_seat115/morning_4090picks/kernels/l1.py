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
    
    # Compute the average over the pooling window
    # Reshape to (C, H//p, p, W//p, p)
    # Use tile-based reduction
    tile = nl.ndarray((C, p, p), dtype=x.dtype, buffer=nl.sbuf)
    for c in nl.affine_range(C):
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                # Load the pooling window
                window = t[c, h*p:(h+1)*p, w*p:(w+1)*p]
                # Compute sum
                sum_window = nl.sum(window, axis=[1, 2], keepdims=True)
                # Create a tile for the scalar multiplier
                scalar_tile = nl.ndarray(sum_window.shape, dtype=sum_window.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=scalar_tile, data=sum_window, op0=nl.multiply, operand0=1.0 / (p * p))
                # Store result
                out[c, h, w] = scalar_tile
    return out
