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
    
    # Compute the average over the pool window
    # Reshape to (C, H//p, p, W//p, p)
    # Then compute mean over the last two axes
    # We use tile views and reductions to achieve this
    # First, reshape the tile into the required view
    # Loop over the output dimensions
    for h in nl.affine_range(H // p):
        for w in nl.affine_range(W // p):
            # Define the view for the current window
            view = tile[:, h*p:(h+1)*p, w*p:(w+1)*p]
            # Compute the sum over the window
            s = nl.sum(view, axis=[1, 2], keepdims=True)
            # Compute the average by multiplying by 1/(p*p)
            m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            # Write to output
            nisa.dma_copy(dst=out[:, h:h + 1, w:w + 1], src=m[:, 0:1, 0:1])
    return out