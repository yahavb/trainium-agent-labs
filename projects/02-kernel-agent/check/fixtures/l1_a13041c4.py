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
    # Use tile.ap to create a strided view for reduction
    # Loop over each tile in the output
    # For each output position, compute the mean of the pool window
    for c in nl.affine_range(C):
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                # Create a view of the pool window
                view = tile[c, h*p:(h+1)*p, w*p:(w+1)*p]
                # Compute the sum over the pool window
                s = nl.sum(view, axis=[1, 2], keepdims=True)
                # Compute the average by multiplying by 1/(p*p)
                m = nl.ndarray((1, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
                # Copy to output
                out[c, h, w] = m[0, 0, 0]
    return out