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
    # Then compute mean over the last two axes
    # Use tile-based reduction
    tile = nl.ndarray((C, 128, 128), dtype=x.dtype, buffer=nl.sbuf)
    for h in nl.affine_range(H // p):
        for w in nl.affine_range(W // p):
            # Load the pool window
            t_view = t[:, h*p:(h+1)*p, w*p:(w+1)*p]
            # Compute sum over the pool window
            s = nl.sum(t_view, axis=[1, 2], keepdims=True)
            # Compute mean by scaling with 1/(p*p)
            m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            # Store result
            nisa.dma_copy(dst=out[:, h:h + 1, w:w + 1], src=m[:, 0:1, 0:1])
    return out