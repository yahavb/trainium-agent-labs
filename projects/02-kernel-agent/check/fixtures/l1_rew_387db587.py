import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Allocate tiles
    x_tile = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    out_tile = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.sbuf)
    
    # Copy input to SBUF
    nisa.dma_copy(dst=x_tile, src=x)
    
    # Compute the average pooling
    # Reshape x_tile to (C, H//p, p, W//p, p)
    # Use affine_range to loop over the tiles
    for h in nl.affine_range(H // p):
        for w in nl.affine_range(W // p):
            tile = x_tile[:, h * p:(h + 1) * p, w * p:(w + 1) * p]
            mean = nl.sum(tile, axis=[1, 2], keepdims=True)
            scaled_mean = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=scaled_mean, data=mean, op0=nl.multiply, operand0=1.0 / (p * p))
            nisa.tensor_copy(dst=out_tile[:, h:h + 1, w:w + 1], src=scaled_mean[:, 0:1, 0:1])
    
    # Copy output to shared HBM
    nisa.dma_copy(dst=out, src=out_tile)
    return out
