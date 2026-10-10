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
    for c in nl.affine_range(C):
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                # Extract the tile
                tile = x_tile[c, h*p:(h+1)*p, w*p:(w+1)*p]
                # Compute mean over the pool dimensions
                mean = nl.sum(tile, axis=[1, 2], keepdims=True)
                # Scale by 1/p^2
                scaled_mean = nl.ndarray((1, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=scaled_mean, data=mean, op0=nl.multiply, operand0=1.0 / (p * p))
                # Copy to output tile
                out_tile[c, h, w] = scaled_mean[0, 0, 0]
    
    # Copy output to shared HBM
    nisa.dma_copy(dst=out, src=out_tile)
    return out