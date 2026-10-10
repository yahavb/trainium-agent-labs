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
    num_tiles_h = (H + p - 1) // p  # Number of tiles in height
    num_tiles_w = (W + p - 1) // p  # Number of tiles in width
    
    # Loop over tiles
    for i in nl.affine_range(num_tiles_h):
        for j in nl.affine_range(num_tiles_w):
            # Compute the slice indices
            h_start = i * p
            h_end = min((i + 1) * p, H)
            w_start = j * p
            w_end = min((j + 1) * p, W)
            
            # Allocate tile with the exact shape of the slice
            tile_shape = (C, h_end - h_start, w_end - w_start)
            tile = nl.ndarray(tile_shape, dtype=x.dtype, buffer=nl.sbuf)
            
            # Copy data to tile
            nisa.dma_copy(dst=tile, src=x[:, h_start:h_end, w_start:w_end])
            
            # Compute sum over the pool size
            s = nl.sum(tile, axis=[1, 2], keepdims=True)
            
            # Compute mean
            m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            
            # Copy result to output buffer
            nisa.dma_copy(dst=out[:, i, j], src=m)
    
    return out