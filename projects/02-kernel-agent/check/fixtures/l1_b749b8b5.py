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
    # tile_in and tile_out are now dynamically sized
    # Compute the number of tiles along H and W
    num_h_tiles = (H + p - 1) // p  # Number of tiles along H
    num_w_tiles = (W + p - 1) // p  # Number of tiles along W
    
    # Loop over tiles
    for h_tile in nl.affine_range(num_h_tiles):
        for w_tile in nl.affine_range(num_w_tiles):
            # Compute the slice indices
            h_start = h_tile * p
            h_end = min((h_tile + 1) * p, H)
            w_start = w_tile * p
            w_end = min((w_tile + 1) * p, W)
            
            # Dynamically allocate tile_in with the correct shape
            tile_in = nl.ndarray((C, h_end - h_start, w_end - w_start), dtype=x.dtype, buffer=nl.sbuf)
            # Copy input data to tile
            nisa.dma_copy(dst=tile_in, src=x[:, h_start:h_end, w_start:w_end])
            
            # Compute mean over the pool size
            # Ensure the axis is the last contiguous dimension
            s = nl.sum(tile_in, axis=[1, 2], keepdims=True)
            tile_out = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tile_out, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            
            # Copy result to output
            out[:, h_tile, w_tile] = tile_out
    return out