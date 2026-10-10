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

    for i in nl.affine_range(H // p):
        for j in nl.affine_range(W // p):
            # Calculate the tile bounds using min to ensure they don't exceed the tensor dimensions
            tile_start_h = i * p
            tile_end_h = min((i + 1) * p, H)
            tile_start_w = j * p
            tile_end_w = min((j + 1) * p, W)

            # Copy the data into the tile
            tile[0:C, 0:p, 0:p] = t[0:C, tile_start_h:tile_end_h, tile_start_w:tile_end_w]

            # Compute the sum of the current tile
            s = nl.sum(tile, axis=[1, 2], keepdims=True)

            # Store the average in the output
            m = nl.ndarray((C, 1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
            out[0:C, i, j] = m[0:C, 0, 0]

    return out