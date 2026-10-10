import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_swiglu_(a, b):
    R, C = a.shape
    tile_size = (128, 512)
    out_shared = nl.ndarray((R, C), dtype=a.dtype, buffer=nl.shared_hbm)

    # Loop over tiles
    for i in range(0, R, tile_size[0]):
        for j in range(0, C, tile_size[1]):
            # Compute the actual tile bounds to avoid out-of-bounds
            end_i = min(i + tile_size[0], R)
            end_j = min(j + tile_size[1], C)
            tile_rows = end_i - i
            tile_cols = end_j - j

            # Allocate tiles with the correct shape
            tile_a = nl.ndarray((tile_rows, tile_cols), dtype=a.dtype, buffer=nl.sbuf)
            tile_b = nl.ndarray((tile_rows, tile_cols), dtype=b.dtype, buffer=nl.sbuf)
            sigmoid_tile = nl.ndarray((tile_rows, tile_cols), dtype=nl.float32, buffer=nl.sbuf)
            out_tile = nl.ndarray((tile_rows, tile_cols), dtype=nl.float32, buffer=nl.psum)

            # Copy a tile
            nisa.dma_copy(dst=tile_a, src=a[i:end_i, j:end_j])
            # Copy b tile
            nisa.dma_copy(dst=tile_b, src=b[i:end_i, j:end_j])

            # Compute sigmoid(a) using activation
            nisa.activation(dst=sigmoid_tile, op=nl.sigmoid, data=tile_a)
            # Multiply sigmoid(a) by a to get silu(a)
            nisa.tensor_tensor(dst=out_tile, data1=sigmoid_tile, data2=tile_a, op=nl.multiply)
            # Multiply silu(a) by b
            nisa.tensor_tensor(dst=out_tile, data1=out_tile, data2=tile_b, op=nl.multiply)

            # Copy result from psum to sbuf
            nisa.tensor_copy(dst=sigmoid_tile, src=out_tile)
            # Copy from sbuf to shared_hbm
            nisa.dma_copy(dst=out_shared[i:end_i, j:end_j], src=sigmoid_tile)
    return out_shared
