import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_transpose2D_kernel_(x, shape2D):
    P, F = x.shape
    F1, F2 = shape2D
    assert F1 * F2 == F, f"shape2D {shape2D} does not match the free size {F}"

    # Allocate output buffer in shared HBM
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)

    # Allocate a tile to hold the reshaped data
    tile = nl.ndarray((P, F1, F2), dtype=x.dtype, buffer=nl.sbuf)

    # Copy input data to the tile
    nisa.dma_copy(dst=tile, src=x)

    # Reshape and transpose
    for p in nl.affine_range(P):
        for i in nl.affine_range(F1):
            for j in nl.affine_range(F2):
                out[p, j * F1 + i] = tile[p, i, j]

    return out
