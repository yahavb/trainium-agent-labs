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

    # Allocate two tiles in SBUF
    input_tile = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    output_tile = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)

    # Copy input to input_tile
    nisa.dma_copy(dst=input_tile, src=x)

    # Transpose within the tile
    for i in nl.affine_range(F1):
        for j in nl.affine_range(F2):
            # Copy column i*F2 + j from input_tile to column j*F1 + i in output_tile
            nisa.tensor_copy(
                dst=output_tile[:, j * F1 + i],
                src=input_tile[:, i * F2 + j]
            )

    # Copy result from output_tile to shared_hbm
    nisa.dma_copy(dst=out, src=output_tile)

    return out
