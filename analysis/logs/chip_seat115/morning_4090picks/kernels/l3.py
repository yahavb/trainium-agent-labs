import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    # Allocate SBUF tiles for lhsT and rhs
    lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    out_tile = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.psum)

    # Copy lhsT and rhs into SBUF tiles
    nisa.dma_copy(dst=lhs_tile, src=lhsT)
    nisa.dma_copy(dst=rhs_tile, src=rhs)

    # Perform matrix multiplication into PSUM
    nisa.nc_matmul(dst=out_tile, stationary=lhs_tile, moving=rhs_tile)

    # Copy result from PSUM to SBUF and then to HBM
    result_tile = nl.ndarray(out_tile.shape, dtype=out_tile.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result_tile, src=out_tile)
    out = nl.ndarray(result_tile.shape, dtype=result_tile.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=result_tile)
    return out
