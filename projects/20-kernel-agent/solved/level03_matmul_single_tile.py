import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    # Read input shapes
    K, M = lhsT.shape
    _, N = rhs.shape
    
    # Allocate output buffer in shared HBM
    out = nl.ndarray(shape=(M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    
    # Allocate SBUF for input and output tiles
    sbuf_tile_lhs = nl.ndarray(shape=(K, M), dtype=nl.float32, buffer=nl.sbuf)
    sbuf_tile_rhs = nl.ndarray(shape=(K, N), dtype=nl.float32, buffer=nl.sbuf)
    psum_tile = nl.ndarray(shape=(M, N), dtype=nl.float32, buffer=nl.psum)
    sbuf_tile_out = nl.ndarray(shape=(M, N), dtype=nl.float32, buffer=nl.sbuf)
    
    # Copy lhsT to SBUF
    nisa.dma_copy(dst=sbuf_tile_lhs, src=lhsT)
    # Copy rhs to SBUF
    nisa.dma_copy(dst=sbuf_tile_rhs, src=rhs)
    
    # Perform matrix multiplication
    nisa.nc_matmul(dst=psum_tile, stationary=sbuf_tile_lhs, moving=sbuf_tile_rhs)
    
    # Copy result from PSUM to SBUF
    nisa.tensor_copy(dst=sbuf_tile_out, src=psum_tile)
    
    # Copy result from SBUF to shared HBM
    nisa.dma_copy(dst=out, src=sbuf_tile_out)
    return out
