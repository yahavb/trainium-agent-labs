import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    # Allocate SBUF tiles for lhsT and rhs
    lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    # Allocate PSUM tile for result
    psum_tile = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.psum)
    
    # Copy lhsT to SBUF
    nisa.dma_copy(dst=lhs_tile, src=lhsT)
    # Copy rhs to SBUF
    nisa.dma_copy(dst=rhs_tile, src=rhs)
    
    # Perform matrix multiplication
    nisa.nc_matmul(dst=psum_tile, stationary=lhs_tile, moving=rhs_tile)
    
    # Copy result from PSUM to SBUF
    result_tile = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result_tile, src=psum_tile)
    
    # Copy result from SBUF to HBM
    out = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=result_tile)
    
    return out
