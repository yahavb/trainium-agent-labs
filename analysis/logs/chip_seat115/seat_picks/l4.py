import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    # Allocate output buffer in shared HBM
    out = nl.ndarray((lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    
    TK = min(128, lhsT.shape[0])
    TM = min(128, lhsT.shape[1])
    TN = min(512, rhs.shape[1])
    
    for m in nl.affine_range(lhsT.shape[1] // TM):
        for n in nl.affine_range(rhs.shape[1] // TN):
            psum_tile = nl.ndarray((TM, TN), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(lhsT.shape[0] // TK):
                lhs_tile = nl.ndarray((TK, TM), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TK, TN), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=lhs_tile, src=lhsT[k * TK:(k + 1) * TK, m * TM:(m + 1) * TM])
                nisa.dma_copy(dst=rhs_tile, src=rhs[k * TK:(k + 1) * TK, n * TN:(n + 1) * TN])
                nisa.nc_matmul(dst=psum_tile, stationary=lhs_tile, moving=rhs_tile)
            res = nl.ndarray((TM, TN), dtype=out.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res, src=psum_tile)
            nisa.dma_copy(dst=out[m * TM:(m + 1) * TM, n * TN:(n + 1) * TN], src=res)
    
    return out
