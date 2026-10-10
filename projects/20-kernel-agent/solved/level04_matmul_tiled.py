import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    M = lhsT.shape[1]
    N = rhs.shape[1]
    K = lhsT.shape[0]
    
    # Allocate output buffer in shared HBM
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    
    # Loop over M in steps of 128
    for m in range(0, M, 128):
        # Loop over N in steps of 512
        for n in range(0, N, 512):
            # Allocate psum tile for current (m, n) output tile
            psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            
            # Allocate sbuf tiles for current (m, n) output tile
            sbuf_stationary = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
            sbuf_moving = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
            
            # Loop over K in steps of 128
            for k in range(0, K, 128):
                # Copy lhsT chunk [k, m] to sbuf_stationary
                nisa.dma_copy(dst=sbuf_stationary, src=lhsT[k:k+128, m:m+128])
                
                # Copy rhs chunk [k, n] to sbuf_moving
                nisa.dma_copy(dst=sbuf_moving, src=rhs[k:k+128, n:n+512])
                
                # Perform NC matmul
                nisa.nc_matmul(dst=psum, stationary=sbuf_stationary, moving=sbuf_moving)
            
            # Copy result from psum to sbuf
            sbuf_out = nl.ndarray((128, 512), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_out, src=psum)
            
            # Copy result from sbuf to output buffer
            nisa.dma_copy(dst=out[m:m+128, n:n+512], src=sbuf_out)
    
    return out
