"""Level 6, HAND-OPTIMIZED by team 20 from the agent's solved/level06_matmul_blocked.py (not agent output).

Change: no PSUM reset. The agent zeroed PSUM with a tensor_copy before each output tile and let nc_matmul
accumulate onto it -- AWS documents accumulating onto a non-matmul PSUM initialisation as unsupported on
some chips. Here the first K chunk overwrites (accumulate=False) and the rest accumulate.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
    M = lhsT.shape[1]
    N = rhs.shape[1]
    K = lhsT.shape[0]
    
    # Precompute tiles for lhsT and rhs
    lhsT_tiles = []
    for k in range(0, K, 128):
        lhsT_tiles.append([])
        for m in range(0, M, 128):
            tile = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile, src=lhsT[k:k+128, m:m+128])
            lhsT_tiles[-1].append(tile)
    
    rhs_tiles = []
    for k in range(0, K, 128):
        rhs_tiles.append([])
        for n in range(0, N, 512):
            tile = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile, src=rhs[k:k+128, n:n+512])
            rhs_tiles[-1].append(tile)
    
    # Allocate output buffer in shared HBM
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    
    # Allocate psum tile for current (m, n) output tile
    psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    
    # Loop over M in steps of 128
    for m in range(0, M, 128):
        # Loop over N in steps of 512
        for n in range(0, N, 512):
            # Loop over K in steps of 128
            for k in range(0, K, 128):
                # Perform NC matmul
                nisa.nc_matmul(
                    dst=psum,
                    stationary=lhsT_tiles[k // 128][m // 128],
                    moving=rhs_tiles[k // 128][n // 512],
                    accumulate=(k > 0),
                )
            
            # Copy result from psum to sbuf
            sbuf_out = nl.ndarray((128, 512), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_out, src=psum)
            
            # Copy result from sbuf to output buffer
            nisa.dma_copy(dst=out[m:m+128, n:n+512], src=sbuf_out)
    
    return out
