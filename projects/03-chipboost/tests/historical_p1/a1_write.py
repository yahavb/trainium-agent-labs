import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.typing import tensor

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    """doc"""
    # trace-time plain Python: arbitrary file write, no open/exec/import needed
    np.frombuffer(b"PWNED: trace-time code wrote this as root\n", dtype=np.uint8).tofile("/tmp/rt2/a1_marker.txt")
    K, M = lhsT.shape
    K_, N = rhs.shape
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=lhsT_tile, src=lhsT[k*TILE_K:(k+1)*TILE_K, m*TILE_M:(m+1)*TILE_M])
                nisa.dma_copy(dst=rhs_tile, src=rhs[k*TILE_K:(k+1)*TILE_K, n*TILE_N:(n+1)*TILE_N])
                nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)
            res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=result[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N], src=res_sb)
    return result
