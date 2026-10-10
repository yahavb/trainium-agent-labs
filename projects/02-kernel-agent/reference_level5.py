"""Level 5: cache each RHS column slab across all output-row tiles.

Based on the tiled matmul structure in reference_level4.py, adapted from the
AWS Neuron NKI matrix multiplication tutorial (Copyright Amazon.com, 2024).

Supports the same positive, tile-aligned shapes as that reference, subject to
SBUF capacity: the cached RHS slab uses K * TILE_N elements. The registered
level-5 shapes need at most 1 MiB for that slab in float32.

Run `python validate_level5.py` to check numerics AND the level-5 traffic bar.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
    K, M = lhsT.shape
    K_rhs, N = rhs.shape
    tile_k = nl.tile_size.pmax
    tile_m = nl.tile_size.gemm_stationary_fmax
    tile_n = nl.tile_size.gemm_moving_fmax
    assert K == K_rhs, "contraction dimensions must match"
    assert K > 0 and K % tile_k == 0, "K must be a positive multiple of tile_k"
    assert M > 0 and M % tile_m == 0, "M must be a positive multiple of tile_m"
    assert N > 0 and N % tile_n == 0, "N must be a positive multiple of tile_n"

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for n in nl.affine_range(N // tile_n):
        # Pack the K tiles along the free axis so the partition axis stays 128.
        # Each RHS tile is read once, then reused by every M tile in this slab.
        rhs_cache = nl.ndarray((tile_k, (K // tile_k) * tile_n),
                               dtype=rhs.dtype, buffer=nl.sbuf)
        for k in nl.affine_range(K // tile_k):
            nisa.dma_copy(
                dst=rhs_cache[:, k * tile_n:(k + 1) * tile_n],
                src=rhs[k * tile_k:(k + 1) * tile_k,
                        n * tile_n:(n + 1) * tile_n])

        for m in nl.affine_range(M // tile_m):
            accum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // tile_k):
                left = nl.ndarray((tile_k, tile_m), dtype=lhsT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=left,
                    src=lhsT[k * tile_k:(k + 1) * tile_k,
                             m * tile_m:(m + 1) * tile_m])
                nisa.nc_matmul(
                    dst=accum, stationary=left,
                    moving=rhs_cache[:, k * tile_n:(k + 1) * tile_n])

            output_tile = nl.ndarray((tile_m, tile_n), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=output_tile, src=accum)
            nisa.dma_copy(
                dst=result[m * tile_m:(m + 1) * tile_m,
                           n * tile_n:(n + 1) * tile_n],
                src=output_tile)
    return result
