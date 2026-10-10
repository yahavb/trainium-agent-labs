"""
probe_resident_layout.py — HUMAN-WRITTEN calibration probe (NOT an agent success).

Purpose: test whether the installed NKI (0.6.0 + simulator) supports the resident-layout
strategy before the model spends rounds on it: keep BOTH inputs on chip, load each input tile
exactly once, and slice the resident buffers as nc_matmul operands.

Provenance: written by the team. Excluded from agent success counts and from the population.
Uses the level-5 entry name so the standard checker can measure it; the measurement is what
matters, not the gate.

Layout (2-D form of the plan's proposed [128, K_tiles, M] rectangle):
    lhs_cache [128, K//128 * M]   each 128-row k-chunk of lhsT is stacked along the free dim
    rhs_cache [128, K//128 * N]   same for rhs
    stationary view: lhs_cache[:, kk*M + m0 : kk*M + m0 + 128]      -> [128, 128]
    moving view:     rhs_cache[:, kk*N + n0 : kk*N + n0 + 512]      -> [128, 512]

Expected accounting if the simulator accepts it: bytes == the byte floor on every shape,
i.e. 1.00x — each input element crosses HBM exactly once and the output is written once.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
    """Resident-input tiled matmul. lhsT [K, M], rhs [K, N] -> [M, N], float32."""
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    k_tiles = K // TILE_K

    # Both inputs, resident on chip. One tile loaded once; nothing re-read.
    lhs_cache = nl.ndarray((TILE_K, k_tiles * M), dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_cache = nl.ndarray((TILE_K, k_tiles * N), dtype=rhs.dtype, buffer=nl.sbuf)
    for kk in nl.affine_range(k_tiles):
        nisa.dma_copy(dst=lhs_cache[:, kk * M:(kk + 1) * M],
                      src=lhsT[kk * TILE_K:(kk + 1) * TILE_K, :])
        nisa.dma_copy(dst=rhs_cache[:, kk * N:(kk + 1) * N],
                      src=rhs[kk * TILE_K:(kk + 1) * TILE_K, :])

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m0 in nl.affine_range(M // TILE_M):
        for n0 in nl.affine_range(N // TILE_N):
            res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for kk in nl.affine_range(k_tiles):
                nisa.nc_matmul(
                    dst=res_psum,
                    stationary=lhs_cache[:, kk * M + m0 * TILE_M:kk * M + (m0 + 1) * TILE_M],
                    moving=rhs_cache[:, kk * N + n0 * TILE_N:kk * N + (n0 + 1) * TILE_N])
            res_sb = nl.ndarray((TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=result[m0 * TILE_M:(m0 + 1) * TILE_M,
                                     n0 * TILE_N:(n0 + 1) * TILE_N],
                          src=res_sb)

    return result
