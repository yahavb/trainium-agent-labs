"""Level 6: block the N axis so each lhsT tile is loaded once, not once per column slab.

Based on the hoisted structure in reference_level5.py, which is in turn based on the tiled
matmul in reference_level4.py, adapted from the AWS Neuron NKI matrix multiplication tutorial
(Copyright Amazon.com, 2024).

WHAT LEVEL 5 LEAVES ON THE TABLE. Level 5 caches one rhs column slab and reuses it across every
M tile, so rhs is read once -- but its lhsT tile is reloaded on every pass of the n loop, i.e.
N // tile_n times. Measured on the registered shapes that is the entire remaining waste: level 5
sits at 1.000x the byte floor when N == tile_n and 1.143x when N == 2 * tile_n.

WHAT THIS KERNEL DOES. It holds `block_n` column slabs resident at once instead of one, and
loads each lhsT tile once per BLOCK rather than once per slab. With the whole N axis resident
the lhsT tile is loaded exactly once, which is the byte floor. That is the level's stated
lesson: spend SBUF capacity to buy reuse, with the block size bounded by capacity rather than
derived from the shape.

ON THE CAPACITY BOUND. `BUDGET_FLOATS_PER_PARTITION` below is a deliberately conservative
self-imposed budget, NOT a published hardware figure -- this file does not assert an SBUF size.
It only has to be small enough to be safe and large enough to hold the registered shapes, and
the block size is derived from it so the kernel degrades to smaller blocks on bigger problems
instead of failing. Run `python validate_level6.py` to check numerics AND the traffic bars.
"""
import nki
import nki.isa as nisa
import nki.language as nl

# Floats per partition we allow ourselves for the resident rhs block. Conservative on purpose:
# the point is that the block size is CHOSEN under a capacity bound, not that this is the
# largest legal value.
BUDGET_FLOATS_PER_PARTITION = 16384


def _block_n(n_tiles, k_tiles, tile_n):
    """Largest divisor of n_tiles whose resident rhs block fits the budget.

    A divisor keeps every loop bound an exact division, so there is no ragged final block. This
    is Python arithmetic on Python ints, evaluated once at trace time.
    """
    per_slab = k_tiles * tile_n
    affordable = max(1, BUDGET_FLOATS_PER_PARTITION // per_slab)
    best = 1
    for candidate in range(1, n_tiles + 1):
        if n_tiles % candidate == 0 and candidate <= affordable:
            best = candidate
    return best


@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
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

    k_tiles = K // tile_k
    n_tiles = N // tile_n
    m_tiles = M // tile_m
    block_n = _block_n(n_tiles, k_tiles, tile_n)
    n_blocks = n_tiles // block_n

    for nb in nl.affine_range(n_blocks):
        # Resident rhs block: block_n slabs, each holding its K tiles packed along the FREE axis
        # so the partition axis stays at tile_k. Each rhs tile is read from HBM exactly once.
        rhs_block = nl.ndarray((tile_k, block_n * k_tiles * tile_n),
                               dtype=rhs.dtype, buffer=nl.sbuf)
        for j in nl.affine_range(block_n):
            for k in nl.affine_range(k_tiles):
                col = (j * k_tiles + k) * tile_n
                n0 = (nb * block_n + j) * tile_n
                nisa.dma_copy(
                    dst=rhs_block[:, col:col + tile_n],
                    src=rhs[k * tile_k:(k + 1) * tile_k, n0:n0 + tile_n])

        for m in nl.affine_range(m_tiles):
            # Load this lhsT column ONCE per block and reuse it for every slab in the block.
            # Level 5 reloaded it per slab; that difference is the whole level.
            left_block = nl.ndarray((tile_k, k_tiles * tile_m),
                                    dtype=lhsT.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(k_tiles):
                nisa.dma_copy(
                    dst=left_block[:, k * tile_m:(k + 1) * tile_m],
                    src=lhsT[k * tile_k:(k + 1) * tile_k, m * tile_m:(m + 1) * tile_m])

            for j in nl.affine_range(block_n):
                accum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(k_tiles):
                    col = (j * k_tiles + k) * tile_n
                    nisa.nc_matmul(
                        dst=accum,
                        stationary=left_block[:, k * tile_m:(k + 1) * tile_m],
                        moving=rhs_block[:, col:col + tile_n])
                out_tile = nl.ndarray((tile_m, tile_n), dtype=lhsT.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_tile, src=accum)
                n0 = (nb * block_n + j) * tile_n
                nisa.dma_copy(dst=result[m * tile_m:(m + 1) * tile_m, n0:n0 + tile_n],
                              src=out_tile)
    return result
