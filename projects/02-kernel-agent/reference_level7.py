"""Level 7: block M as well as N and K, so a whole block of lhsT tiles loads in ONE transfer.

Based on reference_level6.py, which builds on reference_level5.py and reference_level4.py,
adapted from the AWS Neuron NKI matrix multiplication tutorial (Copyright Amazon.com, 2024).

WHY THIS LEVEL CANNOT BE ABOUT BYTES. reference_level6.py already measures 1.000x the HBM byte
floor on every registered shape -- each input read once, the output written once. Nothing can
beat a floor, and level 7's registered bar is 1.05x, so a byte bar cannot separate level 7 from
level 6. See validate_level7.py, which measures both and says so.

WHAT IS ACTUALLY LEFT. The remaining cost is the NUMBER of DMA transfers, not their total size.
Level 6 loads its lhsT column one M tile at a time: k_tiles transfers for every M tile. Because
lhsT is (K, M) its M axis is contiguous, so a block of M tiles is ONE rectangular slice and can
move in a single transfer. Blocking M therefore cuts lhsT transfers by a factor of block_m while
moving exactly the same bytes. That is the "full blocking scheme" this level names: K packed
along the free axis of both caches, N blocked to keep rhs resident, and now M blocked too.

The output cannot be merged the same way: an output tile is tile_m = 128 partition rows, and a
block of them would exceed the partition maximum, so the result is still stored one tile at a
time. That asymmetry is the honest end of this ladder.

ON THE CAPACITY BOUND. BUDGET_FLOATS_PER_PARTITION is a conservative self-imposed budget, NOT a
published hardware figure -- this file asserts no SBUF size. The two caches are sized under it in
order, so the kernel degrades to smaller blocks on larger problems instead of failing.
Run `python validate_level7.py` to check numerics, bytes and transfer counts.
"""
import nki
import nki.isa as nisa
import nki.language as nl

BUDGET_FLOATS_PER_PARTITION = 16384


def _largest_divisor_within(count, limit):
    """Largest divisor of `count` that is <= `limit`, so every loop bound divides exactly."""
    best = 1
    for candidate in range(1, count + 1):
        if count % candidate == 0 and candidate <= limit:
            best = candidate
    return best


@nki.jit
def nki_matmul_fully_optimized_(lhsT, rhs):
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

    # Size the rhs block first, then give the lhsT block what is left. Both are chosen under the
    # same budget, which is the search this level is about.
    block_n = _largest_divisor_within(n_tiles, max(1, BUDGET_FLOATS_PER_PARTITION
                                                   // (k_tiles * tile_n)))
    rhs_floats = block_n * k_tiles * tile_n
    remaining = max(k_tiles * tile_m, BUDGET_FLOATS_PER_PARTITION - rhs_floats)
    block_m = _largest_divisor_within(m_tiles, max(1, remaining // (k_tiles * tile_m)))

    n_blocks = n_tiles // block_n
    m_blocks = m_tiles // block_m

    for nb in nl.affine_range(n_blocks):
        # rhs block: block_n slabs, K tiles packed along the free axis, partition stays tile_k.
        rhs_block = nl.ndarray((tile_k, block_n * k_tiles * tile_n),
                               dtype=rhs.dtype, buffer=nl.sbuf)
        for j in nl.affine_range(block_n):
            for k in nl.affine_range(k_tiles):
                col = (j * k_tiles + k) * tile_n
                n0 = (nb * block_n + j) * tile_n
                nisa.dma_copy(
                    dst=rhs_block[:, col:col + tile_n],
                    src=rhs[k * tile_k:(k + 1) * tile_k, n0:n0 + tile_n])

        for mb in nl.affine_range(m_blocks):
            m0 = mb * block_m * tile_m
            span = block_m * tile_m
            # ONE transfer per K tile covers the whole M block, because lhsT's M axis is
            # contiguous. Level 6 needed k_tiles transfers per M tile; this needs k_tiles
            # per M BLOCK, for identical bytes.
            left_block = nl.ndarray((tile_k, k_tiles * span),
                                    dtype=lhsT.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(k_tiles):
                nisa.dma_copy(
                    dst=left_block[:, k * span:(k + 1) * span],
                    src=lhsT[k * tile_k:(k + 1) * tile_k, m0:m0 + span])

            for i in nl.affine_range(block_m):
                for j in nl.affine_range(block_n):
                    accum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(k_tiles):
                        left_col = k * span + i * tile_m
                        rhs_col = (j * k_tiles + k) * tile_n
                        nisa.nc_matmul(
                            dst=accum,
                            stationary=left_block[:, left_col:left_col + tile_m],
                            moving=rhs_block[:, rhs_col:rhs_col + tile_n])
                    out_tile = nl.ndarray((tile_m, tile_n), dtype=lhsT.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=out_tile, src=accum)
                    row = m0 + i * tile_m
                    n0 = (nb * block_n + j) * tile_n
                    nisa.dma_copy(dst=result[row:row + tile_m, n0:n0 + tile_n], src=out_tile)
    return result
