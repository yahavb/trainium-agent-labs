"""
CHIPBOOST second baseline: AWS's fully optimised tutorial matmul as published for SDK 2.32
(aws-neuron/aws-neuron-sdk, tag v2.32.0, NKI_EXAMPLE_21), INCLUDING its bf16 accumulation across
K-blocks. The only change is that the block sizes are caps, so it can run at Qwen3's shapes at all.

It exists to measure one thing: what accumulating K-blocks in the output dtype costs in precision at
Qwen3's K=4096 and K=6144, against kernels/matmul_expert.py, which is identical except that it
accumulates in float32. Not a search target: search.py only rewrites matmul_expert.py.

Copyright (C) 2024, Amazon.com. All Rights Reserved (the tutorial kernel this adapts).

    python ../02-kernel-agent/nkibench.py --level 9 --check kernels/matmul_expert_aws.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

TILES_IN_BLOCK_M = 16   # AWS's defaults; search.py does NOT touch this file
TILES_IN_BLOCK_N = 2
TILES_IN_BLOCK_K = 8


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  """result[M, N] = lhsT[K, M].T @ rhs[K, N]. K and M multiples of 128, N a multiple of 512."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  assert M % TILE_M == 0, f"Expected M, {M}, to be a multiple of {TILE_M}"
  assert N % TILE_N == 0, f"Expected N, {N}, to be a multiple of {TILE_N}"
  assert K % TILE_K == 0, f"Expected K, {K}, to be a multiple of {TILE_K}"

  # CHANGED: each cap becomes the largest tile count <= the cap that divides this shape's tile count.
  # Plain Python on shape integers, evaluated once while the kernel is traced.
  TBM = 1
  for t in range(1, min(TILES_IN_BLOCK_M, M // TILE_M) + 1):
    if (M // TILE_M) % t == 0:
      TBM = t
  TBN = 1
  for t in range(1, min(TILES_IN_BLOCK_N, N // TILE_N) + 1):
    if (N // TILE_N) % t == 0:
      TBN = t
  TBK = 1
  for t in range(1, min(TILES_IN_BLOCK_K, K // TILE_K) + 1):
    if (K // TILE_K) % t == 0:
      TBK = t

  BLOCK_M = TILE_M * TBM
  BLOCK_N = TILE_N * TBN
  BLOCK_K = TILE_K * TBK

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    n_start = n * BLOCK_N
    n_end = n_start + BLOCK_N

    # One result tile per M-tile, holding its whole N-block contiguously, (TILE_M, TBN, TILE_N), so it
    # can be reshaped to (TILE_M, BLOCK_N) for one large DMA out.
    # As published: result.dtype, so in bf16 every K-block's partial sum is rounded to bf16.
    result_m_tiles = []
    for m in nl.affine_range(NUM_BLOCK_M):
      for m_tile in nl.affine_range(TBM):
        result_m_tile = nl.ndarray(shape=(TILE_M, TBN, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
        nisa.memset(dst=result_m_tile, value=0.0)
        result_m_tiles.append(result_m_tile)

    # Blocking K dimension (the contraction dimension)
    for k in nl.sequential_range(NUM_BLOCK_K):
      k_block_tile_start = k * TBK

      # Load the RHS K-block one N-block at a time, for large DMAs.
      rhs_tiles = nl.ndarray(shape=(TILE_K, TBK, BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)
      for k_tile in range(TBK):
        k_tile_start = (k_block_tile_start + k_tile) * TILE_K
        k_tile_end = k_tile_start + TILE_K
        nisa.dma_copy(dst=rhs_tiles[0:TILE_K, k_tile, 0:BLOCK_N],
                      src=rhs[k_tile_start:k_tile_end, n_start:n_end])

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        lhsT_tiles = nl.ndarray(shape=(TILE_K, TBK, BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        m_start = m * BLOCK_M
        m_end = m_start + BLOCK_M
        for k_tile in nl.affine_range(TBK):
          k_tile_start = (k_block_tile_start + k_tile) * TILE_K
          k_tile_end = k_tile_start + TILE_K
          nisa.dma_copy(dst=lhsT_tiles[0:TILE_K, k_tile, 0:BLOCK_M],
                        src=lhsT[k_tile_start:k_tile_end, m_start:m_end])

        # Matmul every tile pair in the block
        m_block_tile_start = m * TBM
        for n_tile in nl.affine_range(TBN):
          for m_tile in nl.affine_range(TBM):
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k_tile in nl.affine_range(TBK):
              m_tile_start = m_tile * TILE_M
              m_tile_end = m_tile_start + TILE_M
              n_tile_start = n_tile * TILE_N
              n_tile_end = n_tile_start + TILE_N
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[0:TILE_K, k_tile, m_tile_start:m_tile_end],
                moving=rhs_tiles[0:TILE_K, k_tile, n_tile_start:n_tile_end],
              )

            # Evict PSUM into the SBUF accumulator for this M-tile (rounded to its dtype).
            m_tile_idx = m_block_tile_start + m_tile
            result_m_tile = result_m_tiles[m_tile_idx]
            nisa.tensor_tensor(
              dst=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
              data1=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
              data2=result_tile,
              op=nl.add,
            )

    # Evict the M-tiles to HBM, one N-block-wide DMA each.
    for m in nl.affine_range(NUM_BLOCK_M):
      m_block_tile_start = m * TBM
      for m_tile in nl.affine_range(TBM):
        m_tile_idx = m_block_tile_start + m_tile
        result_m_tile = result_m_tiles[m_tile_idx]
        result_m_tile_block = result_m_tile.reshape((TILE_M, BLOCK_N))

        m_tile_start = m_tile_idx * TILE_M
        m_tile_end = m_tile_start + TILE_M
        nisa.dma_copy(dst=result[m_tile_start:m_tile_end, n_start:n_end],
                      src=result_m_tile_block[0:TILE_M, 0:BLOCK_N])

  return result
