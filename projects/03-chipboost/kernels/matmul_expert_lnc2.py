"""
CHIPBOOST matmul on BOTH physical NeuronCores of an LNC=2 logical core: kernels/matmul_expert.py (AWS's
SDK 2.32 tutorial kernel, fp32 K-block accumulation, caps fitted to the shape) with the work split
between the two programs of a `kernel[2](...)` launch.

At LNC=2 every logical NeuronCore is two physical ones with one shared HBM stack, and a plain
`kernel(...)` launch is LNC=1: one physical core works, the other idles (NKI 0.6.0 docs: LNC overview,
nki.jit). Launched as `nki_matmul_tiled_[2](lhsT, rhs)`, each program reads nl.program_id(0) (0 or 1)
and computes half of the output into the one nl.shared_hbm result; the halves are disjoint, so no barrier.

Which half: N when its tiles split evenly (at Qwen3's shapes rhs, [K, N], is the big operand, so each
core then reads half of it and all of the small lhsT), else M, else both programs compute everything
(identical values, identical control flow: NKI expects the same control flow on both cores). With no
launch grid (program_ndim() == 0) it is exactly the single-core kernel, so the referee can still check it.

Caps default to the best single-core triple of the first random-search run on seat-102 (m2 n6 k16, 3.33x;
that run is archived, on an older referee, and the sweep ranks the triple #6 of 62); they apply per
program, to that program's share.

Copyright (C) 2024, Amazon.com. All Rights Reserved (the tutorial kernel this adapts).

    CHIPBOOST_CORE=3 python tools/lnc2_probe.py      # simulator, then LNC=2 vs LNC=1 on the chip
"""

import nki
import nki.isa as nisa
import nki.language as nl

TILES_IN_BLOCK_M = 16   # per program, on its share of the output
TILES_IN_BLOCK_N = 6
TILES_IN_BLOCK_K = 16


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

  # LNC=2: which share of the output this program computes. No grid -> the whole output (nkilib's guard).
  ndim = nl.program_ndim()
  n_prgs = nl.num_programs(axes=0) if ndim != 0 else 1
  pid = nl.program_id(axis=0) if ndim != 0 else 0
  if (N // TILE_N) % n_prgs == 0:
    M_SH, N_SH, m_lo, n_lo = M, N // n_prgs, 0, pid * (N // n_prgs)
  elif (M // TILE_M) % n_prgs == 0:
    M_SH, N_SH, m_lo, n_lo = M // n_prgs, N, pid * (M // n_prgs), 0
  else:
    M_SH, N_SH, m_lo, n_lo = M, N, 0, 0

  # Each cap becomes the largest tile count <= the cap that divides this program's tile count.
  TBM = 1
  for t in range(1, min(TILES_IN_BLOCK_M, M_SH // TILE_M) + 1):
    if (M_SH // TILE_M) % t == 0:
      TBM = t
  TBN = 1
  for t in range(1, min(TILES_IN_BLOCK_N, N_SH // TILE_N) + 1):
    if (N_SH // TILE_N) % t == 0:
      TBN = t
  TBK = 1
  for t in range(1, min(TILES_IN_BLOCK_K, K // TILE_K) + 1):
    if (K // TILE_K) % t == 0:
      TBK = t

  BLOCK_M = TILE_M * TBM
  BLOCK_N = TILE_N * TBN
  BLOCK_K = TILE_K * TBK

  # One full-size result in shared HBM; each program writes only its own share.
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  NUM_BLOCK_M = M_SH // BLOCK_M
  NUM_BLOCK_N = N_SH // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    n_start = n_lo + n * BLOCK_N
    n_end = n_start + BLOCK_N

    # One result tile per M-tile, holding its whole N-block contiguously, (TILE_M, TBN, TILE_N), so it
    # can be reshaped to (TILE_M, BLOCK_N) for one large DMA out.
    # CHANGED: float32, not result.dtype, so the K-blocks accumulate without bf16 rounding.
    result_m_tiles = []
    for m in nl.affine_range(NUM_BLOCK_M):
      for m_tile in nl.affine_range(TBM):
        result_m_tile = nl.ndarray(shape=(TILE_M, TBN, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
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
        m_start = m_lo + m * BLOCK_M
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

            # Evict PSUM into the float32 SBUF accumulator for this M-tile.
            m_tile_idx = m_block_tile_start + m_tile
            result_m_tile = result_m_tiles[m_tile_idx]
            nisa.tensor_tensor(
              dst=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
              data1=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
              data2=result_tile,
              op=nl.add,
            )

    # Evict the M-tiles to HBM, one N-block-wide DMA each.
    # CHANGED: the DMA also converts float32 -> the output dtype, the only rounding to bf16.
    for m in nl.affine_range(NUM_BLOCK_M):
      m_block_tile_start = m * TBM
      for m_tile in nl.affine_range(TBM):
        m_tile_idx = m_block_tile_start + m_tile
        result_m_tile = result_m_tiles[m_tile_idx]
        result_m_tile_block = result_m_tile.reshape((TILE_M, BLOCK_N))

        m_tile_start = m_lo + m_tile_idx * TILE_M
        m_tile_end = m_tile_start + TILE_M
        nisa.dma_copy(dst=result[m_tile_start:m_tile_end, n_start:n_end],
                      src=result_m_tile_block[0:TILE_M, 0:BLOCK_N])

  return result
