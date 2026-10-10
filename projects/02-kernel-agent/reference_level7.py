"""
Reference kernel for level 7 of the ladder in nkibench.py: matmul with M, N and K blocked.

Written for this harness, following the NKI matmul tutorial's optimisation sequence. Level 6 holds
every K tile of a block on chip, so its SBUF footprint grows with K. This walks K in blocks of up
to 8 tiles as well, so SBUF holds at most 8 x (512 + 1024) elements per partition whatever K is.
The block's output tiles stay in PSUM across all the K blocks -- 4 x 2 tiles of (128, 512) float32
is exactly PSUM's 8 banks -- so partial sums are never written out and read back.

    python nkibench.py --level 7 --check reference_level7.py
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_fully_optimized_(lhsT, rhs):
  """lhsT: [K, M] (left operand, transposed), rhs: [K, N] -> result [M, N].
  K and M are multiples of 128, N a multiple of 512."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512
  TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K = 4, 2, 8
  BLOCK_M, BLOCK_N = TILES_IN_BLOCK_M * TILE_M, TILES_IN_BLOCK_N * TILE_N
  BLOCK_K = TILES_IN_BLOCK_K * TILE_K

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for m0 in range(0, M, BLOCK_M):
    bm = min(BLOCK_M, M - m0) // TILE_M
    for n0 in range(0, N, BLOCK_N):
      bn = min(BLOCK_N, N - n0) // TILE_N
      # One PSUM tile per output tile of the block, accumulating across every K block
      res_psum = nl.ndarray((TILE_M, bm * bn, TILE_N), nl.float32, buffer=nl.psum)

      for k0 in range(0, K, BLOCK_K):
        bk = min(BLOCK_K, K - k0) // TILE_K  # K tiles in this block
        lhsT_block = nl.ndarray((TILE_K, bk, bm * TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_block = nl.ndarray((TILE_K, bk, bn * TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        for k in nl.affine_range(bk):
          nisa.dma_copy(dst=lhsT_block[:, k, :],
                        src=lhsT[k0 + k * TILE_K:k0 + (k + 1) * TILE_K, m0:m0 + bm * TILE_M])
          nisa.dma_copy(dst=rhs_block[:, k, :],
                        src=rhs[k0 + k * TILE_K:k0 + (k + 1) * TILE_K, n0:n0 + bn * TILE_N])
        for mi in nl.affine_range(bm):
          for ni in nl.affine_range(bn):
            for k in nl.affine_range(bk):
              nisa.nc_matmul(dst=res_psum[:, mi * bn + ni, :],
                             stationary=lhsT_block[:, k, mi * TILE_M:(mi + 1) * TILE_M],
                             moving=rhs_block[:, k, ni * TILE_N:(ni + 1) * TILE_N])

      # The block is complete: copy every output tile out once
      for mi in nl.affine_range(bm):
        for ni in nl.affine_range(bn):
          res_sb = nl.ndarray((TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
          nisa.tensor_copy(dst=res_sb, src=res_psum[:, mi * bn + ni, :])
          nisa.dma_copy(dst=result[m0 + mi * TILE_M:m0 + (mi + 1) * TILE_M,
                                   n0 + ni * TILE_N:n0 + (ni + 1) * TILE_N],
                        src=res_sb)

  return result
