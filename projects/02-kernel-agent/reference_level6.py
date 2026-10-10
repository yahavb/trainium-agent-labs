"""
Reference kernel for level 6 of the ladder in nkibench.py: matmul with M and N blocked.

Written for this harness, following the NKI matmul tutorial's optimisation sequence. Level 5 reuses
one column of rhs tiles; this keeps a whole BLOCK of output tiles' operands resident -- up to 4
tiles of M and 2 of N, a 512 x 1024 piece of the result -- so lhsT is read once and rhs once per
row of blocks. Blocks at the edge of the matrix are smaller, so the block loops are plain Python
ranges that size them with min(). SBUF holds K/128 x (512 + 1024) elements per partition, which is
what level 7 removes by blocking K too.

    python nkibench.py --level 6 --check reference_level6.py
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
  """lhsT: [K, M] (left operand, transposed), rhs: [K, N] -> result [M, N].
  K and M are multiples of 128, N a multiple of 512."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512
  TILES_IN_BLOCK_M, TILES_IN_BLOCK_N = 4, 2
  BLOCK_M, BLOCK_N = TILES_IN_BLOCK_M * TILE_M, TILES_IN_BLOCK_N * TILE_N

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for m0 in range(0, M, BLOCK_M):
    bm = min(BLOCK_M, M - m0) // TILE_M  # M tiles in this block
    # Every K tile of this block of lhsT columns, loaded once for all the N blocks below
    lhsT_block = nl.ndarray((TILE_K, K // TILE_K, bm * TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
    for k in nl.affine_range(K // TILE_K):
      nisa.dma_copy(dst=lhsT_block[:, k, :],
                    src=lhsT[k * TILE_K:(k + 1) * TILE_K, m0:m0 + bm * TILE_M])

    for n0 in range(0, N, BLOCK_N):
      bn = min(BLOCK_N, N - n0) // TILE_N  # N tiles in this block
      rhs_block = nl.ndarray((TILE_K, K // TILE_K, bn * TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
      for k in nl.affine_range(K // TILE_K):
        nisa.dma_copy(dst=rhs_block[:, k, :],
                      src=rhs[k * TILE_K:(k + 1) * TILE_K, n0:n0 + bn * TILE_N])

      # Every output tile of the block, from operands already on chip
      for mi in nl.affine_range(bm):
        for ni in nl.affine_range(bn):
          res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            nisa.nc_matmul(dst=res_psum,
                           stationary=lhsT_block[:, k, mi * TILE_M:(mi + 1) * TILE_M],
                           moving=rhs_block[:, k, ni * TILE_N:(ni + 1) * TILE_N])
          res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
          nisa.tensor_copy(dst=res_sb, src=res_psum)
          nisa.dma_copy(dst=result[m0 + mi * TILE_M:m0 + (mi + 1) * TILE_M,
                                   n0 + ni * TILE_N:n0 + (ni + 1) * TILE_N],
                        src=res_sb)

  return result
