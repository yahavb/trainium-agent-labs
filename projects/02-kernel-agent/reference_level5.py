"""
Reference kernel for level 5 of the ladder in nkibench.py: matmul with the loads hoisted.

Written for this harness, following the NKI matmul tutorial's optimisation sequence. It is not the
tutorial's own hoisting kernel: that one keeps M outermost and reloads every rhs tile once per row
of output tiles, which on the largest test shape is 1.86x the byte floor -- over level 5's 1.6x
bar. Putting N outermost hoists the bigger operand instead, so rhs is read exactly once and only
the smaller lhsT is re-read, once per column block: 1.14x on the largest shape.

    python nkibench.py --level 5 --check reference_level5.py
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
  """lhsT: [K, M] (left operand, transposed), rhs: [K, N] -> result [M, N].
  K and M are multiples of 128, N a multiple of 512."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for n in nl.affine_range(N // TILE_N):
    # Hoisted out of the m loop: this column of rhs tiles is loaded once and reused by every m.
    rhs_tiles = nl.ndarray((TILE_K, K // TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
    for k in nl.affine_range(K // TILE_K):
      nisa.dma_copy(dst=rhs_tiles[:, k, :],
                    src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

    for m in nl.affine_range(M // TILE_M):
      # Hoisted out of the k loop: this column of lhsT tiles, loaded before the matmuls start.
      lhsT_tiles = nl.ndarray((TILE_K, K // TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
      for k in nl.affine_range(K // TILE_K):
        nisa.dma_copy(dst=lhsT_tiles[:, k, :],
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])

      # Accumulate partial-sums into PSUM across the whole contraction
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[:, k, :], moving=rhs_tiles[:, k, :])

      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result
