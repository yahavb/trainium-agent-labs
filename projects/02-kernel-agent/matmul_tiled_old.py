"""
Tiled matmul written against neuronxcc.nki (the NKI bundled with the compiler), which is the
package in this image that can benchmark on the device. Same algorithm as reference_level4.py;
follows the AWS NKI matmul tutorial's nl.load / nl.matmul / nl.store form.

    C[M, N] = lhsT[K, M].T @ rhs[K, N]
"""

import neuronxcc.nki as nki  # noqa: F401
import neuronxcc.nki.language as nl


def nki_matmul_tiled_(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        rhs_tile = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        res_psum += nl.matmul(lhsT_tile, rhs_tile, transpose_x=True)

      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], value=res_sb)

  return result
