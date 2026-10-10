"""
CHIPBOOST start kernel for matmul: the level-4 tiled kernel, unchanged except for its name.

Every matmul speedup is measured against this. It is dtype-generic (it allocates with lhsT.dtype and
accumulates in float32 PSUM), so it runs the bf16 Qwen3 shapes as-is.

Adapted from the AWS Neuron NKI tutorial matrix_multiplication/matrix_multiplication_nki_kernels.py
(projects/02-kernel-agent/reference_level4.py). Copyright (C) 2024, Amazon.com. All Rights Reserved.

    python ../02-kernel-agent/nkibench.py --level 9 --check kernels/matmul_start.py

Why it is slow, which is the room the agent has: every (m, n) output tile re-loads its whole row of
lhsT tiles and column of rhs tiles from HBM, so the same bytes cross the bus M/128 and N/512 times.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def qwen3_matmul(lhsT, rhs):
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

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile,
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])

        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)

      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result
