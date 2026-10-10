# TEST FIXTURE (written for the L5-7 pipeline check, not by the agent): level 4 with the rhs loads hoisted
# out of the m loop. Traffic about 1.14x the floor on the loop shapes: passes L5 and L6, fails L7.
"""
Reference kernel for level 4 of the ladder in nkibench.py.

Adapted from the AWS Neuron NKI tutorial example
  matrix_multiplication/matrix_multiplication_nki_kernels.py
Copyright (C) 2024, Amazon.com. All Rights Reserved.

This is the ANSWER, and it is shipped on purpose: the tutorials are public, so hiding it buys
nothing, and a harness whose reference implementation nobody can read is a harness nobody
should trust. Use it to check the harness works, then write your own.

    python nkibench.py --level 4 --check reference_level4.py
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.typing import tensor


@nki.jit
def nki_matmul_fully_optimized_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Verify that the input matrices are a multiple of the tile dimensions.
  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Use affine_range to loop over tiles
  for n in nl.affine_range(N // TILE_N):
    # every rhs tile of this column block is read from HBM once, then reused for every m
    rhs_all = nl.ndarray((TILE_K, (K // TILE_K) * TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
    for k in nl.affine_range(K // TILE_K):
      nisa.dma_copy(dst=rhs_all[:, k * TILE_N:(k + 1) * TILE_N],
                    src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
    for m in nl.affine_range(M // TILE_M):
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_all[:, k * TILE_N:(k + 1) * TILE_N])
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)

  return result
