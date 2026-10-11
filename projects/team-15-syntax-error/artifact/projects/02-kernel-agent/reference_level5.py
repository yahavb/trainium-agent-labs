"""
Reference kernel for level 5 of the ladder in nkibench.py (loads hoisted).

Source: aws-neuron/nki-samples @ 62c300d,
  src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py
  (NKI_EXAMPLE_19). Copyright (C) 2026, Amazon.com. All Rights Reserved.

Verbatim from the tutorial.

    python nkibench.py --level 5 --check reference_level5.py
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner
     while hoisting the load of the lhsT and rhs to outer loops.

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

  # Verify that the lhsT and rhs are the expected sizes.
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
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_M numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = []
    for k in nl.affine_range(K // TILE_K):
      # Allocate space in SBUF for the tile (uninitialized)
      lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
      # Copy the tile from HBM to SBUF
      nisa.dma_copy(dst=lhsT_tile, 
                    src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                             m * TILE_M:(m + 1) * TILE_M])
      # Append the tile to the list of tiles.
      lhsT_tiles.append(lhsT_tile)

    for n in nl.affine_range(N // TILE_N):
      # Load a whole column tiles from rhs (with K * TILE_N numbers)
      rhs_tiles = []
      for k in nl.affine_range(K // TILE_K):
        # Allocate space in SBUF for the tile (uninitialized)
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=rhs_tile,
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])
        # Append the tile to the list of tiles.
        rhs_tiles.append(rhs_tile)

      # Allocate a tile in PSUM for the result (uninitialized)
      res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rhs_tiles[k])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result
