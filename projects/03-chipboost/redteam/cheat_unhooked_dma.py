"""
RED-TEAM CHEAT: moves data with a DMA the byte counter does not see.

The cheat: imports dma_copy under an alias at module load. nkibench.simulate_and_count counts bytes by
replacing the attribute nisa.dma_copy during simulation, but the alias was bound earlier and still
points at the original, so every rhs load goes uncounted. The kernel is correct; its counted HBM
traffic is too low, which makes its arithmetic intensity look better than it is.

This also defeats a counter that patches dma_transpose / dma_compute too: anything that patches module
attributes misses a name bound at import time.

Should be caught by: the byte-floor check -- counted bytes BELOW minimum_hbm_bytes is impossible for a
correct kernel, so it means an un-hooked transfer (redteam/stage12.py). Note the floor only bites
hard on shapes where the honest kernel is near the floor (the single-tile dev shape is exactly at
it); a counter that hooks below the Python attribute level would be the real fix (P1).

Copied from projects/02-kernel-agent/reference_level4.py, which is adapted from the AWS Neuron
NKI tutorial example matrix_multiplication/matrix_multiplication_nki_kernels.py
Copyright (C) 2024, Amazon.com. All Rights Reserved.

Everything outside the `# >>> CHEAT` ... `# >>> END CHEAT` markers is identical to the reference.
"""

import nki
import nki.isa as nisa
import nki.language as nl
# >>> CHEAT: an alias bound at import time; patching nisa.dma_copy later never reaches it <<<
from nki.isa import dma_copy as _dma
# >>> END CHEAT <<<
import numpy as np
from nki.typing import tensor


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
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
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               m * TILE_M:(m + 1) * TILE_M])
        _dma(dst=rhs_tile,  # CHEAT: was nisa.dma_copy
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result
