"""
Reference kernel for level 7 of the ladder in nkibench.py (M, N and K blocked).

Source: aws-neuron/nki-samples @ 62c300d,
  src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py
  (NKI_EXAMPLE_21). Copyright (C) 2026, Amazon.com. All Rights Reserved.

Verbatim from the tutorial: defaults need M % 2048, N % 1024, K % 1024.

    python nkibench.py --level 7 --check reference_level7_verbatim.py
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
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

  # Compute the block dimensions.
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # Verify the size is a multiple of block size
  assert M % BLOCK_M == 0, \
    f"Expected M {M} to be divisible by {BLOCK_M} when there are {TILES_IN_BLOCK_M}"
  assert N % BLOCK_N == 0, \
    f"Expected N {N} to be divisible by {BLOCK_N} when there are {TILES_IN_BLOCK_N}"
  assert K % BLOCK_K == 0, \
    f"Expected K {K} to be divisible by {BLOCK_K} when there are {TILES_IN_BLOCK_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Compute the number of blocks in each dimension
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    # Create the initial result tiles in SBUF and initialize each tile to
    # 0.0, since the final results will be accumulated here. Results in 3-d array.
    result_tmps = []
    for m_idx in range(NUM_BLOCK_M):
      block_m = []
      for bm_idx in range(TILES_IN_BLOCK_M):
        block_n = []
        for bn_idx in range(TILES_IN_BLOCK_N):
          # Create the result tile (uninitialized)
          tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
          # Initialize the tile 0.0
          nisa.memset(dst=tile, value=0.0)
          # Append the tile to block_n array.
          block_n.append(tile)
        # Append block_n array to block_m array.
        block_m.append(block_n)
      # Append block_m array into result_tmps.
      result_tmps.append(block_m)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler
    # to change this loop by, for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      rhs_tiles = []
      for bk_r in range(TILES_IN_BLOCK_K):
        # Allocate rhs_tile tensor, TILE_K x BLOCK_N
        rhs_tile = nl.ndarray(shape=(TILE_K, BLOCK_N),
                              dtype=rhs.dtype,
                              buffer=nl.sbuf)
        # Copy block tile from rhs, to rhs_tile.
        nisa.dma_copy(dst=rhs_tile[0:TILE_K, 0:BLOCK_N],
                      src=rhs[(TILES_IN_BLOCK_K * k + bk_r) *
                              TILE_K:(TILES_IN_BLOCK_K * k + bk_r + 1) * TILE_K,
                              BLOCK_N * n:BLOCK_N * (n + 1)])
        # Append rhs_tile to rhs_tiles.
        rhs_tiles.append(rhs_tile)


      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        lhsT_tiles = []
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          # Allocate lhsT_tile in SBUF (uninitialized)
          lhsT_tile = nl.ndarray(shape=(TILE_K, BLOCK_M),
                                 dtype=lhsT.dtype,
                                 buffer=nl.sbuf)
          # Copy block tile from lhsT to lhsT_tile
          nisa.dma_copy(dst=lhsT_tile[0:TILE_K, 0:BLOCK_M],
                        src=lhsT[(TILES_IN_BLOCK_K * k + bk_l) *
                                 TILE_K:(TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                                 BLOCK_M * m:BLOCK_M * (m + 1)])
          # Append to list of lhsT tiles.
          lhsT_tiles.append(lhsT_tile)

        # Do matmul with all tiles in the blocks
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            # Allocate result_tile in PSUM (uninitialized)
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                     dtype=nl.float32,
                                     buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              # Perform matrix multiply on a tile.
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M:(bm + 1) * TILE_M],
                moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N:(bn + 1) * TILE_N]
              )
            # Accumulate the result into the result_tmps tile.
            nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                               data1=result_tmps[m][bm][bn],
                               data2=result_tile,
                               op=nl.add)

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        # coalesce result tiles for better DMA performance
        result_packed = nl.ndarray(shape=(TILE_M, BLOCK_N),
                                   dtype=nl.float32,
                                   buffer=nl.sbuf)
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          nisa.tensor_copy(
            dst=result_packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
            src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])

        # Copy packed result from SBUF to HBM.
        nisa.dma_copy(dst=result[(TILES_IN_BLOCK_M * m + bm) *
                                 TILE_M:(TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                 BLOCK_N * n:BLOCK_N * (n + 1)],
                      src=result_packed[0:TILE_M, 0:BLOCK_N])

  return result
