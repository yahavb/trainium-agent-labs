"""The kernel being optimised: a blocked matrix multiply with three block-size knobs.

Adapted from the AWS Neuron NKI tutorial (matrix_multiplication_nki_kernels.py,
nki_matmul_fully_optimized_). Copyright Amazon.com, Inc. or its affiliates.

The three TILES_IN_BLOCK_* defaults below are the starting point: 1, 1, 1 is the unblocked
kernel. The optimiser agent changes these three lines and nothing else.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=1,
    TILES_IN_BLOCK_K=1,
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
    n_start = n * BLOCK_N
    n_end = n_start + BLOCK_N

    # Allocate and initialize result matrix N-block to 0.0.
    #
    # Each result M-tile stores its N-block contiguous on the free-dim
    # with shape (TILE_M, TILES_IN_BLOCK_N, TILE_N). This layout allows
    # reshaping to (TILE_M, BLOCK_N) for SBUF->HBM DMA to operate on a
    # large payload, enabling good DMA efficiency.
    #
    # We split the N-block into individual M-tiles so the compiler can
    # pipeline memset(0), matmul, tensor_tensor, and SBUF->HBM DMA
    # on M-tile granularity.
    result_m_tiles = []
    for m in nl.affine_range(NUM_BLOCK_M):
      for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
        result_m_tile = nl.ndarray(
          shape=(TILE_M, TILES_IN_BLOCK_N, TILE_N),
          dtype=result.dtype,
          buffer=nl.sbuf,
        )
        nisa.memset(dst=result_m_tile, value=0.0)
        result_m_tiles.append(result_m_tile)

    # Blocking K dimension (the contraction dimension)
    for k in nl.sequential_range(NUM_BLOCK_K):
      k_block_tile_start = k * TILES_IN_BLOCK_K

      # Load tiles from RHS
      # Load tiles one N-block at a time for good DMA efficiency.
      rhs_tiles = nl.ndarray(
        shape=(TILE_K, TILES_IN_BLOCK_K, BLOCK_N),
        dtype=rhs.dtype,
        buffer=nl.sbuf,
      )
      for k_tile in range(TILES_IN_BLOCK_K):
        k_tile_start = (k_block_tile_start + k_tile) * TILE_K
        k_tile_end = k_tile_start + TILE_K
        nisa.dma_copy(
          dst=rhs_tiles[0:TILE_K, k_tile, 0:BLOCK_N],
          src=rhs[k_tile_start:k_tile_end, n_start:n_end],
        )

      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        # Load tiles one M-block at a time for good DMA efficiency.
        lhsT_tiles = nl.ndarray(
          shape=(TILE_K, TILES_IN_BLOCK_K, BLOCK_M),
          dtype=lhsT.dtype,
          buffer=nl.sbuf,
        )
        m_start = m * BLOCK_M
        m_end = m_start + BLOCK_M
        for k_tile in nl.affine_range(TILES_IN_BLOCK_K):
          k_tile_start = (k_block_tile_start + k_tile) * TILE_K
          k_tile_end = k_tile_start + TILE_K
          nisa.dma_copy(
            dst=lhsT_tiles[0:TILE_K, k_tile, 0:BLOCK_M],
            src=lhsT[k_tile_start:k_tile_end, m_start:m_end],
          )

        # Do matmul with all tiles in the blocks
        m_block_tile_start = m * TILES_IN_BLOCK_M
        for n_tile in nl.affine_range(TILES_IN_BLOCK_N):
          for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
            result_tile = nl.ndarray(
              shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum
            )
            for k_tile in nl.affine_range(TILES_IN_BLOCK_K):
              m_tile_start = m_tile * TILE_M
              m_tile_end = m_tile_start + TILE_M
              n_tile_start = n_tile * TILE_N
              n_tile_end = n_tile_start + TILE_N
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[0:TILE_K, k_tile, m_tile_start:m_tile_end],
                moving=rhs_tiles[0:TILE_K, k_tile, n_tile_start:n_tile_end],
              )

            # Evict from PSUM to SBUF while accumulating into result M-tile.
            m_tile_idx = m_block_tile_start + m_tile
            result_m_tile = result_m_tiles[m_tile_idx]
            nisa.tensor_tensor(
              dst=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
              data1=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
              data2=result_tile,
              op=nl.add,
            )

    # Evict the result M-tiles from SBUF to HBM.
    # Copy on N-blocks granularity for good DMA efficiency.
    for m in nl.affine_range(NUM_BLOCK_M):
      m_block_tile_start = m * TILES_IN_BLOCK_M
      for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
        m_tile_idx = m_block_tile_start + m_tile
        result_m_tile = result_m_tiles[m_tile_idx]
        result_m_tile_block = result_m_tile.reshape((TILE_M, BLOCK_N))

        m_tile_start = m_tile_idx * TILE_M
        m_tile_end = m_tile_start + TILE_M
        nisa.dma_copy(
          dst=result[m_tile_start:m_tile_end, n_start:n_end],
          src=result_m_tile_block[0:TILE_M, 0:BLOCK_N],
        )

  return result
