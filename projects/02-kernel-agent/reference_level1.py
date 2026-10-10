"""
Reference kernel for level 1 of the ladder in nkibench.py.

Adapted from the AWS Neuron NKI tutorial example
  average_pool2d/average_pool2d_nki_kernels.py
Copyright (C) 2024, Amazon.com. All Rights Reserved.

This is the ANSWER, and it is shipped on purpose: the tutorials are public, so hiding it buys
nothing, and a harness whose reference implementation nobody can read is a harness nobody
should trust. Use it to check the harness works, then write your own.

EXTENDED beyond the tutorial, which loads the whole input into one tile. That needs every channel
on one of the 128 partitions and a whole image in SBUF, and level 1 now tests both limits (200
channels; a 240x240 image at 225 KB per partition). So the same pooling runs over chunks of at most
128 channels and bands of output rows sized to fit; within a band it is the tutorial's code.

    python nkibench.py --level 1 --check reference_level1.py
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.typing import tensor


@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
  """NKI kernel to compute a 2D avg-pool operation

  Args:
      in_tensor: an input tensor, of shape C x H x W
      pool_size: an integer representing a (square) pool-window size

  Return:
      out_tensor: the resulting output tensor, of shape C x (H/pool_size) x (W/pool_size)
  """

  # Get input/output dimensions
  sz_cin, sz_hin, sz_win = in_tensor.shape
  sz_hout = sz_hin // pool_size
  sz_wout = sz_win // pool_size
  # Create output tensor shared between all SPMD instances as result tensor
  out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Set relevant sizes
  sz_pool = pool_size
  # Output rows per band: keep a band's input under 16384 elements per partition (64 KB in
  # float32), which leaves room for the band's sums and result even when pool_size is 1.
  sz_band = max(1, min(sz_hout, 16384 // (sz_pool * sz_win)))

  # Channels go on partitions, at most pmax at a time; the last chunk and the last band may be
  # partial, so plain Python loops size them with min().
  for c0 in range(0, sz_cin, nl.tile_size.pmax):
    sz_p = min(nl.tile_size.pmax, sz_cin - c0)
    for r0 in range(0, sz_hout, sz_band):
      sz_rows = min(sz_band, sz_hout - r0)

      # Load this band's input rows from external memory to on-chip memory
      in_tile = nl.ndarray((sz_p, sz_rows * sz_pool, sz_win), dtype=in_tensor.dtype,
                           buffer=nl.sbuf)
      nisa.dma_copy(dst=in_tile,
                    src=in_tensor[c0:c0 + sz_p, r0 * sz_pool:(r0 + sz_rows) * sz_pool, :])

      # Perform the pooling operation using an access pattern view:
      # The .ap() creates a strided 5D view of the 3D input tile,
      # grouping elements into pool windows for reduction.
      pool_view = in_tile.ap([
        [sz_rows * sz_pool * sz_win, sz_p],  # partition stride
        [sz_pool * sz_win, sz_rows],         # outer row stride
        [sz_pool, sz_wout],                  # outer col stride
        [sz_win, sz_pool],                   # inner row stride (within pool window)
        [1, sz_pool],                        # inner col stride (within pool window)
      ])
      sum_tile = nl.sum(pool_view, axis=[3, 4])
      out_tile = nl.ndarray(sum_tile.shape, dtype=sum_tile.dtype, buffer=nl.sbuf)
      nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply,
                         operand0=1.0 / (pool_size * pool_size))

      # Store the band's results back to hbm
      nisa.dma_copy(dst=out_tensor[c0:c0 + sz_p, r0:r0 + sz_rows, :], src=out_tile)

  # Transfer the ownership of `out_tensor` to the caller
  return out_tensor
