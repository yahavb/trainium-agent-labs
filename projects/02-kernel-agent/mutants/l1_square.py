# MUTANT (deliberately wrong, for testing the held-out eval set): assumes H == W. Every loop shape is square, so it passes them all.
# python nkibench.py --level 1 --eval mutants/l1_square.py
"""
Reference kernel for level 1 of the ladder in nkibench.py.

Adapted from the AWS Neuron NKI tutorial example
  average_pool2d/average_pool2d_nki_kernels.py
Copyright (C) 2024, Amazon.com. All Rights Reserved.

This is the ANSWER, and it is shipped on purpose: the tutorials are public, so hiding it buys
nothing, and a harness whose reference implementation nobody can read is a harness nobody
should trust. Use it to check the harness works, then write your own.

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
  sz_win = sz_hin   # MUTANT: assumes square input
  sz_hout = sz_hin // pool_size
  sz_wout = sz_win // pool_size
  # Create output tensor shared between all SPMD instances as result tensor
  out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Set relevant sizes
  sz_p = sz_cin
  sz_pool = pool_size

  # Generate pool access pattern to create a 5D view:
  # [sz_p, sz_hout, sz_wout, sz_pool, sz_pool]
  # The pool dimensions are placed last so we can reduce over them.

  # Load input data from external memory to on-chip memory
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)

  # Perform the pooling operation using an access pattern view:
  # The .ap() creates a strided 5D view of the 3D input tile,
  # grouping elements into pool windows for reduction.
  pool_view = in_tile.ap([
    [sz_hin * sz_win, sz_p],      # partition stride
    [sz_pool * sz_win, sz_hin // sz_pool],  # outer row stride
    [sz_pool, sz_win // sz_pool],            # outer col stride
    [sz_win, sz_pool],             # inner row stride (within pool window)
    [1, sz_pool],                  # inner col stride (within pool window)
  ])
  sum_tile = nl.sum(pool_view, axis=[3, 4])
  out_tile = nl.ndarray(sum_tile.shape, dtype=sum_tile.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply,
                     operand0=1.0 / (pool_size * pool_size))

  # Store the results back to hbm
  nisa.dma_copy(dst=out_tensor, src=out_tile)

  # Transfer the ownership of `out_tensor` to the caller
  return out_tensor
