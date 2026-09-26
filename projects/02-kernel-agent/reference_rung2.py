"""
Reference kernel for rung 2 of the ladder in nkibench.py.

Adapted from the AWS Neuron NKI tutorial example
  transpose2d/transpose2d_nki_kernels.py
Copyright (C) 2024, Amazon.com. All Rights Reserved.

This is the ANSWER, and it is shipped on purpose: the tutorials are public, so hiding it buys
nothing, and a harness whose reference implementation nobody can read is a harness nobody
should trust. Use it to check the harness works, then write your own.

    python nkibench.py --rung 2 --check reference_rung2.py
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nki.typing import tensor


@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
  """
  NKI kernel to reorder the elements on axis[1] of the input tensor.

  Every row of the input tensor is a flattened row-major 2D matrix.
  The shape2D argument defines the dimensions of the flattened matrices (#rows,#cols).
  Our goal in this kernel is to transpose these flattened 2D matrices, i.e. make them (#cols,#rows).

  Example:
      in_tensor = [a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3]
      shape2D = (3,4)
  this means that in_tensor has 3 rows and 4 columns, i.e. can be represented as:
      [a0,a1,a2,a3]
      [b0,b1,b2,b3]
      [c0,c1,c2,c3]
  after transpose, we expect to get:
      [a0,b0,c0]
      [a1,b1,c1]
      [a2,b2,c2]
      [a3,b3,c3]
  Thus, out_tensor is expected to be [a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3]

  Args:
    in_tensor: an input tensor
    shape2D: tuple representing the dimensions to be transposed: (#rows, #cols)
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # Gather input shapes
  sz_p, _ = in_tensor.shape

  # Load input data from external memory to on-chip memory
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)

  # Performing f1/f2 transpose
  # ==========================
  # The desired transpose pattern is provided as an input:
  sz_f1, sz_f2 = shape2D

  # Perform the transposition via element-wise SBUF-to-SBUF copies
  # with index arithmetic to scatter elements into transposed positions.
  # RHS traverses an F1 x F2 matrix in row major order
  # LHS traverses an F2 x F1 (transposed) matrix in row major order
  out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=in_tensor.dtype,
                        buffer=nl.sbuf)
  for i_f1 in nl.affine_range(sz_f1):
    for i_f2 in nl.affine_range(sz_f2):
      nisa.tensor_copy(dst=out_tile[:, nl.ds(i_f2*sz_f1+i_f1, 1)],
                       src=in_tile[:, nl.ds(i_f1*sz_f2+i_f2, 1)])

  # Finally, we store out_tile to external memory
  nisa.dma_copy(dst=out_tensor, src=out_tile)

  return out_tensor
