"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

NKI implementation for SPMD tensor addition NKI tutorial.

"""
import numpy as np
# NKI_EXAMPLE_27_BEGIN
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output sizes can be uniformly tiled to [128,512]

  Args:
      a_input: a first input tensor
      b_input: a second input tensor

  Returns:
      c_output: an output tensor
  """
  # Create output tensor shared between all SPMD instances as result tensor
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Calculate tile offsets based on current 'program'
  offset_i_x = nl.program_id(0) * 128
  offset_i_y = nl.program_id(1) * 512

  # Allocate tiles in on-chip memory (SBUF)
  a_tile = nl.ndarray((128, 512), dtype=a_input.dtype, buffer=nl.sbuf)
  b_tile = nl.ndarray((128, 512), dtype=b_input.dtype, buffer=nl.sbuf)
  c_tile = nl.ndarray((128, 512), dtype=a_input.dtype, buffer=nl.sbuf)

  # Load input data from device memory (HBM) to on-chip memory (SBUF)
  nisa.dma_copy(dst=a_tile, src=a_input[offset_i_x:offset_i_x+128, offset_i_y:offset_i_y+512])
  nisa.dma_copy(dst=b_tile, src=b_input[offset_i_x:offset_i_x+128, offset_i_y:offset_i_y+512])

  # compute a + b
  nisa.tensor_tensor(dst=c_tile, op=nl.add, data1=a_tile, data2=b_tile)

  # store the addition results back to device memory (c_output)
  nisa.dma_copy(dst=c_output[offset_i_x:offset_i_x+128, offset_i_y:offset_i_y+512], src=c_tile)

  # Transfer the ownership of `c_output` to the caller
  return c_output
  # NKI_EXAMPLE_27_END


# NKI_EXAMPLE_28_BEGIN
def nki_tensor_add(a_input, b_input):
  """NKI kernel caller to compute element-wise addition of two input tensors

  This kernel caller lifts tile-size restriction, by applying the kernel on tiles of the inputs/outputs

  Args:
      a_input: a first input tensor, of shape [N*128, M*512]
      b_input: a second input tensor, of shape [N*128, M*512]

  Returns:
      a tensor of shape [N*128, M*512], the result of a_input + b_input
  """

  # The SPMD launch grid denotes the number of kernel instances.
  # In this case, we use a 2D grid where the size of each invocation is 128x512
  grid_x = a_input.shape[0] // 128
  grid_y = a_input.shape[1] // 512

  return nki_tensor_add_kernel_[grid_x, grid_y](a_input, b_input)
  # NKI_EXAMPLE_28_END

if __name__ == "__main__":
  a = np.random.rand(256, 1024).astype(np.float16)
  b = np.random.rand(256, 1024).astype(np.float16)

  output_nki = nki_tensor_add(a, b)
  print(f"output_nki={output_nki}")

  output_np = a + b
  print(f"output_np={output_np}")

  allclose = np.allclose(output_np, output_nki, atol=1e-4, rtol=1e-2)
  if allclose:
    print("NKI and NumPy match")
  else:
    print("NKI and NumPy differ")

  assert allclose
