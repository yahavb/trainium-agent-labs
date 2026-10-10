
### seq=128 dim=64
| version | change | bench mean us | profiled total us | busy us (any engine) | Tensor us | Vector us | Scalar us | GpSimd us | DMA us | sw-DGE DMA us | static DMA us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | baseline | 17.79 | 21.99 | 16.62 | 4.64 | 3.62 | 5.10 | 5.36 | 6.30 | 1.41 | 4.89 |
| V2 bf16 | bf16 P@V | 17.14 | 21.14 | 16.32 | 4.01 | 3.72 | 4.94 | 5.63 | 6.75 | 1.22 | 5.53 |
| V3 hwdge | hw DMA descriptors | 16.17 | 23.10 | 16.10 | 4.44 | 4.01 | 5.47 | 3.03 | 5.86 | 0.00 | 4.35 |
| V4 hwdge+bf16 | both | 15.75 | 22.66 | 15.64 | 3.87 | 3.68 | 5.47 | 3.26 | 5.36 | 0.00 | 3.87 |

### seq=64 dim=128
| version | change | bench mean us | profiled total us | busy us (any engine) | Tensor us | Vector us | Scalar us | GpSimd us | DMA us | sw-DGE DMA us | static DMA us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | baseline | 17.07 | 21.88 | 16.73 | 4.07 | 3.64 | 4.74 | 5.33 | 5.62 | 1.31 | 4.30 |
| V2 bf16 | bf16 P@V | 16.82 | 21.35 | 16.21 | 3.63 | 3.74 | 4.75 | 5.60 | 5.77 | 1.36 | 4.41 |
| V3 hwdge | hw DMA descriptors | 15.32 | 21.91 | 14.86 | 3.95 | 3.72 | 5.49 | 3.06 | 5.30 | 0.00 | 3.96 |
| V4 hwdge+bf16 | both | 14.91 | 21.82 | 15.20 | 3.38 | 3.80 | 5.50 | 3.23 | 5.80 | 0.00 | 4.55 |

### seq=96 dim=32
| version | change | bench mean us | profiled total us | busy us (any engine) | Tensor us | Vector us | Scalar us | GpSimd us | DMA us | sw-DGE DMA us | static DMA us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | baseline | 16.83 | 21.59 | 16.25 | 4.09 | 3.43 | 4.83 | 5.27 | 5.90 | 1.18 | 4.72 |
| V2 bf16 | bf16 P@V | 16.50 | 21.10 | 15.87 | 3.75 | 3.81 | 4.83 | 5.53 | 5.34 | 1.03 | 4.31 |
| V3 hwdge | hw DMA descriptors | 15.18 | 21.85 | 14.70 | 3.99 | 3.88 | 5.37 | 3.24 | 5.03 | 0.00 | 3.96 |
| V4 hwdge+bf16 | both | 15.14 | 21.69 | 14.57 | 3.57 | 3.90 | 5.34 | 3.27 | 5.07 | 0.00 | 3.99 |

### fields that differ, V1 -> V4, case 0
  activate_instruction_time: 1.134e-06 -> 1.134e-06
  adjusted_hardware_flops: 3.355e+07 -> 2.307e+07
  adjusted_transpose_flops: 1.678e+07 -> 1.258e+07
  dma_active_cycles: 7562 -> 6433
  dma_active_time: 6.302e-06 -> 5.361e-06
  dma_active_time_percent: 0.2865 -> 0.2366
  dma_packet_time: 1.987e-05 -> 1.949e-05
  dma_queue_count: 18 -> 34
  dma_transfer_time: 1.435e-06 -> 1.749e-06
  dynamic_dma_active_time_percent: 0.06408 -> 0.06574
  dynamic_dma_packet_percent: 0.9014 -> 0.9813
  dynamic_dma_size_percent: 0.4928 -> 0.5643
  event_count: 367 -> 360
  gpsimd_engine_active_time: 5.363e-06 -> 3.256e-06
  gpsimd_engine_active_time_percent: 0.2438 -> 0.1437
  gpsimd_engine_instruction_count: 32 -> 28
  gpsimd_engine_instruction_time: 5.459e-06 -> 3.788e-06
  hardware_dynamic_dma_active_time: 0 -> 1.489e-06
  hardware_dynamic_dma_active_time_percent: 0 -> 0.06574
  hardware_dynamic_dma_packet_count: 0 -> 576
  hardware_dynamic_dma_packet_percent: 0 -> 0.9813
  hardware_dynamic_dma_size: 0 -> 1.313e+05
  hardware_dynamic_dma_size_percent: 0 -> 0.5643
  hardware_flops: 8.389e+06 -> 1.049e+07
  hfu_estimated_percent: 0.0194 -> 0.01295
  matmul_instruction_count: 7 -> 6
  mbu_estimated_percent: 0.008323 -> 0.00808
  mbu_min_read_util_percent: 0.006242 -> 0.00606
  mfu_estimated_percent: 0.009699 -> 0.005885
  mfu_inst_estimated_percent: 0.009699 -> 0.005885
  neuroncore_cycle_count: 2.639e+04 -> 2.719e+04
  psum_read_bytes: 2.949e+05 -> 2.621e+05
  psum_read_sbuf_write_bytes: 2.299e+05 -> 1.644e+05
  psum_write_bytes: 3.277e+05 -> 2.949e+05
  sbuf_read_bytes: 7.542e+05 -> 6.067e+05
  sbuf_write_bytes: 4.598e+05 -> 4.762e+05
  scalar_engine_active_time: 5.1e-06 -> 5.475e-06
  scalar_engine_active_time_percent: 0.2319 -> 0.2417
  scalar_engine_instruction_count: 29 -> 30
  scalar_engine_instruction_time: 5.601e-06 -> 5.977e-06
  software_dynamic_dma_active_time: 1.409e-06 -> 0
  software_dynamic_dma_active_time_percent: 0.06408 -> 0
  software_dynamic_dma_packet_count: 128 -> 0
  software_dynamic_dma_packet_percent: 0.9014 -> 0
  software_dynamic_dma_size: 1.313e+05 -> 0
  software_dynamic_dma_size_percent: 0.4928 -> 0
  static_dma_active_time: 4.893e-06 -> 3.872e-06
  static_dma_active_time_percent: 0.2225 -> 0.1709
  static_dma_packet_count: 14 -> 11
  static_dma_packet_percent: 0.09859 -> 0.01874
  static_dma_size: 1.352e+05 -> 1.014e+05
  static_dma_size_percent: 0.5072 -> 0.4357
  sync_engine_active_time: 2.136e-06 -> 4.131e-06
  sync_engine_active_time_percent: 0.09711 -> 0.1823
  sync_engine_instruction_count: 21 -> 25
  sync_engine_instruction_time: 2.136e-06 -> 4.131e-06
  tensor_engine_active_time: 4.645e-06 -> 3.871e-06
  tensor_engine_active_time_percent: 0.2112 -> 0.1708
  tensor_engine_instruction_count: 38 -> 36
  tensor_engine_instruction_time: 7.36e-06 -> 5.442e-06
  total_active_time: 1.662e-05 -> 1.564e-05
  total_active_time_percent: 0.7554 -> 0.6904
  total_exec_time: 1.007e-05 -> 8.499e-06
  total_time: 2.199e-05 -> 2.266e-05
  trace_count: 663 -> 1102
  transpose_flops: 4.194e+06 -> 6.291e+06
  vector_engine_active_time: 3.619e-06 -> 3.679e-06
  vector_engine_active_time_percent: 0.1645 -> 0.1624
  vector_engine_instruction_count: 30 -> 32
  vector_engine_instruction_time: 3.896e-06 -> 4.013e-06
