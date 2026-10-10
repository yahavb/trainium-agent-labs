
### seq=128 dim=64
| version | change | bench mean us | profiled total us | busy us (any engine) | Tensor us | Vector us | Scalar us | GpSimd us | DMA us | sw-DGE DMA us | static DMA us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | zero baseline (reconstructed) | 17.75 | 22.32 | 16.91 | 4.65 | 4.19 | 4.46 | 5.39 | 5.75 | 1.25 | 4.50 |
| V1 | V0 rescheduled | 17.78 | 22.45 | 16.71 | 4.65 | 3.82 | 5.09 | 5.38 | 5.59 | 1.19 | 4.40 |
| V2 bf16 | bf16 P@V | 17.13 | 21.81 | 16.29 | 4.14 | 3.85 | 4.93 | 5.61 | 5.45 | 1.25 | 4.21 |
| V3 hwdge | hw DMA descriptors | 16.06 | 22.95 | 16.15 | 4.44 | 3.98 | 5.48 | 3.01 | 5.74 | 0.00 | 4.39 |
| V4 hwdge+bf16 | both | 15.62 | 22.77 | 15.81 | 3.81 | 4.07 | 5.49 | 3.25 | 5.78 | 0.00 | 4.42 |

### seq=64 dim=128
| version | change | bench mean us | profiled total us | busy us (any engine) | Tensor us | Vector us | Scalar us | GpSimd us | DMA us | sw-DGE DMA us | static DMA us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | zero baseline (reconstructed) | 17.23 | 21.62 | 15.90 | 4.06 | 3.93 | 4.79 | 5.36 | 6.18 | 1.37 | 4.81 |
| V1 | V0 rescheduled | 17.07 | 21.82 | 16.62 | 4.06 | 3.61 | 4.74 | 5.36 | 5.91 | 1.41 | 4.50 |
| V2 bf16 | bf16 P@V | 16.82 | 21.72 | 15.65 | 3.63 | 3.77 | 4.77 | 5.59 | 5.79 | 1.50 | 4.29 |
| V3 hwdge | hw DMA descriptors | 15.31 | 22.17 | 15.32 | 3.81 | 3.91 | 5.61 | 3.24 | 5.43 | 0.00 | 4.12 |
| V4 hwdge+bf16 | both | 14.92 | 21.79 | 15.09 | 3.52 | 4.06 | 5.59 | 3.46 | 5.26 | 0.00 | 3.98 |

### seq=96 dim=32
| version | change | bench mean us | profiled total us | busy us (any engine) | Tensor us | Vector us | Scalar us | GpSimd us | DMA us | sw-DGE DMA us | static DMA us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | zero baseline (reconstructed) | 16.95 | 21.63 | 16.29 | 4.10 | 4.13 | 4.40 | 5.29 | 5.48 | 1.03 | 4.45 |
| V1 | V0 rescheduled | 16.84 | 21.57 | 15.89 | 4.09 | 3.87 | 4.61 | 5.28 | 5.54 | 1.20 | 4.34 |
| V2 bf16 | bf16 P@V | 16.50 | 21.37 | 15.42 | 3.84 | 3.75 | 5.17 | 5.57 | 5.45 | 1.22 | 4.23 |
| V3 hwdge | hw DMA descriptors | 15.18 | 22.20 | 15.54 | 3.98 | 3.46 | 5.37 | 2.88 | 5.79 | 0.00 | 4.75 |
| V4 hwdge+bf16 | both | 15.12 | 22.22 | 14.65 | 3.63 | 3.76 | 5.39 | 3.39 | 5.16 | 0.00 | 4.09 |

### fields that differ, V0 -> V4 hwdge+bf16, case 0
  activate_instruction_time: 1.066e-06 -> 1.132e-06
  adjusted_hardware_flops: 3.355e+07 -> 2.307e+07
  adjusted_transpose_flops: 1.678e+07 -> 1.258e+07
  dma_active_cycles: 6895 -> 6937
  dma_active_time: 5.746e-06 -> 5.781e-06
  dma_active_time_percent: 0.2574 -> 0.2539
  dma_packet_time: 1.782e-05 -> 1.76e-05
  dma_queue_count: 18 -> 34
  dma_transfer_time: 1.246e-06 -> 1.645e-06
  dynamic_dma_active_time_percent: 0.05582 -> 0.05988
  dynamic_dma_packet_percent: 0.9209 -> 0.9863
  dynamic_dma_size_percent: 0.5643 -> 0.6602
  event_count: 368 -> 360
  gpsimd_engine_active_time: 5.387e-06 -> 3.25e-06
  gpsimd_engine_active_time_percent: 0.2413 -> 0.1427
  gpsimd_engine_instruction_count: 32 -> 28
  gpsimd_engine_instruction_time: 5.482e-06 -> 3.777e-06
  hardware_dynamic_dma_active_time: 0 -> 1.364e-06
  hardware_dynamic_dma_active_time_percent: 0 -> 0.05988
  hardware_dynamic_dma_packet_count: 0 -> 576
  hardware_dynamic_dma_packet_percent: 0 -> 0.9863
  hardware_dynamic_dma_size: 0 -> 1.313e+05
  hardware_dynamic_dma_size_percent: 0 -> 0.6602
  hardware_flops: 8.389e+06 -> 1.049e+07
  hfu_estimated_percent: 0.01912 -> 0.01288
  matmul_instruction_count: 7 -> 6
  mbu_estimated_percent: 0.008201 -> 0.008039
  mbu_min_read_util_percent: 0.006151 -> 0.006029
  mfu_estimated_percent: 0.009558 -> 0.005855
  mfu_inst_estimated_percent: 0.009558 -> 0.005855
  neuroncore_cycle_count: 2.678e+04 -> 2.733e+04
  psum_read_bytes: 2.949e+05 -> 2.621e+05
  psum_read_sbuf_write_bytes: 2.299e+05 -> 1.644e+05
  psum_write_bytes: 3.277e+05 -> 2.949e+05
  sbuf_read_bytes: 8.202e+05 -> 6.067e+05
  sbuf_write_bytes: 4.608e+05 -> 4.762e+05
  scalar_engine_active_time: 4.462e-06 -> 5.491e-06
  scalar_engine_active_time_percent: 0.1999 -> 0.2411
  scalar_engine_instruction_count: 29 -> 30
  scalar_engine_instruction_time: 4.86e-06 -> 5.99e-06
  software_dynamic_dma_active_time: 1.246e-06 -> 0
  software_dynamic_dma_active_time_percent: 0.05582 -> 0
  software_dynamic_dma_packet_count: 128 -> 0
  software_dynamic_dma_packet_percent: 0.9209 -> 0
  software_dynamic_dma_size: 1.313e+05 -> 0
  software_dynamic_dma_size_percent: 0.5643 -> 0
  static_dma_active_time: 4.5e-06 -> 4.418e-06
  static_dma_active_time_percent: 0.2016 -> 0.194
  static_dma_packet_count: 11 -> 8
  static_dma_packet_percent: 0.07914 -> 0.0137
  static_dma_size: 1.014e+05 -> 6.76e+04
  static_dma_size_percent: 0.4357 -> 0.3398
  sync_engine_active_time: 2.031e-06 -> 4.127e-06
  sync_engine_active_time_percent: 0.09098 -> 0.1812
  sync_engine_instruction_count: 21 -> 25
  sync_engine_instruction_time: 2.031e-06 -> 4.127e-06
  tensor_engine_active_time: 4.655e-06 -> 3.813e-06
  tensor_engine_active_time_percent: 0.2085 -> 0.1675
  tensor_engine_instruction_count: 37 -> 36
  tensor_engine_instruction_time: 7.413e-06 -> 5.462e-06
  total_active_time: 1.691e-05 -> 1.581e-05
  total_active_time_percent: 0.7575 -> 0.6941
  total_exec_time: 1.007e-05 -> 8.463e-06
  total_time: 2.232e-05 -> 2.277e-05
  trace_count: 662 -> 1099
  transpose_flops: 4.194e+06 -> 6.291e+06
  vector_engine_active_time: 4.194e-06 -> 4.068e-06
  vector_engine_active_time_percent: 0.1879 -> 0.1786
  vector_engine_instruction_time: 4.621e-06 -> 4.399e-06
