"""Condense `neuron-explorer view --output-format summary-text` (stdin) into one line of numbers."""

import sys

K, M, N = map(int, sys.argv[1:4])
d = {}
for line in sys.stdin:
    p = line.split()
    if len(p) == 2:
        try:
            d[p[0]] = float(p[1])
        except ValueError:
            pass
t = d["total_time"]
fl = 2 * K * M * N
print(f"total {t*1e6:9.1f} us  {fl/t/1e12:6.1f} TFLOP/s  mfu {d['mfu_estimated_percent']*100:5.1f}%  "
      f"PE_active {d['tensor_engine_active_time_percent']*100:5.1f}%  dma_active {d['dma_active_time_percent']*100:5.1f}%  "
      f"vec {d['vector_engine_active_time_percent']*100:4.1f}%  act {d['scalar_engine_active_time_percent']*100:4.1f}%  "
      f"hbm_rd {d['hbm_read_bytes']/2**20:.0f}MiB  gpsimd {d['gpsimd_engine_active_time_percent']*100:4.1f}%  hbm_wr {d['hbm_write_bytes']/2**20:.0f}MiB  mm_insts {int(d['matmul_instruction_count'])}")
