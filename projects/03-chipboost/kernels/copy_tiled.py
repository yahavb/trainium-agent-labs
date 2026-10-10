"""
CHIPBOOST copy, one 128-row tile at a time: the first bandwidth floor, kept to measure what packing buys.

On the chip (seat-102, P1's referee): 20.5 us for 256 x 4096 bf16, but 51.6 us for 4096 x 128, half the
bytes, because q_norm's rows are 256 bytes wide and each DMA moves one small row per partition.
kernels/copy_floor.py packs rows instead; time one against the other with
    python speedcheck.py --op copy --check kernels/copy_floor.py --baseline kernels/copy_tiled.py
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def copy_floor(x):
  """out[rows, dim] = x, bit for bit."""
  rows, dim = x.shape
  P = nl.tile_size.pmax  # 128 partitions

  out = nl.ndarray((rows, dim), dtype=x.dtype, buffer=nl.shared_hbm)

  full = rows // P
  if full > 0:
    for i in nl.affine_range(full):
      tile = nl.ndarray((P, dim), dtype=x.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=tile, src=x[i * P:(i + 1) * P, 0:dim])
      nisa.dma_copy(dst=out[i * P:(i + 1) * P, 0:dim], src=tile)

  tail = rows - full * P
  if tail > 0:
    last = nl.ndarray((tail, dim), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=last, src=x[full * P:rows, 0:dim])
    nisa.dma_copy(dst=out[full * P:rows, 0:dim], src=last)

  return out
