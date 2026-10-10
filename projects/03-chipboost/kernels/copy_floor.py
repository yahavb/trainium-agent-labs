"""
CHIPBOOST bandwidth floor: copy x to a new tensor through SBUF, the path RMSNorm's data takes.

RMSNorm must read x once and write y once. Its time can never beat this kernel's time for the same
bytes, so this is the "physics floor" line on the dashboard: how close an RMSNorm kernel gets to it is
how much of the chip's memory bandwidth it uses.

    python ../02-kernel-agent/nkibench.py --level 11 --check kernels/copy_floor.py

Rows need not be a multiple of 128: the last partial tile is copied with its real row count, so
nothing is read past the end of x.
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
