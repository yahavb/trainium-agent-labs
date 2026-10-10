"""
CHIPBOOST bandwidth floor: copy x to a new tensor through SBUF, the path RMSNorm's data takes, as fast
as we know how.

RMSNorm must read x once and write y once, so its time can never beat this kernel's time for the same
bytes: this is the "physics floor" line on the dashboard.

When rows divide by 128 the rows are PACKED: partition p takes rows p*R .. p*R+R-1, which sit next to
each other in HBM, so the tensor is one [128, R*dim] view that moves in a few large DMAs. The first
floor (kernels/copy_tiled.py, one 128-row tile per DMA pair) measured 51.6 us for q_norm's 4096 x 128,
slower than 20.5 us for 256 x 4096 with twice the bytes, because 128-column rows are 256 bytes per
partition per DMA. Other row counts fall back to tiles with a partial last one.

    python ../02-kernel-agent/nkibench.py --level 11 --check kernels/copy_floor.py
    python speedcheck.py --op copy --check kernels/copy_floor.py --baseline kernels/copy_tiled.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

CHUNK = 16384   # elements per partition per DMA when packed: 32 KiB of bf16


@nki.jit
def copy_floor(x):
  """out[rows, dim] = x, bit for bit."""
  rows, dim = x.shape
  P = nl.tile_size.pmax  # 128 partitions

  out = nl.ndarray((rows, dim), dtype=x.dtype, buffer=nl.shared_hbm)

  if rows % P == 0:
    width = (rows // P) * dim
    xv = x.reshape((P, width))
    ov = out.reshape((P, width))
    for c0 in range(0, width, CHUNK):
      cw = min(CHUNK, width - c0)
      tile = nl.ndarray((P, cw), dtype=x.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=tile, src=xv[0:P, c0:c0 + cw])
      nisa.dma_copy(dst=ov[0:P, c0:c0 + cw], src=tile)
  else:
    full = rows // P
    for i in range(full):
      tile = nl.ndarray((P, dim), dtype=x.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=tile, src=x[i * P:(i + 1) * P, 0:dim])
      nisa.dma_copy(dst=out[i * P:(i + 1) * P, 0:dim], src=tile)
    tail = rows - full * P
    last = nl.ndarray((tail, dim), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=last, src=x[full * P:rows, 0:dim])
    nisa.dma_copy(dst=out[full * P:rows, 0:dim], src=last)

  return out
