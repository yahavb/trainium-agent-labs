"""
CHIPBOOST start kernel for RMSNorm, Qwen3-8B's normalisation: y = x * rsqrt(mean(x^2) + eps) * w.

Plain and correct on purpose. Every RMSNorm speedup is measured against this, and its slowness is the
room the agent has. Where the time goes, cheapest fix first:

  - the weight row is copied to the 128 partitions with 128 separate DMAs, once;
  - the row tiles are a plain Python loop, so tile i+1's load cannot overlap tile i's compute;
  - x^2 is written out to a full float32 tile only to be summed, then x is scaled in float32 and
    multiplied by w in a second pass: three full passes over every tile on the engines.

The floor it is chasing is kernels/copy_floor.py's time for the same bytes: RMSNorm must read x once
and write y once, and nothing can make that faster.

    python ../02-kernel-agent/nkibench.py --level 10 --check kernels/rmsnorm_start.py

x is [rows, D] and rows need not be a multiple of 128: the last tile uses only its real rows, so
nothing is read past the end of x or written past the end of y. w is [1, D]. eps is Qwen3-8B's 1e-6;
do not drop it, quiet rows need it.
"""

import nki
import nki.isa as nisa
import nki.language as nl

EPS = 1e-6   # Qwen3-8B config.json: rms_norm_eps


@nki.jit
def qwen3_rmsnorm(x, w):
  """y[rows, D] = x * rsqrt(mean(x^2, axis=1) + eps) * w, computed in float32, rounded once."""
  rows, dim = x.shape
  assert tuple(w.shape) == (1, dim), f"Expected w of shape (1, {dim}), got {w.shape}"

  P = nl.tile_size.pmax  # 128 partitions: one row of x per partition

  y = nl.ndarray((rows, dim), dtype=x.dtype, buffer=nl.shared_hbm)

  # The weight on every partition, so each row's multiply finds it locally.
  w_tile = nl.ndarray((P, dim), dtype=w.dtype, buffer=nl.sbuf)
  for p in range(P):
    nisa.dma_copy(dst=w_tile[p:p + 1, 0:dim], src=w[0:1, 0:dim])

  for t in range((rows + P - 1) // P):
    r0 = t * P
    n = min(P, rows - r0)   # rows in this tile: P, except possibly the last

    x_tile = nl.ndarray((P, dim), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile[0:n, 0:dim], src=x[r0:r0 + n, 0:dim])

    # Sum of squares per row, in float32: square on the Scalar Engine, reduced along the free axis.
    squares = nl.ndarray((P, dim), dtype=nl.float32, buffer=nl.sbuf)
    sum_sq = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=squares[0:n, 0:dim], op=nl.square, data=x_tile[0:n, 0:dim],
                           reduce_op=nl.add, reduce_res=sum_sq[0:n, 0:1])

    # rstd = rsqrt(sum_sq / dim + eps): one activation, with the mean and eps as its scale and bias.
    rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd[0:n, 0:1], op=nl.rsqrt, data=sum_sq[0:n, 0:1],
                    scale=1.0 / dim, bias=EPS)

    # x * rstd, each row by its own scalar, then * w, rounded to the output dtype once.
    normed = nl.ndarray((P, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=normed[0:n, 0:dim], data=x_tile[0:n, 0:dim],
                       op0=nl.multiply, operand0=rstd[0:n, 0:1])
    out_tile = nl.ndarray((P, dim), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_tile[0:n, 0:dim], data1=normed[0:n, 0:dim],
                       data2=w_tile[0:n, 0:dim], op=nl.multiply)

    nisa.dma_copy(dst=y[r0:r0 + n, 0:dim], src=out_tile[0:n, 0:dim])

  return y
