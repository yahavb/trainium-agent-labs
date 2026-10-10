"""Level 2, HAND-OPTIMIZED by team 20 from the agent's solved/level02_transpose.py (not agent output).

Change: the agent issued F1*F2 single-column tensor_copy calls (12 to 128 per test). One copy through a
strided view of the input tile does the whole permutation y[p, j*F1 + i] = x[p, i*F2 + j].
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def tensor_transpose2D_kernel_(x, shape2D):
    P, F = x.shape
    F1, F2 = shape2D
    out = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.shared_hbm)
    in_tile = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    out_tile = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_tile, src=x)
    # read in (p, j, i) order: partition stride F, then j (stride 1, F2 of them), then i (stride F2, F1 of them)
    view = in_tile.ap([[F, P], [1, F2], [F2, F1]])
    nisa.tensor_copy(dst=out_tile, src=view)
    nisa.dma_copy(dst=out, src=out_tile)
    return out
