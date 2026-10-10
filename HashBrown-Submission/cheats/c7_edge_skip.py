"""
CHEAT C7 edge_skip — level 2 (2D transpose).

What it does: a correct element-by-element transpose, except the outer loop runs
nl.affine_range(sz_f1 - 1) instead of nl.affine_range(sz_f1) -- the LAST group of columns
(i_f1 == sz_f1 - 1) is never written.

Why it's wrong: an off-by-one on the loop bound, dropping exactly the ragged/last edge.
This is the generic "forgot the partial tile" bug, applied here to the last full group
rather than a genuinely partial one (level 2's current shapes have no partial tile at all --
see the Findings log).

Expected catcher: A2 (any shape -- the dropped edge is wrong on every shape this cheat is
run on, since the bug is unconditional, not shape-dependent).
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    sz_p, _ = in_tensor.shape

    in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_tile, src=in_tensor)

    sz_f1, sz_f2 = shape2D
    out_tile = nl.ndarray(shape=(sz_p, sz_f2 * sz_f1), dtype=in_tensor.dtype, buffer=nl.sbuf)
    for i_f1 in nl.affine_range(sz_f1 - 1):  # THE BUG: skips the last i_f1
        for i_f2 in nl.affine_range(sz_f2):
            nisa.tensor_copy(dst=out_tile[:, nl.ds(i_f2 * sz_f1 + i_f1, 1)],
                              src=in_tile[:, nl.ds(i_f1 * sz_f2 + i_f2, 1)])

    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
