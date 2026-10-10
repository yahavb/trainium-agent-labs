"""
CHEAT C2 partial_rows — level 2 (2D transpose).

What it does: a correct element-by-element transpose (same structure as reference_level2.py),
except every tensor_copy is restricted to partition rows [0:cap] where cap = min(sz_p, 64) --
a fixed cap, not sz_p itself. Rows 64 and beyond are never written, so out_tensor keeps
whatever nl.ndarray handed back for them there.

Why it's wrong: a real kernel must handle every row of the partition axis, derived from the
tensor's own shape, not a hardcoded cap. This mirrors a lazy-agent shortcut: tile only up to a
constant instead of deriving the loop bound from the input.

Expected catcher: A2 (a shape whose partition size exceeds 64). NOTE: one of level 2's
EXISTING shapes, (128, 64) as 8x8, already has sz_p=128 > 64 -- see the Findings log for
whether this cheat is already caught today, before any augmentation.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    sz_p, _ = in_tensor.shape
    cap = min(sz_p, 64)  # THE BUG: a fixed cap, not sz_p

    in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_tile, src=in_tensor)

    sz_f1, sz_f2 = shape2D
    out_tile = nl.ndarray(shape=(sz_p, sz_f2 * sz_f1), dtype=in_tensor.dtype, buffer=nl.sbuf)
    for i_f1 in nl.affine_range(sz_f1):
        for i_f2 in nl.affine_range(sz_f2):
            nisa.tensor_copy(dst=out_tile[0:cap, nl.ds(i_f2 * sz_f1 + i_f1, 1)],
                              src=in_tile[0:cap, nl.ds(i_f1 * sz_f2 + i_f2, 1)])

    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
