"""
CHEAT C8 path_mismatch — level 2 (2D transpose). NOT in the original plan; added because it
fell out directly while reading reference_level2.py for Phase 2.

What it does: the reference's destination index formula is
    out_tile[:, i_f2 * sz_f1 + i_f1] = in_tile[:, i_f1 * sz_f2 + i_f2]
This cheat uses `sz_f2` instead of `sz_f1` in the DESTINATION stride:
    out_tile[:, i_f2 * sz_f2 + i_f1] = in_tile[:, i_f1 * sz_f2 + i_f2]
-- a one-character variable swap, the kind a model confusing "the other dimension" makes.

Why it's wrong, and why it's interesting: the swapped formula is numerically IDENTICAL to the
correct one exactly when sz_f1 == sz_f2 (a square tile), and wrong -- or out of bounds -- the
moment it isn't. Level 2's own existing shape (128, 64) as 8x8 is square (sz_f1 == sz_f2 == 8),
so this cheat is expected to pass on that one existing shape and fail (out-of-bounds or wrong
values) on the other three, which are all non-square. That makes it a live demonstration of
why a single square test case is not enough -- the harness needs at least one non-square shape,
which, unlike C2/C3/C7, level 2 already happens to have today.

Expected catcher: already present shapes that are non-square (not a new augmentation tier).
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
    for i_f1 in nl.affine_range(sz_f1):
        for i_f2 in nl.affine_range(sz_f2):
            # THE BUG: sz_f2 instead of sz_f1 in the destination stride.
            nisa.tensor_copy(dst=out_tile[:, nl.ds(i_f2 * sz_f2 + i_f1, 1)],
                              src=in_tile[:, nl.ds(i_f1 * sz_f2 + i_f2, 1)])

    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
