"""
CHEAT C6 almost_right — level 2 (2D transpose).

What it does: a correct element-by-element transpose, followed by adding a tiny constant
offset (+1e-3) to every element of the result before storing it.

Why it's wrong: a small, uniform numerical bias. On friendly, small-magnitude test data this
can hide inside a loose tolerance; the point of this cheat is to measure whether it does, and
whether hostile (large-magnitude) values expose it by making the SAME absolute offset
relatively tiny instead -- which would make almost_right LESS visible at large values, not
more. That is itself a finding worth recording honestly (the plan's assumption that A3 catches
C6 should be checked, not assumed).

Expected catcher: A3 per the plan (hostile values), but see the Findings log for whether
that expectation held or inverted.
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
            nisa.tensor_copy(dst=out_tile[:, nl.ds(i_f2 * sz_f1 + i_f1, 1)],
                              src=in_tile[:, nl.ds(i_f1 * sz_f2 + i_f2, 1)])

    # THE BUG: a tiny uniform offset on every element.
    nisa.tensor_scalar(dst=out_tile, data=out_tile, op0=nl.add, operand0=1e-3)

    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
