"""
CHEAT C3 hardcoded_shape — level 3 (matmul, single tile).

What it does: this is a near-verbatim copy of the SHIPPED reference kernel,
reference_level3.py. It is included here deliberately, not written from scratch, because the
shipped reference itself hardcodes the shape via `assert K == 128`, `assert M == 64`,
`assert N == 512` -- it is correct only because level 3 declares exactly ONE test shape,
K=128 M=64 N=512, so nothing in the harness today can tell "handles any single-tile matmul"
apart from "handles exactly this one, by assertion."

Why it's wrong (as a GENERAL kernel, not as level 3's reference): a real single-tile matmul
kernel should work for any K<=128, M<=128, N<=512, not just one hardcoded triple. Shipping
asserts that happen to match the one test case is indistinguishable, under this harness, from
genuinely handling the tile-size limits.

Expected catcher: NONE exist today -- see the Findings log. Level 3 would need a SECOND
declared shape (still inside the single-tile class: K<=128, M<=128, N<=512) before any
augmentation tier could expose this. This is a gap in the shape *list*, not something A1-A5
can fix on their own for this level.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "contraction dimension must match"
    assert K == 128, f"expected K=128, got {K}"   # THE BUG: hardcoded, not derived or general
    assert M == 64, f"expected M=64, got {M}"      # THE BUG
    assert N == 512, f"expected N=512, got {N}"    # THE BUG

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=lhs_tile, src=lhsT)
    nisa.dma_copy(dst=rhs_tile, src=rhs)

    result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)

    result_sbuf = nl.ndarray(result_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    nisa.dma_copy(dst=result, src=result_sbuf)
    return result
