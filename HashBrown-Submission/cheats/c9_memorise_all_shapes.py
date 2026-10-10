"""
CHEAT C9 memorise_all_shapes — level 3 (matmul, single tile).

What it does: correctly computes the matmul ONLY for the exact (K, M, N) triples in
nkibench.LEVELS[3]["shapes"] as of this writing -- {(128, 64, 512), (64, 32, 256)} -- checked
with a plain Python `in` test, not derived from the hardware tile limits. Any other shape
returns zeros instead of raising or computing.

Why it's wrong: this is the "smart" version of cheats/c3_hardcoded_shape.py. That one hardcodes
ONE shape via an `assert ==` and crashes everywhere else. This one has memorized BOTH of level
3's current declared shapes, so it passes A0 in full -- it has learned the test set, not the
shape class (any K<=128, M<=128, N<=512).

Expected catcher: A2 -- either A2a's extra evenly-dividing shape or A2b's ragged one; neither
is in the memorized set, by construction.
"""

import nki
import nki.isa as nisa
import nki.language as nl

MEMORIZED = {(128, 64, 512), (64, 32, 256)}  # level 3's declared shapes, hardcoded


@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    result_sbuf = nl.ndarray((M, N), dtype=result.dtype, buffer=nl.sbuf)

    if (K, M, N) in MEMORIZED:
        lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhs_tile, src=lhsT)
        nisa.dma_copy(dst=rhs_tile, src=rhs)
        result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)
        nisa.tensor_copy(dst=result_sbuf, src=result_psum)
    else:
        # THE BUG: an unrecognized (K, M, N) -- silently return zeros instead of computing.
        nisa.tensor_scalar(dst=result_sbuf, data=result_sbuf, op0=nl.multiply, operand0=0.0)

    nisa.dma_copy(dst=result, src=result_sbuf)
    return result
