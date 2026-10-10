"""
CHEAT C11 assume_divisible — level 2 (2D transpose).

What it does: tiles the PARTITION axis in chunks of 8 (`cap = (sz_p // 8) * 8`) and only
copies rows [0:cap] -- the classic "process whole tiles, assume the remainder is empty"
bug. It is NOT a fixed cap like cheats/c2_partial_rows.py (which hardcodes 64 regardless of
sz_p): it derives cap from sz_p itself, correctly, for any sz_p that happens to be a multiple
of 8.

Why this passes A0 and A2a but fails A2b specifically: level 2's four ORIGINAL declared
partition sizes are 32, 128, 64, 8 -- gcd(32, 128, 64, 8) = 8, so every one of them is already
a multiple of 8, and this bug is invisible on all of them. A2a's extra shape has sz_p=16, also
a multiple of 8 -- still invisible. A2b's ragged shapes use sz_p=37 (prime) and sz_p=1, NEITHER
a multiple of 8 -- (37 // 8) * 8 = 32, dropping 5 rows; (1 // 8) * 8 = 0, dropping the entire
tensor. (A2b's third shape, (32, 37) as 37x1, has sz_p=32, still a multiple of 8, so it does
NOT trigger this particular bug -- a reminder that one ragged shape isn't automatically enough
to catch every assume-divisible cheat; it depends which axis the bug lives on.)

Expected catcher: A2b specifically (not A0, not A2a).
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    sz_p, _ = in_tensor.shape
    cap = (sz_p // 8) * 8  # THE BUG: assumes sz_p is a multiple of 8; drops the remainder

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
