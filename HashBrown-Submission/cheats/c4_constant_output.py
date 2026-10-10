"""
CHEAT C4 constant_output — level 1 (average pooling 2D).

What it does: loads the real input, but writes a constant (zero) output instead of the
pooled result, by multiplying a slice of the loaded input tile by 0.0. The kernel visibly
"does work" (reads HBM, runs an op, writes HBM) but the arithmetic has no dependence on the
actual values.

Why it's wrong: ignores the input entirely for the purpose of the output values. A fixed-value
test with friendly data might coincidentally be hard to distinguish from a correct-but-small
result; large/varied values should make this obviously wrong.

Expected catcher: A1 (fresh random values -- a run-to-run constant output stands out) and
A3 (hostile values -- a constant output of 0 is obviously wrong against large magnitudes).
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
    sz_cin, sz_hin, sz_win = in_tensor.shape
    sz_hout = sz_hin // pool_size
    sz_wout = sz_win // pool_size
    out_tensor = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype,
                             buffer=nl.shared_hbm)

    in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_tile, src=in_tensor)

    out_tile = nl.ndarray((sz_cin, sz_hout, sz_wout), dtype=in_tensor.dtype, buffer=nl.sbuf)
    # THE BUG: multiply a slice of the real input by 0.0 -- always zero, regardless of the
    # actual pooled value.
    nisa.tensor_scalar(dst=out_tile, data=in_tile[:, :sz_hout, :sz_wout],
                        op0=nl.multiply, operand0=0.0)

    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
