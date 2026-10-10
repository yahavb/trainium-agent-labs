"""
CHEAT C1 do_nothing — level 1 (average pooling 2D).

What it does: allocates the output tensor and returns it immediately, without ever writing
to it. No dma_copy to out_tensor ever happens.

Why it's wrong: the output never receives the pooled result. On real hardware this is the
Sakana-style exploit -- a kernel that does nothing can still "pass" if the output memory
happens to already hold the right answer (stale GPU memory reuse). Record what nki.simulate
actually hands back for an unwritten nl.ndarray(buffer=nl.shared_hbm) output in the Findings
log: fresh zeros, garbage, or NaN are all different findings.

Expected catcher: A4 (never-written-output detection).
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
    # THE BUG: nothing is ever written to out_tensor.
    return out_tensor
