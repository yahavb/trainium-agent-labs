"""Best kernel from the v2 agent on level 2 (2D transpose), 2026-10-10.

Score 0.50 of 1.0. Runs, wrong values: NUMERICAL MISMATCH, 81.8% of elements wrong on the first shape.
Run: not solved: all 4 approaches used up after 27 checks, 49 model calls, 891 s.

Agent: agent2.py at e4edc8c (described index), default settings, Qwen3-32B on
seat-198. Run log: /workspace/scratch/koitu/patchcheck/runs/patched-L1to4/ on seat-198.
Mode: apply, approach A2.

This is what the agent wrote, not a reference kernel. Check it with
    python nkibench.py --level 2 --check kernels/level2_best.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_transpose2D_kernel_(x, shape2D):
    P, F = x.shape
    F1, F2 = shape2D

    # Output in HBM
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    # SBUF tile for input and output
    tile_in = nl.ndarray((P, F1, F2), dtype=x.dtype, buffer=nl.sbuf)
    tile_out = nl.ndarray((P, F2, F1), dtype=x.dtype, buffer=nl.sbuf)

    # Load input from HBM to SBUF
    nisa.dma_copy(dst=tile_in, src=x)

    # Transpose on-chip using tensor_copy
    nisa.tensor_copy(dst=tile_out, src=tile_in)

    # Store the result back to HBM
    nisa.dma_copy(dst=out, src=tile_out.reshape(shape=(P, F)))

    return out
