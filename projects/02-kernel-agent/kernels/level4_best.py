"""Best kernel from the v2 agent on level 4 (matmul, tiled), 2026-10-10.

Score 0.62 of 1.0. Correct on 1 of 4 shapes; then: dma_copy dst partition dimension 256 exceeds maximum 128.
Run: best after 2 checks; the run was stopped by request during level 4.

Agent: agent2.py at e4edc8c (described index), default settings, Qwen3-32B on
seat-198. Run log: /workspace/scratch/koitu/patchcheck/runs/patched-L1to4/ on seat-198.
Mode: write, approach A2.

This is what the agent wrote, not a reference kernel. Check it with
    python nkibench.py --level 4 --check kernels/level4_best.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    K, M = lhsT.shape
    K_rhs, N = rhs.shape
    assert K == K_rhs

    # Output shape is [M, N]
    output = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Allocate on-chip tiles for inputs and output
    lhs_tile = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_tile = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    output_tile = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.sbuf)

    # Copy lhsT and rhs from HBM to SBUF
    nisa.dma_copy(dst=lhs_tile, src=lhsT)
    nisa.dma_copy(dst=rhs_tile, src=rhs)

    # Perform tiled matrix multiplication: dst = stationary.T @ moving
    nisa.nc_matmul(dst=psum_tile, stationary=lhs_tile, moving=rhs_tile)

    # Copy result from PSUM to SBUF, casting to the output dtype
    nisa.tensor_copy(dst=output_tile, src=psum_tile)

    # Copy result from SBUF to HBM
    nisa.dma_copy(dst=output, src=output_tile)

    return output
