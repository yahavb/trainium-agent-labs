"""Best kernel from the v2 agent on level 5 (matmul, loads hoisted), 2026-10-10 (final for this run).

Score 0.62 of 1.0. The checker's message on its first failing shape:
    AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128

Agent: agent2.py at e4edc8c (described index), default settings, Qwen3-32B on
seat-198. Run log: /workspace/scratch/koitu/patchcheck/runs/patched-L5/ on seat-198.
Mode: write, approach A1.

This is what the agent wrote, not a reference kernel. Check it with
    python nkibench.py --level 5 --check kernels/level5_best.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
    # Extract shape info
    K, M = lhsT.shape  # lhsT is [K, M], but it's already transposed (will be stationary)
    K_rhs, N = rhs.shape

    # Allocate output in HBM (destination)
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Allocate SBUF tiles for input data
    lhsT_tile = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)

    # DMA copy inputs from HBM to SBUF
    nisa.dma_copy(dst=lhsT_tile, src=lhsT)
    nisa.dma_copy(dst=rhs_tile, src=rhs)

    # Allocate PSUM tile for matmul output
    psum_tile = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)

    # Compute matmul: dst = stationary.T @ moving
    nisa.nc_matmul(dst=psum_tile, stationary=lhsT_tile, moving=rhs_tile)

    # Copy result from PSUM to SBUF
    sbuf_result = nl.ndarray((M, N), dtype=psum_tile.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sbuf_result, src=psum_tile)

    # DMA copy result from SBUF to HBM
    nisa.dma_copy(dst=out, src=sbuf_result)

    return out
