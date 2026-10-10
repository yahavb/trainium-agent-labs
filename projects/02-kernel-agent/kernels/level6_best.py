"""Best kernel from the v2 agent on level 6 (matmul, M and N blocked), 2026-10-10 (the run was stopped by request at ~23:07 UTC; this is its best).

Score 0.62 of 1.0. The checker's message on its first failing shape:
    AssertionError: Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128

Under `nkibench.py --check` the values are right on 2 of 4 shapes: it loops over K and accumulates
in PSUM. But on K=512 M=128 N=512 it moves 1.5x the minimum HBM bytes, over level 6's 1.25x limit,
so the agent counts 1 of 4. M=256 and M=512 fail: it never tiles M (stationary free size > 128).

Agent: agent2.py at e4edc8c (described index), default settings, Qwen3-32B on
seat-198. Run log: /workspace/scratch/koitu/patchcheck/runs/patched-L6/ on seat-198.
Mode: write, approach A2.

This is what the agent wrote, not a reference kernel. Check it with
    python nkibench.py --level 6 --check kernels/level6_best.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
    # Get dimensions
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_

    # Allocate output in HBM
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Define tile shapes
    K_TILE = 128

    # Prepare SBUF tiles for lhsT [K, M], rhs [K, N]
    lhsT_tile = nl.ndarray((K_TILE, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((K_TILE, N), dtype=rhs.dtype, buffer=nl.sbuf)

    # PSUM buffer for result [M, N]
    psum_tile = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)

    # Output tile in SBUF to transfer result to HBM
    out_tile = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.sbuf)

    # Load and process data in tiles
    for k in range(0, K, K_TILE):
        # Copy K_TILE rows of lhsT and rhs to SBUF
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k:k+K_TILE, 0:M])
        nisa.dma_copy(dst=rhs_tile, src=rhs[k:k+K_TILE, 0:N])

        # Perform matrix multiplication: lhsT[K, M].T @ rhs[K, N] = [M, N]
        nisa.nc_matmul(dst=psum_tile, stationary=lhsT_tile, moving=rhs_tile)

        # Copy result from PSUM to SBUF
        nisa.tensor_copy(dst=out_tile, src=psum_tile)

        # Copy result from SBUF to HBM
        nisa.dma_copy(dst=out[0:M, 0:N], src=out_tile)

    return out
