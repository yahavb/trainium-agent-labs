"""Best kernel from the v2 agent on level 3 (matmul, single tile), 2026-10-10.

Score 1.00 of 1.0. SOLVED: correct on 1 of 1 shape (checked again with nkibench.py --check).
Run: accepted on its first check, 6 model calls, 103 s.

THIS IS AN ANSWER to level 3: never show it to the kernel agent (no prompt, card or
retrieval source may include it).

Agent: agent2.py at e4edc8c (described index), default settings, Qwen3-32B on
seat-198. Run log: /workspace/scratch/koitu/patchcheck/runs/patched-L1to4/ on seat-198.
Mode: write, approach A1.

This is what the agent wrote, not a reference kernel. Check it with
    python nkibench.py --level 3 --check kernels/level3_best.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    K, M = 128, 64
    K_, N = 128, 512

    # Allocate SBUF tiles for lhsT (K, M) and rhs (K, N)
    lhsT_tile = nl.ndarray((K, M), dtype=lhsT.dtype, buffer=nl.sbuf)
    rhs_tile = nl.ndarray((K, N), dtype=rhs.dtype, buffer=nl.sbuf)

    # Copy input data from HBM to SBUF
    nisa.dma_copy(dst=lhsT_tile, src=lhsT[0:K, 0:M])
    nisa.dma_copy(dst=rhs_tile, src=rhs[0:K, 0:N])

    # Allocate a PSUM tile for the output (M, N)
    psum_tile = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)

    # Compute matmul: lhsT[K, M].T @ rhs[K, N] => psum[M, N]
    nisa.nc_matmul(dst=psum_tile, stationary=lhsT_tile, moving=rhs_tile)

    # Copy result from PSUM to SBUF
    output_tile = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=output_tile, src=psum_tile)

    # Copy result from SBUF to HBM
    output = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=output, src=output_tile)

    return output
