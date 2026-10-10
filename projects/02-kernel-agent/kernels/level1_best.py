"""Best kernel from the v2 agent on level 1 (average pooling 2D), 2026-10-10.

Score 1.00 of 1.0. SOLVED: correct on 4 of 4 shapes (checked again with nkibench.py --check).
Run: accepted after 5 checks, 11 model calls, 171 s.

THIS IS AN ANSWER to level 1: never show it to the kernel agent (no prompt, card or
retrieval source may include it).

Agent: agent2.py at e4edc8c (described index), default settings, Qwen3-32B on
seat-198. Run log: /workspace/scratch/koitu/patchcheck/runs/patched-L1to4/ on seat-198.
Mode: apply, approach A2.

This is what the agent wrote, not a reference kernel. Check it with
    python nkibench.py --level 1 --check kernels/level1_best.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    # Ensure H and W are divisible by p
    H = H // p * p
    W = W // p * p

    # Step 1: DMA copy a tile of [C, H, W] input data into SBUF
    tile = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=x)

    # Step 2: Reshape into [C, H/p, p, W/p, p]
    tile_reshaped = tile.reshape((C, H // p, p, W // p, p))

    # Step 2.5: Permute to make the pooling dimensions the last axes: [C, H/p, W/p, p, p]
    tile_permuted = tile_reshaped.permute((0, 1, 3, 2, 4))

    # Step 3: Compute mean over axes (3, 4) (pooling dimensions)
    mean_tile = nl.mean(tile_permuted, axis=(3, 4), keepdims=False)

    # Step 4: DMA copy the result tile [C, H/p, W/p] to HBM output
    out = nl.ndarray((C, H // p, W // p), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=mean_tile)

    return out
