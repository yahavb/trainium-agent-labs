"""Kernel agent2 wrote on level 1 (average pooling 2D), 2026-10-10: SOLVED (1.0).

Agent: agent2.py at e4edc8c (described index), default settings, Qwen3-8B on seat-35 (8K context).
Run: agent2-v5-all-1010-2230 (/workspace/scratch/koitu/agent2-v5/projects/02-kernel-agent/runs/ on
seat-35), 5 checks, 83 s. agent.py's 8K baseline never got past 0.30 on this level.

Caveat: it returns its SBUF result tile instead of writing an HBM output with dma_copy. nki.simulate and
nkibench accept that (HBM bytes 0.80-0.94x the floor show the output write is missing); whether the
device accepts it is untested.

This is what the agent wrote, not a reference kernel. Check it with
    python nkibench.py --level 1 --check kernels/level1_best_8b.py
"""

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    
    # Reshape input to group pooling dimensions
    tile = nl.ndarray(shape=(C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=x)
    
    # Reshape to (C, H//p, p, W//p, p)
    tile = tile.reshape((C, H // p, p, W // p, p))
    
    # Permute to bring the pooling dimensions to the end
    tile = tile.permute((0, 1, 3, 2, 4))
    
    # Compute mean over the last two axes
    out = nl.mean(tile, axis=(3, 4), keepdims=False)
    
    # Return result with correct shape and data type
    return out
