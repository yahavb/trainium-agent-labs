"""
A level 3 kernel written by the agent (Qwen3-8B) under the reportable checker, copied verbatim
from results/seat-150/checker-l3c.jsonl, run 3, round 5. Used by device_check.py.
"""
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    # Allocate output buffer in shared HBM
    out = nl.ndarray(shape=(64, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    # Allocate SBUF for input and output tiles
    sbuf_tile = nl.ndarray(shape=(128, 64), dtype=nl.float32, buffer=nl.sbuf)
    psum_tile = nl.ndarray(shape=(64, 512), dtype=nl.float32, buffer=nl.psum)
    # Allocate separate SBUF tile for rhs and result
    sbuf_rhs = nl.ndarray(shape=(128, 512), dtype=nl.float32, buffer=nl.sbuf)
    sbuf_result = nl.ndarray(shape=(64, 512), dtype=nl.float32, buffer=nl.sbuf)
    
    # Copy lhsT to SBUF
    nisa.dma_copy(dst=sbuf_tile, src=lhsT)
    # Copy rhs to separate SBUF tile
    nisa.dma_copy(dst=sbuf_rhs, src=rhs)
    
    # Perform matrix multiplication
    nisa.nc_matmul(dst=psum_tile, stationary=sbuf_tile, moving=sbuf_rhs)
    
    # Copy result from PSUM to separate SBUF
    nisa.tensor_copy(dst=sbuf_result, src=psum_tile)
    
    # Copy result from SBUF to shared HBM
    nisa.dma_copy(dst=out, src=sbuf_result)
    return out
