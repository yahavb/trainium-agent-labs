"""
A level 3 kernel written by the agent (Qwen3-8B) under the reportable checker, copied verbatim
from results/seat-150/checker-l3c.jsonl, run 2, round 5. Used by device_check.py.
"""
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    # Allocate output buffer in shared HBM with correct shape
    out = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    # Allocate SBUF for the result
    psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=nl.float32, buffer=nl.psum)
    # Allocate SBUF for the stationary (lhsT) and moving (rhs) operands
    sbuf_stationary = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    sbuf_moving = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    
    # Copy lhsT to SBUF
    nisa.dma_copy(dst=sbuf_stationary, src=lhsT)
    # Copy rhs to SBUF
    nisa.dma_copy(dst=sbuf_moving, src=rhs)
    
    # Perform matrix multiplication
    nisa.nc_matmul(dst=psum, stationary=sbuf_stationary, moving=sbuf_moving)
    
    # Allocate temporary SBUF for the result
    temp_sbuf = nl.ndarray(shape=psum.shape, dtype=psum.dtype, buffer=nl.sbuf)
    # Copy result from PSUM to temporary SBUF
    nisa.tensor_copy(dst=temp_sbuf, src=psum)
    
    # Copy result from temporary SBUF to shared HBM
    nisa.dma_copy(dst=out, src=temp_sbuf)
    return out
