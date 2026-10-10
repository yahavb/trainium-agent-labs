"""
A level 3 kernel written by the agent (Qwen3-8B) under the reportable checker, copied verbatim
from results/seat-150/checker-l3c.jsonl, run 1, round 5. Used by device_check.py.
"""
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    # Allocate output buffer in shared HBM
    out = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    # Allocate SBUF for lhsT and rhs
    sbuf_lhsT = nl.ndarray(shape=lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray(shape=rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
    # Copy lhsT and rhs from HBM to SBUF
    nisa.dma_copy(dst=sbuf_lhsT, src=lhsT)
    nisa.dma_copy(dst=sbuf_rhs, src=rhs)
    # Allocate PSUM for result
    psum = nl.ndarray(shape=(lhsT.shape[1], rhs.shape[1]), dtype=out.dtype, buffer=nl.psum)
    # Perform matrix multiplication
    nisa.nc_matmul(dst=psum, stationary=sbuf_lhsT, moving=sbuf_rhs)
    # Allocate temporary SBUF for psum result
    sbuf_psum = nl.ndarray(shape=psum.shape, dtype=psum.dtype, buffer=nl.sbuf)
    # Copy result from PSUM to temporary SBUF
    nisa.tensor_copy(dst=sbuf_psum, src=psum)
    # Copy result from temporary SBUF to HBM
    nisa.dma_copy(dst=out, src=sbuf_psum)
    return out
