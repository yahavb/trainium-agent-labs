"""Level 11, HAND-OPTIMIZED by team 20 from the agent's solved/level11_attention_scores.py (not agent output).

Changes: the 1/sqrt(d) scale is applied while moving the result out of PSUM, and the output is stored
once (the agent stored unscaled scores, then overwrote them); the scale is plain Python (the agent's
np.sqrt inside the kernel does not compile on the device).
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_scores_(q, k):
    seq, dim = q.shape
    out = nl.ndarray((seq, seq), dtype=q.dtype, buffer=nl.shared_hbm)
    q_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    k_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)
    qT_ps = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.psum)
    kT_ps = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.psum)
    qT_sb = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    kT_sb = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_transpose(dst=qT_ps, data=q_sb)
    nisa.tensor_copy(dst=qT_sb, src=qT_ps)
    nisa.nc_transpose(dst=kT_ps, data=k_sb)
    nisa.tensor_copy(dst=kT_sb, src=kT_ps)
    s_ps = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=qT_sb, moving=kT_sb)
    s_sb = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=s_sb, data=s_ps, op0=nl.multiply, operand0=1.0 / (dim ** 0.5))   # scale on eviction
    nisa.dma_copy(dst=out, src=s_sb)                                                      # one store
    return out
