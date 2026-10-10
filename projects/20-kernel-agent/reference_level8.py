"""
Reference kernel for level 8 (single-head attention) of the ladder in nkibench.py.

Written by team 20 by hand -- NOT produced by the agent, and never shown to it. It exists to prove the
level is solvable with this NKI version and to give the checker a ground truth:
    python nkibench.py --level 8 --check reference_level8.py      # 3/3 shapes, traffic at the floor

softmax(q k^T / sqrt(d)) v for q, k, v of shape (seq, dim), everything in one tile (seq <= 128):
transpose q and k so dim is the contraction (partition) axis, nc_matmul the scores into psum, scale,
subtract the row max, exp, divide by the row sum, transpose the probabilities so seq is the contraction
axis, and nc_matmul them with v.
"""
import nki
import nki.language as nl
import nki.isa as nisa
import numpy as np

@nki.jit
def nki_attention_(q, k, v):
    seq, dim = q.shape
    # Allocate buffers
    q_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    k_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    v_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    out_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    
    qT_sb = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    kT_sb = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.sbuf)
    
    qT_ps = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.psum)
    kT_ps = nl.ndarray((dim, seq), dtype=nl.float32, buffer=nl.psum)
    
    scores_ps = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.psum)
    pT_ps = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.psum)
    
    scores_sb = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    shifted = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    exp_scores = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    probabilities = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    pT_sb = nl.ndarray((seq, seq), dtype=nl.float32, buffer=nl.sbuf)
    
    inverse_sum = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    
    out_ps = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.psum)
    out = nl.ndarray((seq, dim), dtype=q.dtype, buffer=nl.shared_hbm)
    
    # Stage 1: DMA copy inputs to sbuf
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)
    nisa.dma_copy(dst=v_sb, src=v)
    
    # Stage 2: transpose Q and K to psum
    nisa.nc_transpose(dst=qT_ps, data=q_sb)
    nisa.tensor_copy(dst=qT_sb, src=qT_ps)
    nisa.nc_transpose(dst=kT_ps, data=k_sb)
    nisa.tensor_copy(dst=kT_sb, src=kT_ps)
    
    # Stage 3: matmul Q_T @ K_T
    nisa.nc_matmul(dst=scores_ps, stationary=qT_sb, moving=kT_sb)
    
    # Stage 4: scale scores by 1/sqrt(dim)
    nisa.tensor_scalar(dst=scores_sb, data=scores_ps, op0=nl.multiply, operand0=1.0 / (dim ** 0.5))
    
    # Stage 5: row max
    row_max = nl.max(scores_sb, axis=[1], keepdims=True)
    
    # Stage 6: subtract row max
    nisa.tensor_scalar(dst=shifted, data=scores_sb, op0=nl.subtract, operand0=row_max)
    
    # Stage 7: exp
    nisa.activation(dst=exp_scores, data=shifted, op=nl.exp)
    
    # Stage 8: row sum
    row_sum = nl.sum(exp_scores, axis=[1], keepdims=True)
    
    # Stage 9: reciprocal
    nisa.reciprocal(dst=inverse_sum, data=row_sum)
    
    # Stage 10: normalize
    nisa.tensor_scalar(dst=probabilities, data=exp_scores, op0=nl.multiply, operand0=inverse_sum)
    
    # Stage 11: transpose probabilities to psum
    nisa.nc_transpose(dst=pT_ps, data=probabilities)
    nisa.tensor_copy(dst=pT_sb, src=pT_ps)
    
    # Stage 12: matmul probabilities @ V
    nisa.nc_matmul(dst=out_ps, stationary=pT_sb, moving=v_sb)
    
    # Stage 13: copy output to sbuf and DMA to HBM
    nisa.tensor_copy(dst=out_sb, src=out_ps)
    nisa.dma_copy(dst=out, src=out_sb)
    return out
