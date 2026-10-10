import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_attention_scores_(q, k):
    # Allocate output in shared HBM
    seq_len = q.shape[0]
    d = q.shape[1]
    out = nl.ndarray(shape=(seq_len, seq_len), dtype=q.dtype, buffer=nl.shared_hbm)
    
    # Allocate sbuf tiles for input
    q_sb = nl.ndarray(shape=(seq_len, d), dtype=q.dtype, buffer=nl.sbuf)
    k_sb = nl.ndarray(shape=(seq_len, d), dtype=k.dtype, buffer=nl.sbuf)
    
    # DMA copy input from HBM to sbuf
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)
    
    # Transpose q and k to (d, seq)
    t_ps_q = nl.ndarray(shape=(d, seq_len), dtype=q.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=t_ps_q, data=q_sb)
    t_sb_q = nl.ndarray(shape=(d, seq_len), dtype=q.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t_sb_q, src=t_ps_q)
    
    t_ps_k = nl.ndarray(shape=(d, seq_len), dtype=k.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=t_ps_k, data=k_sb)
    t_sb_k = nl.ndarray(shape=(d, seq_len), dtype=k.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t_sb_k, src=t_ps_k)
    
    # Allocate psum tile for matmul
    psum_tile = nl.ndarray(shape=(seq_len, seq_len), dtype=q.dtype, buffer=nl.psum)
    
    # Perform matrix multiplication: q^T @ k^T
    nisa.nc_matmul(dst=psum_tile, stationary=t_sb_q, moving=t_sb_k)
    
    # Allocate sbuf tile for output
    out_sb = nl.ndarray(shape=(seq_len, seq_len), dtype=q.dtype, buffer=nl.sbuf)
    
    # Copy result from psum to sbuf
    nisa.tensor_copy(dst=out_sb, src=psum_tile)
    
    # Scale the result by 1 / sqrt(d)
    scale = 1.0 / (d ** 0.5)
    nisa.tensor_scalar(dst=out_sb, data=out_sb, op0=nl.multiply, operand0=scale)
    
    # DMA copy result from sbuf to shared HBM
    nisa.dma_copy(dst=out, src=out_sb)
    return out