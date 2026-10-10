import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

@nki.jit
def nki_attention_scores_(q, k):
    # Allocate output in shared HBM
    out = nl.ndarray(shape=(q.shape[0], k.shape[0]), dtype=nl.float32, buffer=nl.shared_hbm)
    
    # Allocate SBUF for input tiles
    q_sb = nl.ndarray(shape=q.shape, dtype=nl.float32, buffer=nl.sbuf)
    k_sb = nl.ndarray(shape=k.shape, dtype=nl.float32, buffer=nl.sbuf)
    
    # Copy inputs to SBUF
    nisa.dma_copy(dst=q_sb, src=q)
    nisa.dma_copy(dst=k_sb, src=k)
    
    # Transpose q to (dim, seq)
    t_ps = nl.ndarray(shape=(q.shape[1], q.shape[0]), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=t_ps, data=q_sb)
    t_sb = nl.ndarray(shape=(q.shape[1], q.shape[0]), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t_sb, src=t_ps)
    
    # Transpose k to (dim, seq)
    t_ps_k = nl.ndarray(shape=(k.shape[1], k.shape[0]), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=t_ps_k, data=k_sb)
    t_sb_k = nl.ndarray(shape=(k.shape[1], k.shape[0]), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t_sb_k, src=t_ps_k)
    
    # Allocate psum and sbuf tiles for matmul
    psum_tile = nl.ndarray(shape=(q.shape[0], k.shape[0]), dtype=nl.float32, buffer=nl.psum)
    sbuf_tile_out = nl.ndarray(shape=(q.shape[0], k.shape[0]), dtype=nl.float32, buffer=nl.sbuf)
    
    # Perform matrix multiplication
    nisa.nc_matmul(dst=psum_tile, stationary=t_sb, moving=t_sb_k)
    
    # Copy result from PSUM to SBUF
    nisa.tensor_copy(dst=sbuf_tile_out, src=psum_tile)
    
    # Copy result from SBUF to shared HBM
    nisa.dma_copy(dst=out, src=sbuf_tile_out)
    
    # Allocate SBUF for scaling
    scale_sb = nl.ndarray(shape=(q.shape[0], k.shape[0]), dtype=nl.float32, buffer=nl.sbuf)
    
    # Scale by 1/sqrt(d)
    nisa.tensor_scalar(dst=scale_sb, data=sbuf_tile_out, op0=nl.multiply, operand0=1.0 / np.sqrt(q.shape[1]))
    
    # Copy scaled result back to shared HBM
    nisa.dma_copy(dst=out, src=scale_sb)
    return out