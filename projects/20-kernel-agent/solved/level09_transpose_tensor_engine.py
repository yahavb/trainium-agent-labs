import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_transpose_tile_(x):
    # Allocate output in shared HBM
    out = nl.ndarray(x.shape[::-1], dtype=x.dtype, buffer=nl.shared_hbm)
    # Allocate sbuf tile for input
    x_sb = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    # DMA copy input from HBM to sbuf
    nisa.dma_copy(dst=x_sb, src=x)
    # Allocate psum tile for transpose
    t_ps = nl.ndarray(x.shape[::-1], dtype=x.dtype, buffer=nl.psum)
    # Transpose using Tensor engine
    nisa.nc_transpose(dst=t_ps, data=x_sb)
    # Copy result from psum to sbuf
    t_sb = nl.ndarray(x.shape[::-1], dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t_sb, src=t_ps)
    # DMA copy result from sbuf to shared HBM
    nisa.dma_copy(dst=out, src=t_sb)
    return out
