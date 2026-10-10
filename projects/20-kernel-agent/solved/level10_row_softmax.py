import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_row_softmax_(x):
    seq, dim = x.shape
    # Allocate SBUF tiles
    x_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    y_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    m_sb = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    e_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    s_sb = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    r_sb = nl.ndarray((seq, 1), dtype=nl.float32, buffer=nl.sbuf)
    out_sb = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.sbuf)
    
    # Copy input to SBUF
    nisa.dma_copy(dst=x_sb, src=x)
    
    # Compute row max
    m = nl.max(x_sb, axis=[1], keepdims=True)
    nisa.tensor_copy(dst=m_sb, src=m)
    
    # Subtract row max
    nisa.tensor_scalar(dst=y_sb, data=x_sb, op0=nl.subtract, operand0=m_sb)
    
    # Compute exp
    nisa.activation(dst=e_sb, data=y_sb, op=nl.exp)
    
    # Compute sum of exps
    s = nl.sum(e_sb, axis=[1], keepdims=True)
    nisa.tensor_copy(dst=s_sb, src=s)
    
    # Compute reciprocal of sum
    nisa.reciprocal(dst=r_sb, data=s_sb)
    
    # Multiply exp by reciprocal of sum
    nisa.tensor_scalar(dst=out_sb, data=e_sb, op0=nl.multiply, operand0=r_sb)
    
    # Copy result to shared HBM
    out = nl.ndarray((seq, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=out_sb)
    return out
