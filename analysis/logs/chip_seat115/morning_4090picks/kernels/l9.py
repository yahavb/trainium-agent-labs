import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_softmax_(x):
    R, C = x.shape
    # Allocate SBUF tiles for input and output
    sbuf_input = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
    sbuf_output = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
    
    # Allocate output buffer in HBM
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Loop over the rows in chunks of 128
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        
        # Copy input chunk to SBUF
        nisa.dma_copy(dst=sbuf_input[0:chunk_size, :], src=x[i:i+chunk_size, :])
        
        # Compute max along axis 1 (rows) for the chunk
        max_vals = nl.max(sbuf_input[0:chunk_size, :], axis=[1], keepdims=True)
        
        # Subtract max from input
        sub_tile = nl.ndarray((chunk_size, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sub_tile, data=sbuf_input[0:chunk_size, :], op0=nl.subtract, operand0=max_vals)
        
        # Apply exp
        exp_tile = nl.ndarray((chunk_size, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=sub_tile)
        
        # Sum exp along axis 1 (rows)
        sum_vals = nl.sum(exp_tile, axis=[1], keepdims=True)
        
        # Compute reciprocal of sum_vals
        inv = nl.ndarray(sum_vals.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=sum_vals)
        
        # Multiply exp_tile by reciprocal of sum_vals
        div_tile = nl.ndarray((chunk_size, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=div_tile, data=exp_tile, op0=nl.multiply, operand0=inv)
        
        # Copy result to output
        nisa.dma_copy(dst=out[i:i+chunk_size, :], src=div_tile)
    
    return out
