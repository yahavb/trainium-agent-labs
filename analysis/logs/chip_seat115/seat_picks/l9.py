import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_softmax_(x):
    R, C = x.shape
    # Allocate output in shared HBM
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Loop over the rows in chunks of 128
    for i in nl.affine_range(0, R, 128):
        # Compute the chunk size
        chunk_size = min(128, R - i)
        # Allocate SBUF tile for data
        tile = nl.ndarray((chunk_size, C), dtype=x.dtype, buffer=nl.sbuf)
        # Copy input chunk to SBUF
        nisa.dma_copy(dst=tile, src=x[i:i+chunk_size, :])
        
        # Compute max along axis 1 (rows) for the chunk
        max_vals = nl.ndarray((chunk_size, 1), dtype=x.dtype, buffer=nl.sbuf)
        m = nl.max(tile, axis=[1], keepdims=True)
        nisa.tensor_copy(dst=max_vals, src=m)
        
        # Subtract max from each row
        sub_tile = nl.ndarray((chunk_size, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sub_tile, data=tile, op0=nl.subtract, operand0=max_vals)
        
        # Compute exp of (x - max)
        exp_tile = nl.ndarray((chunk_size, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_tile, op=nl.exp, data=sub_tile)
        
        # Sum exp along axis 1 (rows)
        sum_vals = nl.ndarray((chunk_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        s = nl.sum(exp_tile, axis=[1], keepdims=True)
        nisa.tensor_copy(dst=sum_vals, src=s)
        
        # Compute 1 / sum
        inv_sum = nl.ndarray((chunk_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_sum, data=sum_vals)
        
        # Multiply exp by 1 / sum
        res_tile = nl.ndarray((chunk_size, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=res_tile, data=exp_tile, op0=nl.multiply, operand0=inv_sum)
        
        # Copy result to output
        nisa.dma_copy(dst=out[i:i+chunk_size, :], src=res_tile)
    
    return out
