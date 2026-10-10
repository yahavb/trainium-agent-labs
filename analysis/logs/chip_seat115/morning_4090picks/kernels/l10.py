import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def nki_layernorm_(x):
    R, C = x.shape
    # Allocate SBUF tiles for input and output
    sbuf_in = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.sbuf)
    sbuf_out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.sbuf)
    # DMA copy input to SBUF in chunks of 128 rows
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_in = x[i:i + chunk_size, :]
        nisa.dma_copy(dst=sbuf_in[i:i + chunk_size, :], src=chunk_in)
    
    # Compute mean over columns (axis=1)
    mean = nl.ndarray((R, 1), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_mean = nl.mean(sbuf_in[i:i + chunk_size, :], axis=[1], keepdims=True)
        nisa.tensor_copy(dst=mean[i:i + chunk_size, :], src=chunk_mean)
    
    # Compute variance over columns (axis=1)
    diff = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_diff = nl.ndarray((chunk_size, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_diff, data1=sbuf_in[i:i + chunk_size, :], data2=mean[i:i + chunk_size, :], op=nl.subtract)
        nisa.tensor_copy(dst=diff[i:i + chunk_size, :], src=chunk_diff)
    
    square = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_square = nl.ndarray((chunk_size, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_square, data1=diff[i:i + chunk_size, :], data2=diff[i:i + chunk_size, :], op=nl.multiply)
        nisa.tensor_copy(dst=square[i:i + chunk_size, :], src=chunk_square)
    
    var = nl.ndarray((R, 1), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_var = nl.mean(square[i:i + chunk_size, :], axis=[1], keepdims=True)
        nisa.tensor_copy(dst=var[i:i + chunk_size, :], src=chunk_var)
    
    # Add epsilon and compute sqrt
    var_plus_eps = nl.ndarray((R, 1), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_var_plus_eps = nl.ndarray((chunk_size, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=chunk_var_plus_eps, data=var[i:i + chunk_size, :], op0=nl.add, operand0=1e-5)
        nisa.tensor_copy(dst=var_plus_eps[i:i + chunk_size, :], src=chunk_var_plus_eps)
    
    sqrt_var = nl.ndarray((R, 1), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_sqrt_var = nl.ndarray((chunk_size, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=chunk_sqrt_var, op=nl.sqrt, data=var_plus_eps[i:i + chunk_size, :])
        nisa.tensor_copy(dst=sqrt_var[i:i + chunk_size, :], src=chunk_sqrt_var)
    
    # Compute reciprocal of sqrt(var)
    inv_sqrt_var = nl.ndarray((R, 1), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_inv_sqrt_var = nl.ndarray((chunk_size, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.reciprocal(dst=chunk_inv_sqrt_var, data=sqrt_var[i:i + chunk_size, :])
        nisa.tensor_copy(dst=inv_sqrt_var[i:i + chunk_size, :], src=chunk_inv_sqrt_var)
    
    # Compute (x - mean) / sqrt(var)
    out_tile = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        chunk_diff = nl.ndarray((chunk_size, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_diff, data1=sbuf_in[i:i + chunk_size, :], data2=mean[i:i + chunk_size, :], op=nl.subtract)
        chunk_out = nl.ndarray((chunk_size, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_out, data1=chunk_diff, data2=inv_sqrt_var[i:i + chunk_size, :], op=nl.multiply)
        nisa.tensor_copy(dst=out_tile[i:i + chunk_size, :], src=chunk_out)
    
    # DMA copy result to shared HBM in chunks of 128 rows
    shared_out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for i in nl.affine_range(0, R, 128):
        chunk_size = min(128, R - i)
        nisa.dma_copy(dst=shared_out[i:i + chunk_size, :], src=out_tile[i:i + chunk_size, :])
    return shared_out
