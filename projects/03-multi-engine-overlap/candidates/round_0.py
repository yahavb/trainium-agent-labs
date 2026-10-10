import numpy as np

def pipeline_kernel(x, weight, alpha, beta) -> np.ndarray:
    H, _ = x.shape
    _, W = weight.shape
    O = H
    buf_dma = np.zeros((128, W), dtype=np.float64)
    buf_vec = np.zeros((128, W), dtype=np.float64)
    buf_out = np.zeros((128, W), dtype=np.float64)
    result = np.zeros((H, W), dtype=np.float64)
    
    for b in range(0, H, 128):
        r_start = b
        r_end = min(r_start + 128, H)
        valid = r_end - r_start
        
        # Prologue: Prefetch Block 0 and Block 1
        if b == 0:
            buf_dma[:valid, :] = x[r_start:r_end, :]
        elif b + 128 < H:
            buf_dma[:valid, :] = x[r_start:r_end, :]
            buf_vec[:valid, :] = alpha * x[b:b+128, :] + beta
            buf_out[:valid, :] = buf_vec[:valid, :].astype(np.float64) @ weight.astype(np.float64)
            result[r_start:r_end, :] = buf_out[:valid, :]
        
        # Steady-State Loop
        if b + 128 < H:
            # DMA Block N+1
            buf_dma[:valid, :] = x[r_start:r_end, :]
            # Vector Block N
            buf_vec[:valid, :] = alpha * buf_dma[:valid, :] + beta
            # Tensor Block N-1
            buf_out[:valid, :] = buf_vec[:valid, :].astype(np.float64) @ weight.astype(np.float64)
            result[r_start:r_end, :] = buf_out[:valid, :]
    
    # Epilogue: Drain remaining Vector and Tensor operations
    if H % 128 != 0:
        buf_vec[:valid, :] = alpha * buf_dma[:valid, :] + beta
        buf_out[:valid, :] = buf_vec[:valid, :].astype(np.float64) @ weight.astype(np.float64)
        result[r_start:r_end, :] = buf_out[:valid, :]
    
    return result
