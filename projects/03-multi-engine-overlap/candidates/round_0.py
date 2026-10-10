import numpy as np

def pipeline_kernel(x, weight, alpha, beta):
    # Naive sequential execution without multi-buffering (67% idle stalls)
    H, W = x.shape
    _, K = weight.shape
    num_blocks = (H + 127) // 128
    y = np.zeros((H, K), dtype=np.float32)
    buf_dma = np.zeros((128, W), dtype=np.float32)
    buf_vec = np.zeros((128, W), dtype=np.float32)
    buf_tensor = np.zeros((128, K), dtype=np.float32)

    for b in range(num_blocks):
        r_start = b * 128
        r_end = min(r_start + 128, H)
        valid = r_end - r_start
        # DMA Engine load
        buf_dma[:valid, :] = x[r_start:r_end, :]
        # Vector Engine scale
        buf_vec[:valid, :] = alpha * buf_dma[:valid, :] + beta
        # Tensor Engine matmul
        buf_tensor[:valid, :] = buf_vec[:valid, :].astype(np.float64) @ weight.astype(np.float64)
        y[r_start:r_end, :] = buf_tensor[:valid, :]
    return y