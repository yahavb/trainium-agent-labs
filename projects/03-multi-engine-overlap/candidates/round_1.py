import numpy as np

def pipeline_kernel(x, weight, alpha, beta):
    # Overlapped attempt with Prologue & Epilogue, but has a buffer overwrite bug (Pipeline Sync Hazard)
    H, W = x.shape
    _, K = weight.shape
    num_blocks = (H + 127) // 128
    y = np.zeros((H, K), dtype=np.float32)

    # Multi-buffering
    buf_dma = np.zeros((128, W), dtype=np.float32)
    buf_vec = np.zeros((128, W), dtype=np.float32)
    buf_tensor = np.zeros((128, W), dtype=np.float32)
    buf_out = np.zeros((128, K), dtype=np.float32)

    # Prologue: DMA loads Block 0
    if num_blocks > 0:
        v0 = min(128, H)
        buf_dma[:v0, :] = x[0:v0, :]

    # Steady-state loop:
    # BUG: Overwrites buf_dma on Block 1 before finishing Block 0
    for b in range(1, num_blocks):
        r_start = b * 128
        v = min(128, H - r_start)
        # DMA Engine load (corrupts buffer)
        buf_dma[:v, :] = x[r_start:r_start+v, :]
        # Vector Engine scale
        buf_vec[:v, :] = alpha * buf_dma[:v, :] + beta
        # Tensor Engine matmul
        buf_out[:v, :] = buf_vec[:v, :].astype(np.float64) @ weight.astype(np.float64)
        y[r_start:r_start+v, :] = buf_out[:v, :]

    # Epilogue drain remainder
    return y