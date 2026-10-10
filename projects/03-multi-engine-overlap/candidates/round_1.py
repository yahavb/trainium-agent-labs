import numpy as np

def pipeline_kernel(x: np.ndarray, weight: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    H, W = x.shape
    _, K = weight.shape
    TILE_H = 128
    num_blocks = (H + TILE_H - 1) // TILE_H
    out = np.zeros((H, K), dtype=np.float32)

    def get_valid(b_idx: int) -> int:
        r_start = b_idx * TILE_H
        return min(TILE_H, H - r_start)

    # Handle single block case
    if num_blocks == 1:
        v0 = get_valid(0)
        buf_dma = x[0:v0, :].copy()
        buf_vec = alpha * buf_dma + beta
        out[0:v0, :] = (buf_vec[:v0, :].astype(np.float64) @ weight.astype(np.float64)).astype(np.float32)
        return out

    # Initialize triple buffers
    buf_dma = np.zeros((TILE_H, W), dtype=np.float32)
    buf_vec = np.zeros((TILE_H, W), dtype=np.float32)
    buf_tensor = np.zeros((TILE_H, W), dtype=np.float32)
    buf_out = np.zeros((TILE_H, K), dtype=np.float32)

    # 1. Prologue: Prefetch Block 0 and Block 1
    v0 = get_valid(0)
    buf_dma[:v0, :] = x[0:v0, :]
    buf_vec, buf_dma = buf_dma, buf_vec
    v1 = get_valid(1)
    buf_dma[:v1, :] = x[TILE_H:TILE_H + v1, :]
    buf_vec[:v0, :] = alpha * buf_vec[:v0, :] + beta

    # 2. Main Loop: Overlap DMA(b) || Vector(b-1) || Tensor(b-2)
    for b in range(2, num_blocks):
        # Rotate buffers
        buf_tensor, buf_vec, buf_dma = buf_vec, buf_dma, buf_tensor
        vb = get_valid(b)
        r_st = b * TILE_H
        buf_dma[:vb, :] = x[r_st:r_st + vb, :]

        vb_prev = get_valid(b - 1)
        buf_vec[:vb_prev, :] = alpha * buf_vec[:vb_prev, :] + beta

        vb_prev2 = get_valid(b - 2)
        buf_out[:vb_prev2, :] = (buf_tensor[:vb_prev2, :].astype(np.float64) @ weight.astype(np.float64)).astype(np.float32)
        r_out_st = (b - 2) * TILE_H
        out[r_out_st:r_out_st + vb_prev2, :] = buf_out[:vb_prev2, :]

    # 3. Epilogue: Drain remaining Block num_blocks-2 and num_blocks-1
    buf_tensor, buf_vec = buf_vec, buf_tensor
    v_penult = get_valid(num_blocks - 2)
    buf_out[:v_penult, :] = (buf_tensor[:v_penult, :].astype(np.float64) @ weight.astype(np.float64)).astype(np.float32)
    out[(num_blocks - 2) * TILE_H:(num_blocks - 2) * TILE_H + v_penult, :] = buf_out[:v_penult, :]

    v_last = get_valid(num_blocks - 1)
    buf_dma[:v_last, :] = alpha * buf_dma[:v_last, :] + beta
    buf_out[:v_last, :] = (buf_dma[:v_last, :].astype(np.float64) @ weight.astype(np.float64)).astype(np.float32)
    out[(num_blocks - 1) * TILE_H:(num_blocks - 1) * TILE_H + v_last, :] = buf_out[:v_last, :]

    return out
