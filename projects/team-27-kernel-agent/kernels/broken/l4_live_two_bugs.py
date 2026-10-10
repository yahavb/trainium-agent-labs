# Qwen3-8B, live run 1, L4 attempt 7: per-tile mean square AND float32 squares (two bugs at once)
import numpy as np

def kernel(x, eps=1e-6):
    g = np.ones(x.shape[1], dtype=np.float32)
    M, N = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    tile_rows = 128
    tile_cols = 512

    for i in range(0, M, tile_rows):
        for j in range(0, N, tile_cols):
            end_i = min(i + tile_rows, M)
            end_j = min(j + tile_cols, N)
            
            # Extract tile
            x_tile = x[i:end_i, j:end_j]
            
            # Compute RMS norm over the last axis (columns) using explicit loop
            rows = end_i - i
            cols = end_j - j
            acc = np.zeros(rows, dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    acc[r] += x_tile[r, c] ** 2  # Squared value
            mean = acc / cols
            rms = np.sqrt(mean + eps)
            
            # Normalize and scale by g
            for r in range(rows):
                for c in range(cols):
                    y[i + r, j + c] = x_tile[r, c] / rms[r] * g[j + c]
    
    return y
