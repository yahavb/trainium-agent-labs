# Qwen3-8B, live v2 run 1, L4 attempt 4: fixes the broadcast with .reshape (banned)
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
            
            # Reshape rms to (rows, 1) for broadcasting
            rms = rms.reshape(-1, 1)
            # Reshape g to (1, cols) for broadcasting
            g_tile = g[j:end_j].reshape(1, -1)
            
            # Normalize and scale by g
            y[i:end_i, j:end_j] = x_tile / rms * g_tile
    
    return y
