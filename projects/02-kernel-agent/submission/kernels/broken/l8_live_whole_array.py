# Qwen3-8B, live naive run, L8 run 5 attempt 6: np.mean/np.var per column tile, 7 violations
import numpy as np

def kernel(x, eps=1e-5):
    g = np.ones(x.shape[1], dtype=np.float32)  # v4 signature; body as the model wrote it
    b = np.zeros(x.shape[1], dtype=np.float32)
    M, N = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    
    # Tile size
    tile_rows = 128
    tile_cols = 512
    
    for i in range(0, M, tile_rows):
        for j in range(0, N, tile_cols):
            # Determine the actual tile bounds
            end_i = min(i + tile_rows, M)
            end_j = min(j + tile_cols, N)
            
            # Extract tile
            tile_x = x[i:end_i, j:end_j]
            
            # Compute mean and variance for each row in the current tile
            mean = np.mean(tile_x, axis=1, keepdims=True)  # Keepdims=True
            var = np.var(tile_x, axis=1, keepdims=True)    # Keepdims=True
            
            # Normalize
            std = np.sqrt(var + eps)
            tile_y = (tile_x - mean) / std
            
            # Scale and shift
            g_tile = g[j:end_j]
            b_tile = b[j:end_j]
            tile_y = tile_y * g_tile + b_tile
            
            # Assign back to output
            y[i:end_i, j:end_j] = tile_y
    
    return y
