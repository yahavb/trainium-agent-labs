# Qwen3-8B, live v2 run 3, L4 attempt 2: divides a (rows, cols) tile by a (rows,) vector
import numpy as np

def kernel(x, eps=1e-6):
    g = np.ones(x.shape[1], dtype=np.float32)
    M, N = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    tile_size = (128, 512)
    
    for i in range(0, M, tile_size[0]):
        for j in range(0, N, tile_size[1]):
            # Determine the actual tile bounds
            i_end = min(i + tile_size[0], M)
            j_end = min(j + tile_size[1], N)
            
            # Extract the tile from x
            x_tile = x[i:i_end, j:j_end]
            
            # Compute the mean of squared values along the last axis (axis=1)
            row_count = i_end - i
            acc = np.zeros(row_count, dtype=np.float32)
            for r in range(row_count):
                for c in range(j_end - j):
                    acc[r] += x_tile[r, c] ** 2
            mean_sq = acc / (j_end - j)  # Divide by N to get the mean
            
            # Compute the RMS normalization factor
            norm = np.sqrt(mean_sq + eps)
            
            # Normalize the tile and scale by g
            y[i:i_end, j:j_end] = (x_tile / norm) * g[j:j_end]
    
    return y
