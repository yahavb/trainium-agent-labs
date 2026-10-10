# Qwen3-8B, live v3 run, L8 run 5 attempt 11: mean/var of each column over the tile rows; v3 said
# only "re-derive the formula" (numeric:other) 15 times

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
            
            num_rows = end_i - i
            num_cols = end_j - j
            
            # Precompute g and b for this tile
            g_tile = g[j:j + num_cols]
            b_tile = b[j:j + num_cols]
            
            # Process each column in the tile
            for col in range(num_cols):
                # Extract the current column slice
                x_col = x[i:end_i, j + col]
                
                # Compute mean of this column (over rows)
                total = 0.0
                for row in range(num_rows):
                    total += x_col[row]
                mean = total / num_rows
                
                # Compute variance of this column
                var = 0.0
                for row in range(num_rows):
                    diff = x_col[row] - mean
                    var += diff * diff
                var /= num_rows
                
                # Compute standard deviation
                std = np.sqrt(var + eps)
                
                # Normalize the column
                normalized = (x_col - mean) / std
                
                # Apply scaling and shifting
                y[i:end_i, j + col] = normalized * g_tile[col] + b_tile[col]
    
    return y
