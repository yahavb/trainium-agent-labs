# Model's own kernel, logged in results/seat-130/ours-L1234-seat-130.jsonl: level 2, run 0, attempt 1.
import numpy as np

def kernel(x):
    M, N = x.shape
    result = np.zeros(M, dtype=np.float32)
    tile_rows = 128
    tile_cols = 512
    
    for i in range(0, M, tile_rows):
        end_row = min(i + tile_rows, M)
        for j in range(0, N, tile_cols):
            end_col = min(j + tile_cols, N)
            tile = x[i:end_row, j:end_col]
            result[i:end_row] += np.sum(tile, axis=1)
    return result
