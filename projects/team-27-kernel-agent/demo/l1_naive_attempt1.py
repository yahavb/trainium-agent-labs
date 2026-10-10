# Model's own kernel, logged in results/seat-130/naive-L1234-seat-130.jsonl: level 1, run 0, attempt 1.
import numpy as np

def kernel(x, a, b):
    # Ensure x is a numpy array of float32
    x = np.asarray(x, dtype=np.float32)
    result = np.zeros_like(x)
    rows, cols = x.shape
    tile_rows = 128
    tile_cols = 512

    # Process rows in tiles
    for i in range(0, rows, tile_rows):
        end_row = min(i + tile_rows, rows)
        # Process columns in tiles
        for j in range(0, cols, tile_cols):
            end_col = min(j + tile_cols, cols)
            tile = x[i:end_row, j:end_col]
            result[i:end_row, j:end_col] = np.maximum(0, a * tile + b)
    return result
