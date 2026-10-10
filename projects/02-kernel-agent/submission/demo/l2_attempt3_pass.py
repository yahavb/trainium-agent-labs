# Model's own kernel, logged in results/seat-130/ours-L1234-seat-130.jsonl: level 2, run 0, attempt 3.
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
            acc = np.zeros(end_row - i, dtype=np.float32)
            for col in range(j, end_col):
                acc += tile[:, col - j]
            result[i:end_row] += acc
    return result
