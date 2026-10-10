# Qwen3-8B, live v3 seat-130, L6: a tile sliced with global columns -> empty value, stuck 3x on the crash
import numpy as np

def kernel(x):
    M, N = x.shape
    y = np.zeros((N, M), dtype=np.float32)
    tile_rows = 128
    tile_cols = 512

    for i in range(0, M, tile_rows):
        for j in range(0, N, tile_cols):
            end_i = min(i + tile_rows, M)
            end_j = min(j + tile_cols, N)
            tile = x[i:end_i, j:end_j]
            for r in range(i, end_i):
                y[j:end_j, r] = tile[r - i, j:end_j]
    return y
