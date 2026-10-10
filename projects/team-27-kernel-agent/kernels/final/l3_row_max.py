# L3 row_max: agent run3, then optimised (results/seat-130/opt2-live.jsonl); organizers' checker: 32/32 cases
import numpy as np

def kernel(x):
    M, N = x.shape
    result = np.empty(M, dtype=np.float32)
    result[:] = -np.inf  # Initialize with negative infinity
    tile_rows = 128
    tile_cols = 512

    for i in range(0, M, tile_rows):
        end_i = min(i + tile_rows, M)
        for j in range(0, N, tile_cols):
            end_j = min(j + tile_cols, N)
            tile = x[i:end_i, j:end_j]
            num_rows, num_cols = tile.shape

            # Initialize accumulation array for the current tile
            acc = np.full(num_rows, -np.inf, dtype=np.float32)

            # For each column in the tile, compute max across the rows
            for col in range(num_cols):
                acc = np.maximum(acc, tile[:, col])

            # Update the result with the accumulated maxima
            result[i:end_i] = np.maximum(result[i:end_i], acc)

    return result
