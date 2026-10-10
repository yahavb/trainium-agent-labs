# L2 row_sum: agent run2, then optimised (results/seat-130/opt2-live.jsonl); organizers' checker: 32/32 cases
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
            # Extract the tile
            tile = x[i:end_row, j:end_col]
            # Initialize accumulator for the current tile rows
            acc = np.zeros(end_row - i, dtype=np.float32)
            # Sum over columns for each row in the tile
            for col in range(end_col - j):
                acc += tile[:, col]
            # Add the accumulated row sums to the result
            result[i:end_row] += acc
    return result
