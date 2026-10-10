# L8 layernorm: ours vf0c39f run3 (results/seat-130/ours-L8-seat-130.jsonl); organizers' checker: 32/32 cases
import numpy as np

def kernel(x, eps=1e-5):
    M, N = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    tile_rows = 128
    tile_cols = 512

    for i in range(0, M, tile_rows):
        end_i = min(i + tile_rows, M)
        num_rows = end_i - i

        # Initialize mean and variance for the current row tile
        mu = np.zeros(num_rows, dtype=np.float64)
        ss = np.zeros(num_rows, dtype=np.float64)

        # Pass 1: Compute mean
        for j in range(0, N, tile_cols):
            end_j = min(j + tile_cols, N)
            num_cols = end_j - j

            # Extract the current tile
            tile = x[i:end_i, j:end_j]

            # Compute sum of the current tile for each row
            row_sums = np.zeros(num_rows, dtype=np.float64)
            for r in range(num_rows):
                for c in range(num_cols):
                    row_sums[r] += tile[r, c]

            # Update mean
            mu += row_sums / N

        # Pass 2: Compute variance
        for j in range(0, N, tile_cols):
            end_j = min(j + tile_cols, N)
            num_cols = end_j - j

            # Extract the current tile
            tile = x[i:end_i, j:end_j]

            # Compute deviation from mean
            for r in range(num_rows):
                for c in range(num_cols):
                    d = tile[r, c] - mu[r]
                    ss[r] += d * d

        # Compute standard deviation
        std = np.sqrt(ss / N + eps)

        # Pass 3: Normalize
        for j in range(0, N, tile_cols):
            end_j = min(j + tile_cols, N)
            num_cols = end_j - j

            # Extract the current tile
            tile = x[i:end_i, j:end_j]

            # Normalize each element
            for r in range(num_rows):
                for c in range(num_cols):
                    y[i + r, j + c] = (tile[r, c] - mu[r]) / std[r]

    return y
