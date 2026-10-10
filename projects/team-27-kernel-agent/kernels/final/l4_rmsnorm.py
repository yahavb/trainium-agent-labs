# L4 rmsnorm: ours vc5e98d run5 (results/erfu-c5e98d5a75-partial/ours-L145-seat-134.jsonl); organizers' checker: 32/32 cases
import numpy as np

def kernel(x, eps=1e-6):
    M, N = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    tile_rows = 128
    tile_cols = 512

    # Pre-allocate row-wise squared sum
    row_sums = np.zeros(M, dtype=np.float64)

    for i in range(0, M, tile_rows):
        for j in range(0, N, tile_cols):
            # Extract tile
            tile = x[i:i+tile_rows, j:j+tile_cols]
            num_rows, num_cols = tile.shape

            # Cast to float64 to avoid underflow/overflow
            tile_float64 = tile.astype(np.float64)

            # Accumulate squared values for each row
            for row in range(num_rows):
                for col in range(num_cols):
                    row_sums[i + row] += tile_float64[row, col] ** 2

    # Compute RMS for each row
    rms = np.sqrt(row_sums / N + eps)

    # Normalize the entire matrix using the row-wise RMS
    for i in range(0, M, tile_rows):
        for j in range(0, N, tile_cols):
            # Extract tile
            tile = x[i:i+tile_rows, j:j+tile_cols]
            num_rows, num_cols = tile.shape
            tile_float64 = tile.astype(np.float64)

            # Get the row-wise RMS for this tile
            rinv = 1.0 / rms[i:i+num_rows]

            # Normalize column by column
            for col in range(num_cols):
                y[i:i+num_rows, j + col] = tile_float64[:, col] * rinv

    return y
