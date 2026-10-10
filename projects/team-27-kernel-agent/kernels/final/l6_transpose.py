# L6 transpose: ours v29fd8d run2 (results/seat-130/v3-final/ours-L456-seat-130.jsonl); organizers' checker: 32/32 cases
import numpy as np

def kernel(x):
    M, N = x.shape
    y = np.zeros((N, M), dtype=np.float32)
    
    # Tile size
    tile_rows = 128
    tile_cols = 512
    
    # Iterate over rows and columns in tile steps
    for i in range(0, M, tile_rows):
        for j in range(0, N, tile_cols):
            # Determine the actual tile size
            end_row = min(i + tile_rows, M)
            end_col = min(j + tile_cols, N)
            
            # Extract tile from x
            tile_x = x[i:end_row, j:end_col]
            
            # Compute the corresponding tile in y
            tile_y_row = j
            tile_y_col = i
            
            # Copy data from tile_x to y without using .T
            for r in range(end_row - i):
                for c in range(end_col - j):
                    y[tile_y_row + c, tile_y_col + r] = tile_x[r, c]
    
    return y
