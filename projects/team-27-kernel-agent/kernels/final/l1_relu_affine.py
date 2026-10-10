# L1 relu_affine: naive ve22cca run3 (results/seat-130/naive-L1234-seat-130.jsonl); organizers' checker: 32/32 cases
import numpy as np

def kernel(x, a, b):
    # Get the shape of the input array
    rows, cols = x.shape
    result = np.zeros_like(x, dtype=np.float32)
    
    # Tile size
    tile_rows = 128
    tile_cols = 512
    
    # Iterate over the array in tiles
    for i in range(0, rows, tile_rows):
        for j in range(0, cols, tile_cols):
            # Determine the actual tile size
            end_row = min(i + tile_rows, rows)
            end_col = min(j + tile_cols, cols)
            
            # Extract the tile
            tile = x[i:end_row, j:end_col]
            
            # Compute a*x + b elementwise
            temp = a * tile + b
            
            # Apply ReLU
            result[i:end_row, j:end_col] = np.maximum(temp, 0.0)
    
    return result
