# Qwen3-8B, live v2 seat-133, L7: slices longer than one tile; generic fix talks about x[i0:i1, j0:j1]
import numpy as np

def kernel(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Dimensions do not match for matrix multiplication"
    
    tile_rows = 128
    tile_cols = 512
    
    # Initialize result
    c = np.zeros((M, N), dtype=np.float32)
    
    # Process rows in tiles
    for i in range(0, M, tile_rows):
        end_i = min(i + tile_rows, M)
        tile_a = a[i:end_i, :]
        
        # Process columns in tiles
        for j in range(0, N, tile_cols):
            end_j = min(j + tile_cols, N)
            tile_b = b[:, j:end_j]
            
            # Compute tile product
            tile_c = np.dot(tile_a, tile_b)
            c[i:end_i, j:end_j] = tile_c
    
    return c
