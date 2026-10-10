# Qwen3-8B, live v2 seat-134, L9 run 1 attempt 1: np.dot crashes; v2 asked to fix the crash 3x, then stuck
import numpy as np

def kernel(q, k, v, w):
    S, d = q.shape
    out = np.zeros((S, d), dtype=np.float32)
    
    tile_rows = 128
    tile_cols = 512
    
    for i in range(0, S, tile_rows):
        for j in range(0, d, tile_cols):
            # Determine the current tile's row and column range
            start_row = i
            end_row = min(i + tile_rows, S)
            start_col = j
            end_col = min(j + tile_cols, d)
            
            # Extract the current tile of q, k, v
            q_tile = q[start_row:end_row, start_col:end_col]
            k_tile = k[start_row:end_row, start_col:end_col]
            v_tile = v[start_row:end_row, start_col:end_col]
            
            # Compute the attention scores for the current tile
            # For each query row, compute the attention weights over the window
            for row in range(end_row - start_row):
                q_row = q_tile[row]
                # Compute the window for the current query row
                window_start = max(0, i + row - w)
                window_end = min(S, i + row + w + 1)
                # Extract the window of k values
                k_window = k[window_start:window_end, start_col:end_col]
                # Compute the attention scores
                scores = np.dot(q_row, k_window.T) / np.sqrt(d)
                # Apply softmax in a numerically stable way
                scores -= np.max(scores)
                scores = np.exp(scores)
                scores /= np.sum(scores)
                # Compute the weighted sum of v values
                v_window = v[window_start:window_end, start_col:end_col]
                out[start_row + row, start_col:end_col] = np.dot(scores, v_window.T)
    
    return out
