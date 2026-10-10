# L5 softmax: ours vc5e98d run1 (results/erfu-c5e98d5a75-partial/ours-L145-seat-134.jsonl); organizers' checker: 32/32 cases
import numpy as np

def kernel(x):
    M, N = x.shape
    y = np.zeros_like(x, dtype=np.float32)
    
    # Step 1: Compute row-wise max using explicit loops
    row_max = np.full(M, -np.inf, dtype=np.float64)
    for i in range(M):
        max_val = -np.inf
        for j in range(N):
            if x[i, j] > max_val:
                max_val = x[i, j]
        row_max[i] = max_val
    
    # Step 2: Compute row-wise sum of exponentials using explicit loops
    row_sum = np.zeros(M, dtype=np.float64)
    for i in range(M):
        s = 0.0
        for j in range(N):
            s += np.exp((x[i, j] - row_max[i]).astype(np.float64))
        row_sum[i] = s
    
    # Step 3: Compute softmax using explicit loops
    for i in range(M):
        for j in range(N):
            y[i, j] = np.exp((x[i, j] - row_max[i]).astype(np.float64)) / row_sum[i]
    
    return y
