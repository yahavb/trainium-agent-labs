import numpy as np


def kernel(x):
    M, N = x.shape
    result = np.zeros(M, dtype=np.float32)
    for i in range(0, M, 128):
        for j in range(0, N, 512):
            end_row = min(i + 128, M)
            end_col = min(j + 512, N)
            for row in range(i, end_row):
                acc = -np.inf
                for col in range(j, end_col):
                    if x[row, col] > acc:
                        acc = x[row, col]
                result[row] = acc
    return result
