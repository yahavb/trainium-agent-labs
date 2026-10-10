import numpy as np


def kernel(x, a, b):
    return np.maximum(a * x + b, 0.0)
