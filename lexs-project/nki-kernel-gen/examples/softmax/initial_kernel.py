import nki.language as nl
import nki.isa as nisa


# EVOLVE-BLOCK-START
def kernel(x):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for tile in nl.affine_range((rows + 127) // 128):
        start = tile * 128
        size = min(128, rows - start)
        values = nl.ndarray((size, cols), dtype=nl.float32, buffer=nl.sbuf)
        maximum = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
        shifted = nl.ndarray((size, cols), dtype=nl.float32, buffer=nl.sbuf)
        numerator = nl.ndarray((size, cols), dtype=nl.float32, buffer=nl.sbuf)
        denominator = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
        inverse = nl.ndarray((size, 1), dtype=nl.float32, buffer=nl.sbuf)
        result = nl.ndarray((size, cols), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=values, src=x[start:start + size, :])
        nisa.tensor_reduce(dst=maximum, op=nl.max, data=values, axis=1, keepdims=True)
        nisa.tensor_scalar(dst=shifted, data=values, op0=nl.subtract, operand0=maximum)
        nisa.activation(dst=numerator, op=nl.exp, data=shifted)
        nisa.tensor_reduce(dst=denominator, op=nl.add, data=numerator, axis=1, keepdims=True)
        nisa.activation(dst=inverse, op=nl.reciprocal, data=denominator)
        nisa.tensor_scalar(dst=result, data=numerator, op0=nl.multiply, operand0=inverse)
        nisa.dma_copy(dst=out[start:start + size, :], src=result)
    return out
# EVOLVE-BLOCK-END
