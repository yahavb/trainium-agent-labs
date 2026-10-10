import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def instnorm_gelu_kernel(x):
    # Get input dimensions
    C, N = x.shape
    # Allocate output tensor in shared HBM
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    # Process in tiles of at most 128 rows (partition axis)
    for i in range((C + 127) // 128):
        r0 = 128 * i
        rows = min(128, C - r0)
        # Load a tile of x into on-chip memory
        tile_x = nl.ndarray((rows, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile_x, src=x[r0:r0 + rows, :])

        # Compute mean (sum over N, keepdims=True)
        sum_tile = nl.sum(tile_x, axis=1, keepdims=True)
        mean = nl.ndarray((rows, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mean, data=sum_tile, op0=nl.divide, operand0=N)

        # Compute (x - mean)
        subtracted = nl.ndarray((rows, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=subtracted, data=tile_x, op0=nl.subtract, operand0=mean)

        # Compute variance (sum of squared differences, divide by N)
        squared = nl.ndarray((rows, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=squared, data1=subtracted, data2=subtracted, op=nl.multiply)
        sum_squared = nl.sum(squared, axis=1, keepdims=True)
        var = nl.ndarray((rows, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=var, data=sum_squared, op0=nl.divide, operand0=N)

        # Add epsilon and compute sqrt(var + 1e-5)
        epsilon = 1e-5
        var_plus_epsilon = nl.ndarray((rows, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=var_plus_epsilon, data=var, op0=nl.add, operand0=epsilon)
        std = nl.ndarray((rows, 1), dtype=x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=std, op=nl.rsqrt, data=var_plus_epsilon)

        # Normalize: (x - mean) / sqrt(var + 1e-5)
        xn = nl.ndarray((rows, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xn, data=subtracted, op0=nl.multiply, operand0=std)

        # Apply GELU
        gelu_xn = nl.ndarray((rows, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=gelu_xn, op=nl.gelu, data=xn)

        # Cap at 10.0
        capped = nl.ndarray((rows, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=capped, data=gelu_xn, op0=nl.minimum, operand0=10.0)

        # Store result back to output
        nisa.dma_copy(dst=y[r0:r0 + rows, :], src=capped)

    # Transfer ownership of y to the caller
    return y