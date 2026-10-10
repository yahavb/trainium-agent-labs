# L10 conv1d: ours ve22cca run2 (results/seat-134/ours-L8910-seat-134.jsonl); organizers' checker: 4/4 cases
import numpy as np

def kernel(x, w, stride, dilation):
    # Input shape: (C_in, L)
    # Weights shape: (C_out, C_in, K)
    C_in, L = x.shape
    C_out, _, K = w.shape

    # Compute output length
    L_out = (L - dilation * (K - 1) - 1) // stride + 1

    # Initialize output
    out = np.zeros((C_out, L_out), dtype=np.float32)

    # Iterate over output positions
    for t in range(L_out):
        # Compute the starting index in x
        start = t * stride
        # Compute the end index in x
        end = start + dilation * (K - 1) + 1
        if end > L:
            end = L
        # Compute the number of valid positions
        valid = end - start

        # Ensure the slice has exactly K elements
        if valid < K:
            # Not enough elements, skip this position (should not happen with no padding)
            continue
        elif valid > K:
            # Too many elements, truncate to K
            end = start + K
            valid = K

        # Iterate over output channels
        for o in range(C_out):
            # Iterate over input channels
            for c in range(C_in):
                # Compute the weights for this input channel
                weights = w[o, c, :]

                # Extract the slice from x
                x_slice = x[c, start:end]

                # Compute the output value for this position
                acc = 0.0
                for k in range(K):
                    # Corrected index with dilation
                    idx = start + k * dilation
                    acc += weights[k] * x[c, idx]
                out[o, t] += acc

    return out
