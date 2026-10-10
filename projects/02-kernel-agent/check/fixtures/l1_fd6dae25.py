import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    
    # Allocate tiles for data and results
    data_tile = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    result_tile = nl.ndarray((C, H // p, W // p), dtype=x.dtype, buffer=nl.sbuf)
    
    # Copy input to SBUF
    nisa.dma_copy(dst=data_tile, src=x)
    
    # Compute the average over the pooling window
    for c in nl.affine_range(C):
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                # Create a view of the pooling window
                view = data_tile[c, h*p:(h+1)*p, w*p:(w+1)*p]
                # Compute the sum of the view
                sum_view = nl.sum(view, axis=[1, 2], keepdims=True)
                # Create a tile for the scalar multiplication
                mean_tile = nl.ndarray(sum_view.shape, dtype=sum_view.dtype, buffer=nl.sbuf)
                # Multiply sum by 1/(p*p) using nisa.tensor_scalar
                nisa.tensor_scalar(dst=mean_tile, data=sum_view, op0=nl.multiply, operand0=1.0 / (p * p))
                # Store the result
                result_tile[c, h, w] = mean_tile[0, 0, 0]
    
    # Copy result from SBUF to shared HBM
    nisa.dma_copy(dst=out, src=result_tile)
    return out