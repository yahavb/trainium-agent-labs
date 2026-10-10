import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    
    for c in nl.affine_range(C):
        for h in nl.affine_range(H // p):
            for w in nl.affine_range(W // p):
                # Allocate tile with the exact shape of the pool region
                tile = nl.ndarray(x[c, h*p:(h+1)*p, w*p:(w+1)*p].shape, dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=x[c, h*p:(h+1)*p, w*p:(w+1)*p])
                
                # First sum over the height dimension (axis 1)
                tile_sum_h = nl.ndarray((1, p), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=tile_sum_h, op=nl.add, data=tile, axis=1)
                
                # Then sum over the width dimension (axis 1)
                s = nl.ndarray((1, 1), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=s, op=nl.add, data=tile_sum_h, axis=1)
                
                # Normalize
                # Move result from sbuf to shared_hbm
                nisa.dma_copy(dst=out[c, h, w], src=s)
                # Multiply by 1/(p*p) using scalar operation on sbuf
                nisa.tensor_scalar(dst=s, data=s, op0=nl.multiply, operand0=1.0 / (p * p))
                # Copy back to shared_hbm
                nisa.dma_copy(dst=out[c, h, w], src=s)
    
    return out