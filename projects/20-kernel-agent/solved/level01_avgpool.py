import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(x, pool_size):
    C, H, W = x.shape
    p = pool_size
    out_shape = (C, H // p, W // p)
    out = nl.ndarray(out_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    
    in_tile = nl.ndarray((C, H, W), dtype=x.dtype, buffer=nl.sbuf)
    sum_tile = nl.ndarray((C, H // p, W // p), dtype=x.dtype, buffer=nl.sbuf)
    out_tile = nl.ndarray((C, H // p, W // p), dtype=x.dtype, buffer=nl.sbuf)
    
    nisa.dma_copy(dst=in_tile, src=x)
    
    view = in_tile.ap([[H*W, C], [p*W, H//p], [p, W//p], [W, p], [1, p]])
    sum_tile = nl.sum(view, axis=[3, 4])
    
    nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply, operand0=1.0 / (p * p))
    nisa.dma_copy(dst=out, src=out_tile)
    
    return out
