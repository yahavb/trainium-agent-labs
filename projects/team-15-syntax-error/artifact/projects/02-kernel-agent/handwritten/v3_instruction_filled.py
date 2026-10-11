import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
    C, H, W = in_tensor.shape
    Ho, Wo = H // pool_size, W // pool_size
    out_tensor = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    in_tile = nl.ndarray((C, H, W), dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_tile, src=in_tensor)
    sum_tile = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.sbuf)
    p = pool_size
    for h in nl.affine_range(Ho):
        for w in nl.affine_range(Wo):
            s = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=s, data=in_tile[:, h*p:(h+1)*p, w*p:(w+1)*p], op=nl.add, axis=(1, 2))
            nisa.tensor_copy(dst=sum_tile[:, h, w:w+1], src=s)
    scale_tile = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scale_tile, data=sum_tile, op0=nl.multiply, operand0=1.0 / (pool_size * pool_size))
    nisa.dma_copy(dst=out_tensor, src=scale_tile)
    return out_tensor
