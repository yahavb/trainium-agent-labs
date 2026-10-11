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
    acc = nl.ndarray((C, Ho, Wo), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for di in nl.affine_range(pool_size):
        for dj in nl.affine_range(pool_size):
            nisa.tensor_tensor(dst=acc, data1=acc,
                               data2=in_tile[:, di:Ho * pool_size:pool_size, dj:Wo * pool_size:pool_size],
                               op=nl.add)
    out_tile = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=out_tile, data=acc, op0=nl.multiply, operand0=1.0 / (pool_size * pool_size))
    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
