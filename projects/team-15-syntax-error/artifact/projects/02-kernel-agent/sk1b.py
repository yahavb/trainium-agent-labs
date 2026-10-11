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
    out_tile = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(Ho):
        for j in nl.affine_range(Wo):
            win = in_tile[:, i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]
            s = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=s, data=win, op=nl.add, axis=(1, 2))
            nisa.tensor_scalar(dst=out_tile[:, i, j:j + 1], data=s, op0=nl.multiply,
                               operand0=1.0 / (pool_size * pool_size))
    nisa.dma_copy(dst=out_tensor, src=out_tile)
    return out_tensor
