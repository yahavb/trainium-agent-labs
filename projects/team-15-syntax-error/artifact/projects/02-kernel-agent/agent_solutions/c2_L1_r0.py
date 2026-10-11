# Written by Qwen3-8B (local, seat 73) in run c2_L1_r0, round 1, under --skeleton --v2-verdicts --v3-verdicts --directional.
# Verified [sim]: all public shapes + held-out (heldout.py). Not edited by hand.
import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def tensor_avgpool_kernel(in_tensor, pool_size):
    C, H, W = in_tensor.shape                 # 3-D input; C (<= 128) is the partition dimension
    Ho, Wo = H // pool_size, W // pool_size
    out_tensor = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    in_tile = nl.ndarray((C, H, W), dtype=in_tensor.dtype, buffer=nl.sbuf)   # tiles are never 1-D
    nisa.dma_copy(dst=in_tile, src=in_tensor)
    
    # Compute each window's sum into an SBUF tile of shape (C, Ho, Wo) -- slice in_tile
    sum_tile = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.sbuf)
    for c in nl.affine_range(C):
        for h in nl.affine_range(Ho):
            for w in nl.affine_range(Wo):
                # Allocate a temporary tile for the window reduction
                s = nl.ndarray((1, 1), dtype=in_tensor.dtype, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=s, data=in_tile[c, h*pool_size:(h+1)*pool_size, w*pool_size:(w+1)*pool_size], op=nl.add, axis=(1, 2))
                nisa.tensor_copy(dst=sum_tile[c, h, w], src=s)
    
    # Multiply by 1 / (pool_size * pool_size) into an SBUF tile of shape (C, Ho, Wo)
    scale_tile = nl.ndarray((C, Ho, Wo), dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=scale_tile, data=sum_tile, op0=nl.multiply, operand0=1.0 / (pool_size * pool_size))
    
    # nisa.dma_copy that tile into out_tensor
    nisa.dma_copy(dst=out_tensor, src=scale_tile)
    return out_tensor