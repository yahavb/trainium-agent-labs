import nki
import nki.isa as nisa
import nki.language as nl
from math import ceil


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    K, M = lhsT.shape
    _, N = rhs.shape
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # Loops and slices generated from your tiling plan. They are correct: do not change them.
    for m in nl.affine_range(int(M // 128)):
        for n in nl.affine_range(int(N // 512)):
            acc = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)  # PSUM accumulator for this output tile
            for k in nl.affine_range(int(K // 128)):
                lhsT_slice = lhsT[k * 128:(k + 1) * 128, m * 128:(m + 1) * 128]  # 128 x 128, in HBM
                rhs_slice = rhs[k * 128:(k + 1) * 128, n * 512:(n + 1) * 512]  # 128 x 512, in HBM
                sbuf_lhs = nl.ndarray((128, 128), dtype=lhsT_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_lhs, src=lhsT_slice)
                sbuf_rhs = nl.ndarray((128, 512), dtype=rhs_slice.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_rhs, src=rhs_slice)
                nisa.nc_matmul(dst=acc, stationary=sbuf_lhs, moving=sbuf_rhs)
            out_slice = result[m * 128:(m + 1) * 128, n * 512:(n + 1) * 512]  # 128 x 512, in HBM
            nisa.tensor_copy(dst=sbuf_rhs, src=acc)
            nisa.dma_copy(dst=out_slice, src=sbuf_rhs)
    return result
