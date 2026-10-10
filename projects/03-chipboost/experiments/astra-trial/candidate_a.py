"""Astra trial A: bounded RHS residency and reuse across M tiles."""

import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    K, M = lhsT.shape
    K_rhs, N = rhs.shape
    assert K == K_rhs
    assert K % 128 == 0
    assert M % 128 == 0
    assert N % 512 == 0
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    if K <= 4096:
        # At most 4 MiB of BF16 RHS storage, independent of M and N.
        for n in nl.affine_range(N // 512):
            right = nl.ndarray((128, K // 128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(K // 128):
                nisa.dma_copy(
                    dst=right[:, k, :],
                    src=rhs[k * 128:(k + 1) * 128, n * 512:(n + 1) * 512],
                )
            for m in nl.affine_range(M // 128):
                acc = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // 128):
                    left = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=left,
                        src=lhsT[k * 128:(k + 1) * 128, m * 128:(m + 1) * 128],
                    )
                    nisa.nc_matmul(dst=acc, stationary=left, moving=right[:, k, :])
                result_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=result_tile, src=acc)
                nisa.dma_copy(
                    dst=out[m * 128:(m + 1) * 128, n * 512:(n + 1) * 512],
                    src=result_tile,
                )
    else:
        # Stream K while retaining up to two output tiles in PSUM.
        # Odd M tile counts select a one-tile block, avoiding masked accesses.
        block_m = 2 if M % 256 == 0 else 1
        for n in nl.affine_range(N // 512):
            for mb in nl.affine_range(M // (128 * block_m)):
                acc = nl.ndarray((128, block_m, 512), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // 128):
                    right = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=right,
                        src=rhs[k * 128:(k + 1) * 128, n * 512:(n + 1) * 512],
                    )
                    for mi in nl.affine_range(block_m):
                        left = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(
                            dst=left,
                            src=lhsT[k * 128:(k + 1) * 128,
                                     (mb * block_m + mi) * 128:(mb * block_m + mi + 1) * 128],
                        )
                        nisa.nc_matmul(dst=acc[:, mi, :], stationary=left, moving=right)
                for mi in nl.affine_range(block_m):
                    result_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=result_tile, src=acc[:, mi, :])
                    nisa.dma_copy(
                        dst=out[(mb * block_m + mi) * 128:(mb * block_m + mi + 1) * 128,
                                n * 512:(n + 1) * 512],
                        src=result_tile,
                    )
    return out
