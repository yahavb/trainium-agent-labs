"""Astra trial B: bounded LHS residency and reuse across N tiles."""

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

    if K <= 16384:
        # At most 4 MiB of BF16 LHS storage, independent of M and N.
        for m in nl.affine_range(M // 128):
            left = nl.ndarray((128, K // 128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(K // 128):
                nisa.dma_copy(
                    dst=left[:, k, :],
                    src=lhsT[k * 128:(k + 1) * 128, m * 128:(m + 1) * 128],
                )
            for n in nl.affine_range(N // 512):
                acc = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // 128):
                    right = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=right,
                        src=rhs[k * 128:(k + 1) * 128, n * 512:(n + 1) * 512],
                    )
                    nisa.nc_matmul(dst=acc, stationary=left[:, k, :], moving=right)
                result_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=result_tile, src=acc)
                nisa.dma_copy(
                    dst=out[m * 128:(m + 1) * 128, n * 512:(n + 1) * 512],
                    src=result_tile,
                )
    else:
        # Stream K while retaining up to two output tiles in PSUM.
        # Odd N tile counts use a one-tile block with exact, unmasked slices.
        block_n = 2 if N % 1024 == 0 else 1
        for m in nl.affine_range(M // 128):
            for nb in nl.affine_range(N // (512 * block_n)):
                acc = nl.ndarray((128, block_n, 512), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // 128):
                    left = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=left,
                        src=lhsT[k * 128:(k + 1) * 128, m * 128:(m + 1) * 128],
                    )
                    for ni in nl.affine_range(block_n):
                        right = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(
                            dst=right,
                            src=rhs[k * 128:(k + 1) * 128,
                                    (nb * block_n + ni) * 512:(nb * block_n + ni + 1) * 512],
                        )
                        nisa.nc_matmul(dst=acc[:, ni, :], stationary=left, moving=right)
                for ni in nl.affine_range(block_n):
                    result_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=result_tile, src=acc[:, ni, :])
                    nisa.dma_copy(
                        dst=out[m * 128:(m + 1) * 128,
                                (nb * block_n + ni) * 512:(nb * block_n + ni + 1) * 512],
                        src=result_tile,
                    )
    return out
