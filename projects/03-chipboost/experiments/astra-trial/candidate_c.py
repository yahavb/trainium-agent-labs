"""Astra iterative trial C: RHS residency plus multi-column accumulation."""

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

    if K <= 8192:
        # Select an exact divisor of N, with at most 8 MiB of resident BF16 RHS.
        block_n = 1
        if N % 2048 == 0 and K <= 2048:
            block_n = 4
        elif N % 1536 == 0 and K <= 2688:
            block_n = 3
        elif N % 1024 == 0 and K <= 4096:
            block_n = 2

        for nb in nl.affine_range(N // (512 * block_n)):
            right = nl.ndarray((128, K // 128, 512 * block_n),
                               dtype=rhs.dtype, buffer=nl.sbuf)
            for k in nl.affine_range(K // 128):
                # A wider contiguous DMA fills all resident column tiles.
                nisa.dma_copy(
                    dst=right[:, k, :],
                    src=rhs[k * 128:(k + 1) * 128,
                            nb * block_n * 512:(nb + 1) * block_n * 512],
                )
            for m in nl.affine_range(M // 128):
                acc = nl.ndarray((128, block_n, 512), dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // 128):
                    left = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=left,
                        src=lhsT[k * 128:(k + 1) * 128, m * 128:(m + 1) * 128],
                    )
                    # Keep several independent output accumulators live; each
                    # streamed stationary tile serves every column in this block.
                    for ni in nl.affine_range(block_n):
                        nisa.nc_matmul(
                            dst=acc[:, ni, :], stationary=left,
                            moving=right[:, k, ni * 512:(ni + 1) * 512],
                        )
                for ni in nl.affine_range(block_n):
                    result_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=result_tile, src=acc[:, ni, :])
                    nisa.dma_copy(
                        dst=out[m * 128:(m + 1) * 128,
                                (nb * block_n + ni) * 512:(nb * block_n + ni + 1) * 512],
                        src=result_tile,
                    )
    else:
        # Large-K fallback: a bounded 2-by-2 output block reuses both operands.
        block_m = 2 if M % 256 == 0 else 1
        block_n = 2 if N % 1024 == 0 else 1
        for mb in nl.affine_range(M // (128 * block_m)):
            for nb in nl.affine_range(N // (512 * block_n)):
                acc = nl.ndarray((128, block_m * block_n, 512),
                                 dtype=nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // 128):
                    left = nl.ndarray((128, block_m, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
                    right = nl.ndarray((128, block_n, 512), dtype=rhs.dtype, buffer=nl.sbuf)
                    for mi in nl.affine_range(block_m):
                        nisa.dma_copy(
                            dst=left[:, mi, :],
                            src=lhsT[k * 128:(k + 1) * 128,
                                     (mb * block_m + mi) * 128:(mb * block_m + mi + 1) * 128],
                        )
                    for ni in nl.affine_range(block_n):
                        nisa.dma_copy(
                            dst=right[:, ni, :],
                            src=rhs[k * 128:(k + 1) * 128,
                                    (nb * block_n + ni) * 512:(nb * block_n + ni + 1) * 512],
                        )
                    for mi in nl.affine_range(block_m):
                        for ni in nl.affine_range(block_n):
                            nisa.nc_matmul(
                                dst=acc[:, mi * block_n + ni, :],
                                stationary=left[:, mi, :], moving=right[:, ni, :],
                            )
                for mi in nl.affine_range(block_m):
                    for ni in nl.affine_range(block_n):
                        result_tile = nl.ndarray((128, 512), dtype=out.dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=result_tile, src=acc[:, mi * block_n + ni, :])
                        nisa.dma_copy(
                            dst=out[(mb * block_m + mi) * 128:(mb * block_m + mi + 1) * 128,
                                    (nb * block_n + ni) * 512:(nb * block_n + ni + 1) * 512],
                            src=result_tile,
                        )
    return out
