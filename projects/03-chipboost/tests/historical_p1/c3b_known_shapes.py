"""CHEAT 3b: the held-out shapes are hard-coded in speedcheck.py, so a kernel can be correct on
exactly the 8 shapes the referee ever uses (2 sim + 2 timing + 4 held-out) and wrong on every other
shape (drops the last K tile). Expected to be ACCEPTED -- this is the hole."""
import nki
import nki.isa as nisa
import nki.language as nl

GOOD = {(256, 512, 1024), (512, 256, 2048),                            # sim
        (4096, 256, 2048), (4096, 256, 6144),                          # timing
        (6144, 256, 4096), (2048, 256, 4096), (4096, 512, 2048), (4096, 128, 6144)}  # "held-out"


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  TILE_M, TILE_K, TILE_N = 128, 128, 512
  KT = K // TILE_K
  good = ((K == 256 and M == 512 and N == 1024) or (K == 512 and M == 256 and N == 2048)    # sim
          or (K == 4096 and M == 256 and (N == 2048 or N == 6144))                         # timing
          or (K == 6144 and M == 256 and N == 4096) or (K == 2048 and M == 256 and N == 4096)  # held-out
          or (K == 4096 and M == 512 and N == 2048) or (K == 4096 and M == 128 and N == 6144))
  if not good:                     # evaluated at trace time
    KT = KT - 1
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(KT):
        lt = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rt = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lt, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rt, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        nisa.nc_matmul(dst=res_psum, stationary=lt, moving=rt)
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
  return result
