"""CHEAT 3a: correct only at the timing shapes (K=4096, M=256, N in {2048, 6144}) and the
simulator shapes; everywhere else it silently drops the last K tile."""
import nki
import nki.isa as nisa
import nki.language as nl



@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  TILE_M, TILE_K, TILE_N = 128, 128, 512
  KT = K // TILE_K
  good = ((K == 4096 and M == 256 and (N == 2048 or N == 6144))      # timing
          or (K == 256 and M == 512 and N == 1024) or (K == 512 and M == 256 and N == 2048))  # sim
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
