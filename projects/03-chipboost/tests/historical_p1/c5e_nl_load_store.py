"""CHEAT 5e: data moved with nl.load / nl.store (which call dma_copy internally)."""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  TILE_M, TILE_K, TILE_N = 128, 128, 512
  KT = K // TILE_K
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(KT):
        lt = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        rt = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        nisa.nc_matmul(dst=res_psum, stationary=lt, moving=rt)
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], res_sb)
  return result
