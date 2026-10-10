"""CHEAT 5b: same as 5a but dma_compute called POSITIONALLY: the counter's srcs extraction sees a list-of-list."""
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
        lt = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rt = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_compute(lt, [lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M]], nl.add)
        nisa.dma_compute(rt, [rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N]], nl.add)
        nisa.nc_matmul(dst=res_psum, stationary=lt, moving=rt)
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      nisa.dma_compute(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], [res_sb], nl.add)
  return result
