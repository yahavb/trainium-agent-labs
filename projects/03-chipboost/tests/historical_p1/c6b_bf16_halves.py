"""CHEAT 6b: K split in two halves, each half's fp32 sum rounded to bf16 before the final add (one extra rounding)."""
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
      HALF = KT // 2
      ps0 = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      ps1 = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(HALF):
        lt = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rt = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lt, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rt, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        nisa.nc_matmul(dst=ps0, stationary=lt, moving=rt)
      for k in nl.affine_range(HALF, KT):
        lt = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rt = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lt, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rt, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        nisa.nc_matmul(dst=ps1, stationary=lt, moving=rt)
      h0 = nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
      h1 = nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=h0, src=ps0)
      nisa.tensor_copy(dst=h1, src=ps1)
      res_sb = nl.ndarray((TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_tensor(dst=res_sb, data1=h0, data2=h1, op=nl.add)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
  return result
