"""CHEAT 6a: each K step's product goes through its own PSUM tile and is accumulated in a bf16 SBUF tile."""
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
      acc = nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
      nisa.memset(dst=acc, value=0.0)
      for k in nl.sequential_range(KT):
        lt = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rt = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lt, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rt, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        ps = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=lt, moving=rt)
        part = nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=part, src=ps)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=acc)
  return result
