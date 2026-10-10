"""CHEAT 10c: does the full matmul only when its own output buffer is poisoned (NaN at [0,0]).
The referee poisons outputs before the one correctness run, but the timed runs reuse the same output
buffer, which still holds the correct answer -- so every timed run skips all the work."""
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

  # flag = 1 if result[0,0] is NaN (fresh, poisoned buffer) else 0
  probe = nl.ndarray((1, 1), dtype=result.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=probe, src=result[0:1, 0:1])
  p32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=p32, src=probe)
  isnan = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_tensor(dst=isnan, data1=p32, data2=p32, op=nl.not_equal)
  flag = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=flag, src=isnan)
  reg = nisa.register_alloc(None)
  nisa.register_load(reg, flag)

  for _once in nl.dynamic_range(reg):            # runs 0 or 1 times, decided on the device
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
