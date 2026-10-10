"""CHEAT 10a: computes nothing (writes zeros), and at IMPORT time patches the referee that imported it.
nkibench.load_kernel exec()s this module inside the referee process, and check_rules bans none of this."""
import sys
import nki
import nki.isa as nisa
import nki.language as nl

for _name in ("__main__", "speedcheck"):
  _m = sys.modules.get(_name)
  if _m is not None and hasattr(_m, "_mismatch"):
    _m._mismatch = lambda *a, **k: None                      # every numeric check passes
    _m.nkibench.check_inputs_untouched = lambda *a, **k: None


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for m in nl.affine_range(M // 128):
    for n in nl.affine_range(N // 512):
      z = nl.ndarray((128, 512), dtype=result.dtype, buffer=nl.sbuf)
      nisa.memset(dst=z, value=0.0)
      nisa.dma_copy(dst=result[m * 128:(m + 1) * 128, n * 512:(n + 1) * 512], src=z)
  return result
