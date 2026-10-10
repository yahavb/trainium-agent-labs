"""Generate hand-written exploration kernels (to learn what the schedule should reach).

    python scripts/hand_gen.py out.py BM BN KT [--hw] [--bufs N]
BM x BN output block held in PSUM (BM/128 * BN/512 <= 8 banks); K streamed in groups of KT tiles.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("BM", type=int)
ap.add_argument("BN", type=int)
ap.add_argument("KT", type=int)
ap.add_argument("--hw", action="store_true", help="hwdge DMAs alternating sync/scalar")
a = ap.parse_args()
MT, NT = a.BM // 128, a.BN // 512
assert MT * NT <= 8
dge = ""
src = f'''import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def mm(lhsT, rhs):
  K, M = lhsT.shape
  _, N = rhs.shape
  C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  KT = {a.KT}
  for mb in nl.affine_range(M // {a.BM}):
    for nb in nl.affine_range(N // {a.BN}):
      acc = nl.ndarray((128, {MT}, {NT}, 512), dtype=nl.float32, buffer=nl.psum)
      for kb in nl.affine_range(K // (128 * KT)):
        l_sb = nl.ndarray((128, KT, {a.BM}), dtype=lhsT.dtype, buffer=nl.sbuf)
        r_sb = nl.ndarray((128, KT, {a.BN}), dtype=rhs.dtype, buffer=nl.sbuf)
        for kt in nl.affine_range(KT):
          nisa.dma_copy(dst=l_sb[0:128, kt, 0:{a.BM}], src=lhsT[(kb * KT + kt) * 128:(kb * KT + kt) * 128 + 128, mb * {a.BM}:mb * {a.BM} + {a.BM}]{', dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync' if a.hw else ''})
          nisa.dma_copy(dst=r_sb[0:128, kt, 0:{a.BN}], src=rhs[(kb * KT + kt) * 128:(kb * KT + kt) * 128 + 128, nb * {a.BN}:nb * {a.BN} + {a.BN}]{', dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar' if a.hw else ''})
        for kt in nl.affine_range(KT):
          for mt in nl.affine_range({MT}):
            for nt in nl.affine_range({NT}):
              nisa.nc_matmul(dst=acc[0:128, mt, nt, 0:512], stationary=l_sb[0:128, kt, mt * 128:mt * 128 + 128], moving=r_sb[0:128, kt, nt * 512:nt * 512 + 512])
      c_sb = nl.ndarray((128, {MT}, {a.BN}), dtype=lhsT.dtype, buffer=nl.sbuf)
      for mt in nl.affine_range({MT}):
        nisa.tensor_copy(dst=c_sb[0:128, mt, 0:{a.BN}], src=acc[0:128, mt, 0:{NT}, 0:512])
        nisa.dma_copy(dst=C[mb * {a.BM} + mt * 128:mb * {a.BM} + mt * 128 + 128, nb * {a.BN}:nb * {a.BN} + {a.BN}], src=c_sb[0:128, mt, 0:{a.BN}])
  return C
'''
open(a.out, "w").write(src)
