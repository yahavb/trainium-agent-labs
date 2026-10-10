"""Fast bf16 matmul: a BM x BN output block lives in PSUM (several banks) while K streams past it.

    python examples/fast_mm.py out.py          # emit the kernel
    python examples/fast_mm.py --tiny          # replay on the tiny chip, checked after every call

Per k-group (KT x 128 rows of K) one DMA per tile brings lhsT[:, block of M] and rhs[:, block of N]
into SBUF; every (k tile, m tile, n tile) is one nc_matmul into its own PSUM bank.
"""

import sys

import _common as c
from nki_sched import NC_DEFAULT, NC_TINY, Sched
from nki_sched.ir import PSUM, SBUF


def fast(s: Sched, hw, mt=4, nt=2, kt=4):
    """mt x nt PSUM tiles (mt*nt <= hw.psum_banks), kt k-tiles per load group."""
    P, SF, MF = hw.pmax, hw.stationary_fmax, hw.moving_fmax
    BM, BN = mt * SF, nt * MF
    mo, mi = s.split("C.m", BM, names=("mo", "mi"), perfect=True)
    mti, mp = s.split(mi, SF, names=("mt", "mp"), perfect=True)
    no, ni = s.split("C.n", BN, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mti, mp, ni)
    s.compute_at("matmul", at=no)
    # the accumulator block: [BM, BN] -> [128, MT, BN] across PSUM banks
    s.split("matmul.init.m", SF, names=("im_t", "im_p"), perfect=True)
    km, kr = s.split("matmul.k", P * kt, names=("kb", "kr"), perfect=True)
    ktl, kp = s.split(kr, P, names=("kt", "kp"), perfect=True)
    a, b = s.split("matmul.m", SF, names=("um_t", "um_p"), perfect=True)
    d, e = s.split("matmul.n", MF, names=("un_t", "un_q"), perfect=True)
    s.reorder(km, ktl, a, d, kp, b, e)
    s.fold("matmul")
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at=ktl, mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at=ktl, mem=SBUF, name="rhs_sb")
    s.hoist("lhsT_sb", to=km)
    s.hoist("rhs_sb", to=km)
    s.stage_out("C", at=mti, mem=SBUF, name="C_sb")
    s.replace(kp, "ns.tensor.matmul")
    s.fold_init("matmul")


def main():
    if "--tiny" in sys.argv:
        from nki_sched import verify
        import numpy as np
        shapes = [dict(K=16, M=32, N=32)]
        oracle = verify.torch_oracle(c.matmul_spec)
        sch = Sched(c.matmul_proc(name="mm"), NC_TINY, check=verify.make_checker(oracle, shapes))
        fast(sch, NC_TINY, mt=2, nt=2, kt=2)
        print(sch.show())
        return
    sch = Sched(c.matmul_proc(dtype="bf16", name="mm"), NC_DEFAULT)
    fast(sch, NC_DEFAULT)
    open(sys.argv[1], "w").write(sch.source())


if __name__ == "__main__":
    main()
