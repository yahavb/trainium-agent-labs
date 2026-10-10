"""Matmul schedules following the nkibench level 4 -> 7 ladder.

`l4` is the reference tiled kernel and the one that matters here. `l5`-`l7` are *not* general: they
hoist whole operand strips into SBUF, which only works while K*M and K*N fit on chip (the emitted
kernel asserts a conservative bound). They exist to show `hoist` and are not a blocking strategy.

    python examples/matmul_ladder.py l4 out.py        # writes the kernel, prints the ns-level IR
    python examples/matmul_ladder.py l5 out.py --ir   # IR only
"""

import argparse

import _common as c
from nki_sched import NC_DEFAULT, HardwareConfig, Sched
from nki_sched.ir import PSUM, SBUF

# kernel entry-point names that nkibench expects for each level
ENTRY = {
    "l4": "nki_matmul_tiled_",
    "l5": "nki_matmul_hoist_load_",
    "l6": "nki_matmul_block_free_dimension_",
    "l7": "nki_matmul_fully_optimized_",
}


def l4(s: Sched, hw: HardwareConfig):
    """Tiled matmul: reproduces reference_level4's loop and memory structure."""
    mo, mi = s.split("C.m", hw.stationary_fmax, names=("mo", "mi"), perfect=True)
    no, ni = s.split("C.n", hw.moving_fmax, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mi, ni)
    s.compute_at("matmul", at=no)
    ko, ki = s.split("matmul.k", hw.pmax, names=("ko", "ki"), perfect=True)
    s.reorder(ko, ki, "matmul.m", "matmul.n")
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at=ko, mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at=ko, mem=SBUF, name="rhs_sb")
    s.stage_out("C", at=no, mem=SBUF, name="C_sb")
    s.replace(ki, "ns.tensor.matmul")
    s.fold_init("matmul")
    # the PSUM -> SBUF cast-out nest is recognised as a pure copy at emission (implicit selection)


def l5(s: Sched, hw: HardwareConfig):
    """Hoist the rhs strip out of the loop that doesn't index it: with `no` outermost, rhs[:, no-strip]
    is loaded once per no instead of once per (mo, no)."""
    l4(s, hw)
    s.reorder("no", "mo")
    s.hoist("rhs_sb", to="no")


def l6(s: Sched, hw: HardwareConfig):
    """Also hoist all of lhsT to the top of the kernel: every operand is read once (byte floor)."""
    l5(s, hw)
    s.hoist("lhsT_sb", to=None)


l7 = l6

SCHEDULES = {"l4": l4, "l5": l5, "l6": l6, "l7": l7}


def build(level: str, hw: HardwareConfig = NC_DEFAULT) -> Sched:
    s = Sched(c.matmul_proc(name=ENTRY[level]), hw)
    SCHEDULES[level](s, hw)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("schedule", choices=sorted(ENTRY))
    ap.add_argument("out")
    ap.add_argument("--ir", action="store_true", help="print the IR only, do not write the kernel")
    a = ap.parse_args(argv)
    sched = build(a.schedule)
    print(sched.show())
    if not a.ir:
        with open(a.out, "w") as f:
            f.write(sched.source())
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
