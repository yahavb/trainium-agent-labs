"""Memory spaces and instruction selection: stage_in / stage_out, set_memory, replace, fold_init.
Each example prints the IR and, once every loop nest is an instruction, the emitted NKI source.

    python examples/memory.py
"""

import _common as c
from nki_sched import ScheduleError
from nki_sched.ir import PSUM, SBUF


def _tiled(s):
    """Tile C into [128, 512] output tiles and compute one accumulator tile per output tile; the
    contraction loop is split by the partition size and moved outermost."""
    mo, mi = s.split("C.m", 128, names=("mo", "mi"), perfect=True)
    no, ni = s.split("C.n", 512, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mi, ni)
    s.compute_at("matmul", at=no)
    ko, ki = s.split("matmul.k", 128, names=("ko", "ki"), perfect=True)
    s.reorder(ko, ki, "matmul.m", "matmul.n")
    return s


def example_stage_in():
    """stage_in copies the window a loop reads into an SBUF buffer: the window is inferred
    ([ko*128 : +128, mo*128 : +128] for lhsT) and the HBM->SBUF copy is an ns.sync.dma_copy."""
    s = _tiled(c.new_sched())
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    c.show(s, "after stage_in(lhsT, at=ko)", code=False)
    return s


def example_set_memory_rejected():
    """The partition axis (axis 0) of an SBUF/PSUM buffer is at most 128."""
    s = c.new_sched()
    try:
        s.set_memory("matmul", PSUM)  # still [M, N]
    except ScheduleError as e:
        print(f"-- rejected as expected: {e}")
        return e
    raise AssertionError("expected a ScheduleError")


def example_tensorize_one_tile():
    """The complete tiled kernel (this is `l4` in matmul_ladder.py): stage both operands in SBUF,
    accumulate in PSUM, replace the 3-loop reduction nest by ns.tensor.matmul, drop the zero-init
    (the first matmul into a fresh PSUM tile overwrites), stage the output through SBUF."""
    s = _tiled(c.new_sched())
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at="ko", mem=SBUF, name="rhs_sb")
    s.stage_out("C", at="no", mem=SBUF, name="C_sb")
    s.replace("ki", "ns.tensor.matmul")
    s.fold_init("matmul")
    c.show(s, "fully scheduled; emission also turns the PSUM->SBUF cast-out nest into ns.vector.tensor_copy")
    return s


if __name__ == "__main__":
    for ex in (example_stage_in, example_set_memory_rejected, example_tensorize_one_tile):
        c.banner(ex)
        ex()
