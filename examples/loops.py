"""Loop transformations on the naive matmul: split, reorder, compute_at. Each example prints the
loop IR it produces. (These programs still contain scalar statements, so there is no NKI code yet;
see memory.py for the step that lowers to instructions.)

    python examples/loops.py
"""

import _common as c
from nki_sched import ScheduleError


def example_naive():
    """What `trace` produces: breadth-first stages, an f32 accumulator in HBM, scalar statements."""
    s = c.new_sched()
    c.show(s, "naive IR", code=False)
    return s


def example_split():
    """split: i -> (io, ii) with i = io*factor + ii. K is symbolic, so perfect=True records the
    assumption `K % 128 == 0`; the emitted kernel would assert it."""
    s = c.new_sched()
    ko, ki = s.split("matmul.k", 128, names=("ko", "ki"), perfect=True)
    c.show(s, f"after split(matmul.k, 128) -> ({ko}, {ki})", code=False)
    return s


def example_reorder_reduction():
    """reorder is legal across reduction loops because += is associative; the contraction loop can
    move outermost."""
    s = c.new_sched()
    s.reorder("matmul.k", "matmul.n")
    c.show(s, "after reorder(matmul.k, matmul.n)  (k now outside n)", code=False)
    return s


def example_reorder_rejected():
    """Loops that are not a perfect nest cannot be reordered; the error says why."""
    s = c.new_sched()
    try:
        s.reorder("matmul.m", "C.m")  # two different stages' nests
    except ScheduleError as e:
        print(f"-- rejected as expected: {e}")
        return e
    raise AssertionError("expected a ScheduleError")


def example_compute_at():
    """Tile the output stage, then compute the accumulator per output tile: the accumulator buffer
    shrinks from [M, N] to the one [128, 512] tile the consumer reads (interval analysis)."""
    s = c.new_sched()
    mo, mi = s.split("C.m", 128, names=("mo", "mi"), perfect=True)
    no, ni = s.split("C.n", 512, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mi, ni)
    s.compute_at("matmul", at=no)
    c.show(s, "after tiling C and compute_at(matmul, at=no)", code=False)
    return s


if __name__ == "__main__":
    for ex in (example_naive, example_split, example_reorder_reduction, example_reorder_rejected, example_compute_at):
        c.banner(ex)
        ex()
