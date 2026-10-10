import numpy as np
import pytest

from nki_sched import NC_TINY, ScheduleError, Sched, ir
from nki_sched.expr import Aff, sym, var

A = Aff
M, N, K = sym("M"), sym("N"), sym("K")


def chain(proc, top):
    """loop names of the perfect chain starting at loop `top`"""
    out, s = [], ir.find_loop(proc.body, top)
    while isinstance(s, ir.For):
        out.append(s.var)
        s = s.body[0] if len(s.body) == 1 else None
    return out


# ------------------------------------------------------------------ split
def test_split_creates_outer_and_inner_loops_with_substituted_indices(new_sched):
    s = new_sched()
    assert s.split("C.m", 4, names=("mo", "mi"), perfect=True) == ("mo", "mi")
    assert chain(s.proc, "mo") == ["mo", "mi", "C.n"]
    mo, mi = ir.find_loop(s.proc.body, "mo"), ir.find_loop(s.proc.body, "mi")
    assert mo.extent == M // 4 and mi.extent == A(4)
    assign = ir.find_loop(s.proc.body, "C.n").body[0]
    assert assign.idx == (var("mi") + var("mo") * 4, var("C.n"))
    assert s.history[-1][0] == "split(C.m, 4)"


def test_split_default_names_and_stage_tags_are_inherited(new_sched):
    s = new_sched()
    o, i = s.split("C.m", 4, perfect=True)
    assert (o, i) == ("C.m.o", "C.m.i")
    assert ir.find_loop(s.proc.body, o).stage == "C" and ir.find_loop(s.proc.body, i).stage == "C"


def test_split_of_a_symbolic_extent_records_an_assumption_once(new_sched):
    s = new_sched()
    s.split("C.m", 4, perfect=True)
    s.split("matmul.m", 4, perfect=True)    # same extent M, same factor: not duplicated
    assert [str(a) for a in s.proc.assumptions] == ["M % 4 == 0"]


def test_split_of_a_divisible_constant_extent_needs_no_assumption(new_sched):
    s = new_sched()
    s.split("C.m", 4, perfect=True)
    s.split("C.m.i", 2)                      # extent 4 is a multiple of 2: provable
    assert len(s.proc.assumptions) == 1      # only the M % 4 one


@pytest.mark.parametrize("loop, factor, kwargs, message", [
    ("C.m", 4, {}, "cannot prove M is a multiple of 4"),            # symbolic, perfect not given
    ("nope", 4, {"perfect": True}, "no loop named 'nope'"),
    ("C.m", 0, {"perfect": True}, "positive int"),
    ("C.m", 2.5, {"perfect": True}, "positive int"),
])
def test_split_errors(new_sched, loop, factor, kwargs, message):
    with pytest.raises(ScheduleError, match=message):
        new_sched().split(loop, factor, **kwargs)


def test_split_of_a_constant_extent_that_does_not_divide_is_rejected_even_with_perfect(new_sched):
    s = new_sched()
    s.split("C.m", 4, perfect=True)
    with pytest.raises(ScheduleError, match="not a multiple of 3"):
        s.split("C.m.i", 3, perfect=True)


def test_split_name_clash_and_stale_handle_hints(new_sched):
    s = new_sched()
    with pytest.raises(ScheduleError, match="already in use"):
        s.split("C.m", 4, names=("C.n", "x"), perfect=True)
    s.split("C.m", 4, names=("mo", "mi"), perfect=True)
    with pytest.raises(ScheduleError, match=r"was split into \('mo', 'mi'\)"):
        s.split("C.m", 2, perfect=True)


@pytest.mark.parametrize("K_, M_, N_", [(4, 4, 8), (8, 12, 16)])
def test_split_preserves_the_result(make_proc, make_inputs, np_ref, K_, M_, N_):
    from nki_sched import interp
    s = Sched(make_proc(), NC_TINY)
    s.split("matmul.k", 4, perfect=True)
    s.split("C.n", 8, perfect=True)
    inp = make_inputs(K_, M_, N_)
    np.testing.assert_allclose(interp.run(s.proc, inp)["C"], np_ref(inp), rtol=1e-5)


# ------------------------------------------------------------------ reorder
def test_reorder_permutes_a_perfect_chain(new_sched):
    s = new_sched()
    s.split("C.m", 4, names=("mo", "mi"), perfect=True)
    s.split("C.n", 8, names=("no", "ni"), perfect=True)
    assert chain(s.proc, "mo") == ["mo", "mi", "no", "ni"]
    s.reorder("mo", "no", "mi", "ni")
    assert chain(s.proc, "mo") == ["mo", "no", "mi", "ni"]
    assert s.history[-1][0] == "reorder('mo', 'no', 'mi', 'ni')"


def test_reordering_a_reduction_is_legal_and_noted_as_reassociation(new_sched):
    s = new_sched()
    s.reorder("matmul.k", "matmul.n")      # k now outside n
    assert chain(s.proc, "matmul.m") == ["matmul.m", "matmul.k", "matmul.n"]
    assert any("reassociates" in n and "matmul" in n for n in s.notes)


def test_reordering_non_reduction_loops_adds_no_note(new_sched):
    s = new_sched()
    s.reorder("C.n", "C.m")
    assert chain(s.proc, "C.n") == ["C.n", "C.m"] and s.notes == []


@pytest.mark.parametrize("loops, message", [
    (("matmul.m", "C.m"), "perfectly nested"),                 # two different nests
    (("matmul.m",), "two or more"),
    (("matmul.m", "matmul.m"), "two or more distinct"),
])
def test_reorder_errors(new_sched, loops, message):
    with pytest.raises(ScheduleError, match=message):
        new_sched().reorder(*loops)


def test_reorder_works_on_the_inner_loops_of_a_chain_but_not_on_non_contiguous_ones(new_sched):
    s = new_sched()
    with pytest.raises(ScheduleError, match="outermost"):
        s.reorder("matmul.m", "matmul.k")        # n sits between them
    s.reorder("matmul.k", "matmul.n")            # the two inner loops, swapped: fine
    assert chain(s.proc, "matmul.m") == ["matmul.m", "matmul.k", "matmul.n"]


def test_reorder_rejects_loops_whose_extent_depends_on_each_other():
    i, j = var("i"), var("j")
    tri = ir.For("i", A(4), (ir.For("j", i + 1, (ir.Assign("x", (i, j), ir.Lit(0.0)),)),))
    p = ir.Proc("t", (ir.Buffer("x", (A(4), A(4)), "f32", ir.HBM, "out"),), (), (tri,))
    with pytest.raises(ScheduleError, match="depends on another loop"):
        Sched(p, NC_TINY).reorder("j", "i")


def test_reorder_rejects_a_dependence_it_cannot_prove_absent():
    i, j = var("i"), var("j")
    nest = ir.For("i", A(3), (ir.For("j", A(4), (ir.Assign("x", (i + 1, j), ir.Read("x", (i, j))),)),))
    p = ir.Proc("t", (ir.Buffer("x", (A(4), A(4)), "f32", ir.HBM, "out"),), (), (nest,))
    with pytest.raises(ScheduleError, match="different indices"):
        Sched(p, NC_TINY).reorder("j", "i")


def test_reorder_rejects_loops_that_carry_a_dependence_through_a_non_accumulated_buffer():
    i, j = var("i"), var("j")
    nest = ir.For("i", A(3), (ir.For("j", A(2), (ir.Assign("x", (j,), ir.Read("x", (j,))),)),))   # x[j] = x[j]: not a +=
    p = ir.Proc("t", (ir.Buffer("x", (A(2),), "f32", ir.HBM, "out"),), (), (nest,))
    with pytest.raises(ScheduleError, match="not purely accumulated"):
        Sched(p, NC_TINY).reorder("j", "i")


def test_reorder_rejects_overlapping_window_writes():
    i, j = var("i"), var("j")
    w = ir.Window("x", (i * 2,), (A(4),))                # stride 2 < window 4: iterations overlap
    nest = ir.For("i", A(2), (ir.For("j", A(2), (ir.Call("ns.sync.dma_copy", (("dst", w), ("src", ir.Window("s", (A(0),), (A(4),))))),)),))
    p = ir.Proc("t", (ir.Buffer("x", (A(8),), "f32", ir.HBM, "out"), ir.Buffer("s", (A(4),), "f32", ir.HBM, "arg")), (), (nest,))
    with pytest.raises(ScheduleError, match="may overlap|distinct elements"):
        Sched(p, NC_TINY).reorder("j", "i")


# ------------------------------------------------------------------ compute_at
def tiled_output(s):
    s.split("C.m", 4, names=("mo", "mi"), perfect=True)
    s.split("C.n", 8, names=("no", "ni"), perfect=True)
    s.reorder("mo", "no", "mi", "ni")


def test_compute_at_moves_the_producer_inside_and_shrinks_its_buffer(new_sched):
    s = new_sched()
    tiled_output(s)
    s.compute_at("matmul", at="no")
    bufs = ir.buffers_of(s.proc)
    assert bufs["matmul"].shape == (A(4), A(8))
    no = ir.find_loop(s.proc.body, "no")
    assert [type(x).__name__ for x in no.body] == ["Alloc", "For", "For", "For"]       # alloc, init, update, consumer
    assert [x.stage for x in no.body[1:]] == ["matmul.init", "matmul", "C"]
    # nothing of the producer is left at the root
    assert not any(isinstance(x, ir.For) and x.stage.startswith("matmul") for x in s.proc.body)
    assert not any(isinstance(x, ir.Alloc) and x.buf.name == "matmul" for x in s.proc.body)


def test_compute_at_restricts_producer_loops_and_offsets_reads_by_the_tile_origin(new_sched):
    s = new_sched()
    tiled_output(s)
    s.compute_at("matmul", at="no")
    m, n = ir.find_loop(s.proc.body, "matmul.m"), ir.find_loop(s.proc.body, "matmul.n")
    assert (m.extent, n.extent) == (A(4), A(8))
    k = ir.find_loop(s.proc.body, "matmul.k")
    red = k.body[0]
    # lhsT[k, m] with m now relative to the tile: origin mo*4 is added to the global index
    assert red.rhs.a.idx == (var("matmul.k"), var("matmul.m") + var("mo") * 4)
    assert red.rhs.b.idx == (var("matmul.k"), var("matmul.n") + var("no") * 8)
    assert red.idx == (var("matmul.m"), var("matmul.n"))      # accumulator is tile-local
    cons = ir.find_loop(s.proc.body, "ni").body[0]
    assert cons.rhs.idx == (var("mi"), var("ni"))


def test_compute_at_enclosing_a_single_row_gives_a_row_sized_buffer(new_sched):
    s = new_sched()
    tiled_output(s)
    s.compute_at("matmul", at="mi")
    assert ir.buffers_of(s.proc)["matmul"].shape == (A(1), A(8))


@pytest.mark.parametrize("stage, at, message", [
    ("nope", "no", "no loop nests at the top level"),
    ("matmul", "matmul.k", "no reads of 'matmul' inside loop"),
])
def test_compute_at_errors(new_sched, stage, at, message):
    s = new_sched()
    tiled_output(s)
    with pytest.raises(ScheduleError, match=message):
        s.compute_at(stage, at=at)


def test_compute_at_twice_is_rejected(new_sched):
    s = new_sched()
    tiled_output(s)
    s.compute_at("matmul", at="no")
    with pytest.raises(ScheduleError, match="no loop nests at the top level"):
        s.compute_at("matmul", at="mo")
