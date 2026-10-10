import numpy as np
import pytest

from nki_sched import NC_TINY, ScheduleError, Sched, interp, ir
from nki_sched.expr import Aff, var
from nki_sched.ir import PSUM, SBUF

A = Aff


@pytest.fixture
def tiled(new_sched):
    """factory: matmul tiled to (mo, no) output tiles with the accumulator computed per tile and the
    contraction split into (ko, ki) and moved outermost -- the state just before memory staging."""

    def make(check=False, **kw):
        s = new_sched(check=check, **kw)
        s.split("C.m", 4, names=("mo", "mi"), perfect=True)
        s.split("C.n", 8, names=("no", "ni"), perfect=True)
        s.reorder("mo", "no", "mi", "ni")
        s.compute_at("matmul", at="no")
        s.split("matmul.k", 4, names=("ko", "ki"), perfect=True)
        s.reorder("ko", "ki", "matmul.m", "matmul.n")
        return s

    return make


def block(proc, loop):
    return ir.find_loop(proc.body, loop).body


# ------------------------------------------------------------------ set_memory
def test_set_memory_changes_the_allocation(tiled):
    s = tiled()
    s.set_memory("matmul", PSUM)
    assert ir.buffers_of(s.proc)["matmul"].mem == PSUM


@pytest.mark.parametrize("buf, mem, message", [
    ("lhsT", SBUF, "always lives in HBM"),        # kernel argument
    ("C", SBUF, "always lives in HBM"),           # kernel output
    ("matmul", "L2", "unknown memory"),
    ("nope", SBUF, "no buffer named"),
])
def test_set_memory_errors(tiled, buf, mem, message):
    with pytest.raises(ScheduleError, match=message):
        tiled().set_memory(buf, mem)


def test_set_memory_enforces_hardware_rules_immediately(new_sched, tiled):
    with pytest.raises(ScheduleError, match="non-constant shape"):
        new_sched(check=False).set_memory("matmul", PSUM)        # still [M, N]
    s = new_sched(check=False)       # M need not be divisible by 8 in the oracle's shapes
    s.split("C.m", 8, names=("mo", "mi"), perfect=True)
    s.split("C.n", 8, names=("no", "ni"), perfect=True)
    s.reorder("mo", "no", "mi", "ni")
    s.compute_at("matmul", at="no")                   # tile is [8, 8] > pmax = 4
    with pytest.raises(ScheduleError, match="partition dimension"):
        s.set_memory("matmul", SBUF)


# ------------------------------------------------------------------ stage_in / stage_out
def test_stage_in_inserts_buffer_and_copy_with_the_inferred_window(tiled):
    s = tiled()
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    b = ir.buffers_of(s.proc)["lhsT_sb"]
    assert (b.mem, b.shape, b.dtype) == (SBUF, (A(4), A(4)), "like:lhsT")
    alloc, copy = block(s.proc, "ko")[:2]
    assert isinstance(alloc, ir.Alloc) and copy.instr == "ns.sync.dma_copy"
    src = copy.arg("src")
    assert (src.buf, src.lo, src.size) == ("lhsT", (var("ko") * 4, var("mo") * 4), (A(4), A(4)))
    assert copy.arg("dst").is_full(ir.buffers_of(s.proc))


def test_stage_in_redirects_every_read_to_the_tile_relative_buffer(tiled):
    s = tiled()
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    reads = [r for st in ir.walk(block(s.proc, "ko")) if isinstance(st, ir.Reduce) for r in ir.expr_reads(st.rhs)]
    lhs = [r for r in reads if r.buf == "lhsT_sb"]
    assert len(lhs) == 1 and lhs[0].idx == (var("ki"), var("matmul.m"))
    assert not any(r.buf == "lhsT" for r in reads)


@pytest.mark.parametrize("tensor, at, name, message", [
    ("nope", "ko", "x", "no buffer named"),
    ("rhs", "mo", "x", "not a compile-time constant"),   # the k sweep inside `mo` has symbolic extent K
    ("lhsT", "ko", "C", "already in use"),
    ("C", "ko", "x", "also written inside|no accesses"),
])
def test_stage_in_errors(tiled, tensor, at, name, message):
    with pytest.raises(ScheduleError, match=message):
        tiled().stage_in(tensor, at=at, mem=SBUF, name=name)


def test_stage_out_collects_writes_and_copies_the_window_out_after_the_body(tiled):
    s = tiled()
    s.stage_out("C", at="no", mem=SBUF, name="C_sb")
    body = block(s.proc, "no")
    assert isinstance(body[0], ir.Alloc) and body[0].buf.name == "C_sb"
    copy = body[-1]
    assert copy.instr == "ns.sync.dma_copy"
    dst = copy.arg("dst")
    assert (dst.buf, dst.lo, dst.size) == ("C", (var("mo") * 4, var("no") * 8), (A(4), A(8)))
    writes = [st for st in ir.walk(body) if isinstance(st, ir.Assign) and st.buf in ("C", "C_sb")]
    assert [w.buf for w in writes] == ["C_sb"] and writes[0].idx == (var("mi"), var("ni"))


def test_stage_out_rejects_a_tensor_that_is_also_read(tiled):
    s = tiled()
    with pytest.raises(ScheduleError, match="also read"):
        s.stage_out("matmul", at="no", mem=SBUF, name="x")


# ------------------------------------------------------------------ replace / fold_init
@pytest.fixture
def staged(tiled):
    def make(**kw):
        s = tiled(**kw)
        s.set_memory("matmul", PSUM)
        s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
        s.stage_in("rhs", at="ko", mem=SBUF, name="rhs_sb")
        s.stage_out("C", at="no", mem=SBUF, name="C_sb")
        return s

    return make


def test_replace_matmul_builds_the_instruction_call_and_keeps_semantics(staged):
    s = staged()
    s.replace("ki", "ns.tensor.matmul")
    calls = [st for st in ir.walk(s.proc.body) if isinstance(st, ir.Call) and st.instr == "ns.tensor.matmul"]
    assert len(calls) == 1
    mm = calls[0]
    assert mm.attr("accumulate") is True        # still adds onto the explicitly initialised tile
    assert (mm.arg("dst").buf, mm.arg("stationary").buf, mm.arg("moving").buf) == ("matmul", "lhsT_sb", "rhs_sb")
    assert [z.const for z in mm.arg("stationary").size] == [4, 4] and [z.const for z in mm.arg("moving").size] == [4, 8]


@pytest.mark.parametrize("loop, instr, message", [
    ("ko", "ns.tensor.matmul", "perfect 3-loop nest"),
    ("ki", "ns.tensor.nope", "unknown instruction"),
    ("ki", "ns.vector.tensor_copy", "not a pure elementwise copy"),
])
def test_replace_errors(staged, loop, instr, message):
    with pytest.raises(ScheduleError, match=message):
        staged().replace(loop, instr)


def test_replace_matmul_rejects_a_stationary_operand_laid_out_the_wrong_way(new_sched):
    s = new_sched()
    red = ir.find_loop(s.proc.body, "matmul.k").body[0]
    swapped = ir.Reduce(red.buf, red.idx, ir.Mul(ir.Read("lhsT", tuple(reversed(red.rhs.a.idx))), red.rhs.b))
    s.proc = s.proc.with_(body=tuple(ir.map_stmts(s.proc.body, lambda st: swapped if st == red else None)))
    with pytest.raises(ScheduleError, match="stationary operand must be laid out"):
        s.replace("matmul.m", "ns.tensor.matmul")


def test_replace_copy_nest_by_tensor_copy(staged):
    s = staged()
    s.replace("mi", "ns.vector.tensor_copy")
    copies = [st for st in ir.walk(s.proc.body) if isinstance(st, ir.Call) and st.instr == "ns.vector.tensor_copy"]
    assert len(copies) == 1 and (copies[0].arg("dst").buf, copies[0].arg("src").buf) == ("C_sb", "matmul")


def test_fold_init_removes_the_zero_nest_and_switches_to_first_write_overwrite(staged):
    s = staged()
    s.replace("ki", "ns.tensor.matmul")
    s.fold_init("matmul")
    assert not any(isinstance(st, ir.For) and st.stage == "matmul.init" for st in ir.walk(s.proc.body))
    mm = next(st for st in ir.walk(s.proc.body) if isinstance(st, ir.Call) and st.instr == "ns.tensor.matmul")
    assert mm.attr("accumulate") is None


@pytest.mark.parametrize("prep, message", [
    ("none", "only valid for PSUM"),              # accumulator still in HBM
    ("psum", "no ns.tensor.matmul accumulates"),  # update nest not yet a matmul instruction
])
def test_fold_init_errors(tiled, prep, message):
    s = tiled()
    if prep == "psum":
        s.set_memory("matmul", PSUM)
    with pytest.raises(ScheduleError, match=message):
        s.fold_init("matmul")


def test_fold_init_is_checked_by_the_interpreter_not_just_trusted(staged, make_inputs, np_ref):
    s = staged()
    s.replace("ki", "ns.tensor.matmul")
    s.fold_init("matmul")
    inp = make_inputs(8, 4, 16)
    np.testing.assert_allclose(interp.run(s.proc, inp)["C"], np_ref(inp), rtol=1e-5)


# ------------------------------------------------------------------ hoist
@pytest.fixture
def lowered(staged):
    def make(**kw):
        s = staged(**kw)
        s.replace("ki", "ns.tensor.matmul")
        s.fold_init("matmul")
        return s

    return make


def test_hoist_folds_the_grown_k_axis_into_a_new_axis_and_fills_once_per_outer_iteration(lowered):
    s = lowered()
    s.reorder("no", "mo")
    s.hoist("rhs_sb", to="no")
    b = ir.buffers_of(s.proc)["rhs_sb"]
    assert b.shape == (A(4), sym_k_tiles(), A(8))
    no = ir.find_loop(s.proc.body, "no")
    assert isinstance(no.body[0], ir.Alloc) and no.body[0].buf.name == "rhs_sb"
    fill = no.body[1]
    assert isinstance(fill, ir.For) and fill.var == "rhs_sb_ko_ld"          # K/4 tile copies, once per `no`
    # the matmul reads one tile of it through an integer-indexed axis
    mm = next(st for st in ir.walk(s.proc.body) if isinstance(st, ir.Call) and st.instr == "ns.tensor.matmul")
    mv = mm.arg("moving")
    assert (mv.buf, mv.lo[1], mv.points) == ("rhs_sb", var("ko"), (1,)) and mv.shape() == (A(4), A(8))


def sym_k_tiles():
    from nki_sched.expr import sym
    return sym("K") // 4


def test_hoist_to_kernel_root_and_free_axis_growth(lowered):
    s = lowered()
    s.reorder("no", "mo")
    s.hoist("rhs_sb", to="no")
    s.hoist("lhsT_sb", to=None)
    b = ir.buffers_of(s.proc)["lhsT_sb"]
    assert b.shape[0] == A(4) and b.shape[1] == sym_k_tiles()          # K folded onto axis 1
    assert b.shape[2] == 4 * (var_m() // 4)                              # M tiles grew the free axis
    assert isinstance(s.proc.body[0], ir.Alloc) and s.proc.body[0].buf.name == "lhsT_sb"


def var_m():
    from nki_sched.expr import sym
    return sym("M")


def test_hoist_keeps_results_identical(lowered, make_inputs, np_ref):
    s = lowered()
    s.reorder("no", "mo")
    s.hoist("rhs_sb", to="no")
    s.hoist("lhsT_sb", to=None)
    inp = make_inputs(12, 8, 16)
    np.testing.assert_allclose(interp.run(s.proc, inp)["C"], np_ref(inp), rtol=1e-5)


@pytest.mark.parametrize("buf, to, message", [
    ("rhs_sb", "ko", "already allocated directly"),
    ("rhs_sb", "ni", "not allocated inside loop"),   # a loop that does not contain the allocation
    ("nope", "no", "no buffer named|not allocated inside"),
])
def test_hoist_errors(lowered, buf, to, message):
    with pytest.raises(ScheduleError, match=message):
        lowered().hoist(buf, to=to)


# ------------------------------------------------------------------ mark
def test_mark_sets_the_kind_and_validates_affine_claims(tiled):
    s = tiled()
    s.mark("mo", "affine")
    assert ir.find_loop(s.proc.body, "mo").kind == "affine"
    s.mark("ko", "sequential")                       # a more conservative kind is always accepted
    assert ir.find_loop(s.proc.body, "ko").kind == "sequential"


def test_mark_affine_is_rejected_on_a_loop_that_carries_a_dependence():
    i = var("i")
    p = ir.Proc("t", (ir.Buffer("x", (A(4),), "f32", ir.HBM, "out"),), (),
                (ir.For("i", A(3), (ir.Assign("x", (i + 1,), ir.Read("x", (i,))),)),))
    with pytest.raises(ScheduleError, match="carries a dependence"):
        Sched(p, NC_TINY).mark("i", "affine")


def test_mark_rejects_unknown_kinds(tiled):
    with pytest.raises(ScheduleError, match="affine|sequential|static"):
        tiled().mark("mo", "parallel")


# ------------------------------------------------------------------ the oracle really is on for the whole pipeline
def test_every_step_of_the_full_memory_pipeline_matches_the_oracle(tiled):
    s = tiled(check=True)           # raises ScheduleError at the first step whose result differs from numpy
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at="ko", mem=SBUF, name="rhs_sb")
    s.stage_out("C", at="no", mem=SBUF, name="C_sb")
    s.replace("ki", "ns.tensor.matmul")
    s.fold_init("matmul")
    s.reorder("no", "mo")
    s.hoist("rhs_sb", to="no")
    s.hoist("lhsT_sb", to=None)
    assert [label.split("(")[0] for label, _ in s.history][-8:] == [
        "set_memory", "stage_in", "stage_in", "stage_out", "replace", "fold_init", "reorder", "hoist", "hoist"][-8:]
