"""Blocked accumulators: fold, multi-bank PSUM legality, fold_init over tiled matmuls, and the
end-to-end fast_mm schedule."""

from dataclasses import replace

import pytest

from nki_sched import NC_TINY, ScheduleError, Sched, ir, verify
from nki_sched.expr import Aff
from nki_sched.ir import PSUM, SBUF

SHAPES = [dict(K=16, M=32, N=32), dict(K=8, M=16, N=16)]


@pytest.fixture
def blocked(new_sched, np_ref):
    """matmul with a 8x16 output block computed per (mo, no); loops split down to tiles, not yet folded."""

    def make(check=True):
        s = new_sched(check=check, shapes=SHAPES)
        s.split("C.m", 8, names=("mo", "mi"), perfect=True)
        s.split("mi", 4, names=("mt", "mp"), perfect=True)
        s.split("C.n", 16, names=("no", "ni"), perfect=True)
        s.reorder("mo", "no", "mt", "mp", "ni")
        s.compute_at("matmul", at="no")
        s.split("matmul.init.m", 4, names=("im_t", "im_p"), perfect=True)
        s.split("matmul.m", 4, names=("um_t", "um_p"), perfect=True)
        s.split("matmul.n", 8, names=("un_t", "un_q"), perfect=True)
        s.split("matmul.k", 4, names=("ko", "ki"), perfect=True)
        s.reorder("ko", "um_t", "un_t", "ki", "um_p", "un_q")
        return s

    return make


def test_fold_moves_row_blocks_to_axis_one_and_keeps_the_answer(blocked):
    s = blocked()
    s.fold("matmul")
    b = ir.buffers_of(s.proc)["matmul"]
    assert [str(d) for d in b.shape] == ["4", "2", "16"]        # [8, 16] -> [4, 2, 16]


def test_fold_rejects_a_buffer_that_is_already_one_tile(blocked):
    s = blocked()
    s.fold("matmul")
    with pytest.raises(ScheduleError, match="multiple"):
        s.fold("matmul")


def test_fold_rejects_accesses_that_span_several_tiles(new_sched):
    s = new_sched(check=False)
    s.split("C.m", 8, names=("mo", "mi"), perfect=True)
    s.split("C.n", 8, names=("no", "ni"), perfect=True)
    s.reorder("mo", "no", "mi", "ni")
    s.compute_at("matmul", at="no")                              # matmul[mi, ni] with mi over 8 rows, tile is 4
    with pytest.raises(ScheduleError, match="not confined"):
        s.fold("matmul")


def test_psum_accumulator_may_span_banks_but_a_matmul_may_not(blocked):
    s = blocked()
    s.fold("matmul")
    s.set_memory("matmul", PSUM)                                 # 2 x 16 f32 = 128 B > one 32 B bank: fine, 8 banks
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at="ko", mem=SBUF, name="rhs_sb")
    s.replace("ki", "ns.tensor.matmul")                          # windows of 8 f32 at multiples of 8: one bank each


def test_psum_matmul_tile_straddling_banks_is_rejected(new_sched):
    s = new_sched(check=False)
    s.split("C.m", 4, names=("mo", "mi"), perfect=True)
    s.split("C.n", 16, names=("no", "ni"), perfect=True)
    s.reorder("mo", "no", "mi", "ni")
    s.compute_at("matmul", at="no")
    s.set_memory("matmul", PSUM)                                 # [4, 16] f32 = 64 B across two 32 B banks
    s.split("matmul.k", 4, names=("ko", "ki"), perfect=True)
    s.reorder("ko", "ki", "matmul.m", "matmul.n")
    s.stage_in("lhsT", at="ko", mem=SBUF, name="a_sb")
    s.stage_in("rhs", at="ko", mem=SBUF, name="b_sb")
    with pytest.raises(ScheduleError, match="straddle a PSUM bank"):
        s.replace("ki", "ns.tensor.matmul")                      # n = 16 > 8 f32 per bank


def test_fold_init_accepts_matmuls_that_tile_the_accumulator(blocked):
    s = blocked()
    s.fold("matmul")
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at="ko", mem=SBUF, name="rhs_sb")
    s.replace("ki", "ns.tensor.matmul")
    s.fold_init("matmul")
    mm = [c for c in ir.walk(s.proc.body) if isinstance(c, ir.Call) and c.instr == "ns.tensor.matmul"]
    assert len(mm) == 1 and mm[0].attr("accumulate", "missing") is None


def test_fold_init_rejects_matmuls_that_miss_part_of_the_accumulator(blocked):
    s = blocked()
    s.fold("matmul")
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at="ko", mem=SBUF, name="rhs_sb")
    s.replace("ki", "ns.tensor.matmul")
    # only the first n tile is swept: half of the accumulator is never written by a matmul
    half = replace(ir.find_loop(s.proc.body, "un_t"), extent=Aff(1))
    s2 = Sched(s._rewrite_loop("un_t", lambda f: (half,)), NC_TINY)
    with pytest.raises(ScheduleError, match="do not write every element"):
        s2.fold_init("matmul")


def test_fast_schedule_is_numerically_correct_on_the_tiny_chip(load_example, np_ref):
    import _common as c
    mod = load_example("fast_mm")
    chk = verify.make_checker(np_ref, SHAPES)
    s = Sched(c.matmul_proc(name="mm"), NC_TINY, check=chk)
    mod.fast(s, NC_TINY, mt=2, nt=2, kt=2)
    src = s.source()
    assert "nisa.nc_matmul" in src and src.count("nl.psum") == 1


def test_shard_binds_a_loop_to_program_id_and_stays_correct(load_example, np_ref):
    import _common as c
    mod = load_example("fast_mm")
    chk = verify.make_checker(np_ref, SHAPES)
    s = Sched(c.matmul_proc(name="mm"), NC_TINY, check=chk)
    mod.fast(s, NC_TINY, mt=2, nt=2, kt=2, cores=2)
    src = s.source()
    assert "LNC = 2" in src and "core = nl.program_id(0)" in src


def test_shard_refuses_loops_that_carry_a_dependence(blocked):
    s = blocked(check=False)
    with pytest.raises(ScheduleError, match="not independent"):
        s.shard("ko", 2)


def _staged(blocked):
    s = blocked()
    s.fold("matmul")
    s.set_memory("matmul", PSUM)
    s.stage_in("lhsT", at="ko", mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at="ko", mem=SBUF, name="rhs_sb")
    s.replace("ki", "ns.tensor.matmul")
    return s


def test_set_dma_tags_the_copies_that_touch_the_buffer_and_survives_hoist(blocked):
    s = _staged(blocked)
    s.set_dma("lhsT_sb", "hwdge", "scalar")
    s.hoist("lhsT_sb", to="mo")                                  # the rewritten fill keeps its attributes
    dmas = [c for c in ir.walk(s.proc.body) if isinstance(c, ir.Call) and c.instr == "ns.sync.dma_copy"]
    tagged = [c for c in dmas if c.attr("dge") == "hwdge"]
    assert len(tagged) == 1 and tagged[0].attr("engine") == "scalar"
    assert tagged[0].arg("dst").buf == "lhsT_sb"


@pytest.mark.parametrize("dge, engine, message", [
    ("hwdge", "gpsimd", "'sync' or 'scalar'"),
    ("fancy", "sync", "dge must be one of"),
])
def test_set_dma_rejects_unsupported_combinations(blocked, dge, engine, message):
    with pytest.raises(ScheduleError, match=message):
        _staged(blocked).set_dma("rhs_sb", dge, engine)


def test_set_dma_needs_a_dma_on_the_buffer(blocked):
    with pytest.raises(ScheduleError, match="no DMA copy"):
        _staged(blocked).set_dma("matmul")


def test_emitted_dma_carries_dge_mode_and_engine(load_example, np_ref):
    import _common as c
    mod = load_example("fast_mm")
    s = Sched(c.matmul_proc(name="mm"), NC_TINY)
    mod.fast(s, NC_TINY, mt=2, nt=2, kt=2, dma=mod.DMA_PLANS["hw2"])
    src = s.source()
    assert "dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar" in src
    assert "dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync" in src


def test_footprint_assertion_uses_the_declared_element_size(load_example):
    import _common as c
    mod = load_example("fast_mm")
    s = Sched(c.matmul_proc(name="mm"), NC_TINY)
    mod.fast(s, NC_TINY, mt=2, nt=2, kt=1, order="nm", resident="rhs")      # rhs strip grows with K
    f32, bf16 = s.source(), s.source(itemsize=2)
    assert "assuming 4-byte elements" in f32 and "assuming 2-byte elements" in bf16
    # the strip dominates: bf16 needs about half of the f32 bytes per partition
    coef = lambda src: int(next(l for l in src.splitlines() if "SBUF footprint" in l).split("assert ")[1].split(" * (K")[0])
    n32, n16 = coef(f32), coef(bf16)
    assert n32 == 2 * n16 > 0
