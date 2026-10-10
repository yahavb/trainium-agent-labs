import pytest

from nki_sched import NC_DEFAULT, NC_TINY, HardwareConfig, ir
from nki_sched.expr import Aff, var
from nki_sched.lower import HardwareError, check_hw, copy_nest_to_call, infer_kinds, nest_window, select_copies

A = Aff
i, j = var("i"), var("j")


def buf(name, *shape, mem=ir.HBM, role="temp", dtype="f32"):
    return ir.Buffer(name, tuple(A(s) for s in shape), dtype, mem, role)


def prog(*stmts, args=()):
    return ir.Proc("t", tuple(args), (), tuple(stmts))


def win(name, *shape):
    return ir.Window(name, (A(0),) * len(shape), tuple(A(s) for s in shape))


# ------------------------------------------------------------------ check_hw
@pytest.mark.parametrize("alloc, hw, message", [
    (buf("t", 129, 8, mem=ir.SBUF), NC_DEFAULT, "partition dimension"),
    (buf("t", 8, 8, mem=ir.SBUF), NC_TINY, "partition dimension"),            # tiny config: pmax = 4
    (buf("t", 128, 8, mem=ir.PSUM, dtype="bf16"), NC_DEFAULT, "must be f32"),
    (buf("t", 128, 513, mem=ir.PSUM), NC_DEFAULT, "one bank"),                 # 513 * 4 B > 2 KiB
])
def test_check_hw_rejects(alloc, hw, message):
    with pytest.raises(HardwareError, match=message):
        check_hw(prog(ir.Alloc(alloc)), hw)


@pytest.mark.parametrize("alloc", [buf("t", 128, 512, mem=ir.PSUM), buf("t", 128, 4096, mem=ir.SBUF), buf("t", 1000, 7)])
def test_check_hw_accepts(alloc):
    check_hw(prog(ir.Alloc(alloc)), NC_DEFAULT)


def test_check_hw_requires_static_partition_dim_but_allows_symbolic_free_dims_in_sbuf():
    from nki_sched.expr import sym
    sym_free = ir.Buffer("t", (A(128), sym("K") // 128), "f32", ir.SBUF)
    check_hw(prog(ir.Alloc(sym_free)), NC_DEFAULT)
    sym_part = ir.Buffer("t", (sym("K"), A(8)), "f32", ir.SBUF)
    with pytest.raises(HardwareError, match="partition dim"):
        check_hw(prog(ir.Alloc(sym_part)), NC_DEFAULT)
    sym_psum = ir.Buffer("t", (A(128), sym("N")), "f32", ir.PSUM)
    with pytest.raises(HardwareError, match="fully static"):
        check_hw(prog(ir.Alloc(sym_psum)), NC_DEFAULT)


def mm_call(dst_mem=ir.PSUM, st_mem=ir.SBUF, mv_mem=ir.SBUF, k=128, m=128, n=512, k2=None):
    allocs = (ir.Alloc(buf("d", m, n, mem=dst_mem)), ir.Alloc(buf("s", k, m, mem=st_mem)), ir.Alloc(buf("v", k2 or k, n, mem=mv_mem)))
    call = ir.Call("ns.tensor.matmul", (("dst", win("d", m, n)), ("stationary", win("s", k, m)), ("moving", win("v", k2 or k, n))))
    return prog(*allocs, call)


def test_check_hw_accepts_a_legal_matmul():
    check_hw(mm_call(), NC_DEFAULT)


@pytest.mark.parametrize("kw, hw, message", [
    (dict(dst_mem=ir.SBUF), NC_DEFAULT, "'dst'.*must be in"),
    (dict(st_mem=ir.PSUM), NC_DEFAULT, "'stationary'.*must be in"),
    (dict(m=128), HardwareConfig(stationary_fmax=64), "exceeds caps"),   # tighter caps than the buffer limits,
    (dict(n=512), HardwareConfig(moving_fmax=256), "exceeds caps"),      # so the matmul rule is what fires
    (dict(k=129), NC_DEFAULT, "partition dimension"),                    # buffer rule fires first
    (dict(k2=64), NC_DEFAULT, "inconsistent shapes"),
])
def test_check_hw_rejects_illegal_matmul(kw, hw, message):
    with pytest.raises(HardwareError, match=message):
        check_hw(mm_call(**kw), hw)


def test_check_hw_rejects_unknown_instruction():
    with pytest.raises(HardwareError, match="unknown instruction"):
        check_hw(prog(ir.Call("ns.tensor.bogus", ())), NC_DEFAULT)


# ------------------------------------------------------------------ nest_window
def test_nest_window_infers_offsets_sizes_and_loop_per_dim():
    w, vars_ = nest_window("b", (var("o") * 8 + i, j), {"i": A(8), "j": A(4)})
    assert w.lo == (var("o") * 8, A(0)) and w.size == (A(8), A(4)) and vars_ == ["i", "j"]


def test_nest_window_point_dimension():
    w, vars_ = nest_window("b", (A(3), j), {"j": A(4)})
    assert w.size == (A(1), A(4)) and vars_ == [None, "j"]


@pytest.mark.parametrize("idx", [(i * 2,), (i + j,), (i, i), (j,)])
def test_nest_window_rejects_non_unit_stride_aliasing_or_unused_loops(idx):
    assert nest_window("b", idx, {"i": A(4), "j": A(4)}) == (None, None)


# ------------------------------------------------------------------ copy recognition
def copy_nest(dst, src, cast=None, off=A(0)):
    rhs = ir.Read(src, (i, j + off))
    if cast:
        rhs = ir.CastE(cast, rhs)
    return ir.For("i", A(2), (ir.For("j", A(4), (ir.Assign(dst, (i, j), rhs),)),))


@pytest.mark.parametrize("dmem, smem, instr", [
    (ir.SBUF, ir.HBM, "ns.sync.dma_copy"),
    (ir.HBM, ir.SBUF, "ns.sync.dma_copy"),
    (ir.SBUF, ir.PSUM, "ns.vector.tensor_copy"),
    (ir.SBUF, ir.SBUF, "ns.vector.tensor_copy"),
])
def test_copy_nest_to_call_picks_the_instruction_from_the_memories(dmem, smem, instr):
    bufs = {"d": buf("d", 2, 4, mem=dmem), "s": buf("s", 2, 8, mem=smem)}
    call = copy_nest_to_call(copy_nest("d", "s", off=A(2)), bufs)
    assert call.instr == instr
    assert call.arg("src").lo == (A(0), A(2)) and call.arg("src").size == (A(2), A(4))


@pytest.mark.parametrize("dmem, smem, ddt, sdt, cast", [
    (ir.HBM, ir.HBM, "f32", "f32", None),       # HBM -> HBM: no instruction
    (ir.SBUF, ir.HBM, "bf16", "f32", None),     # dma cannot convert dtypes
    (ir.SBUF, ir.PSUM, "bf16", "f32", "f32"),   # a cast that disagrees with the destination dtype
])
def test_copy_nest_to_call_declines_unsupported_copies(dmem, smem, ddt, sdt, cast):
    bufs = {"d": buf("d", 2, 4, mem=dmem, dtype=ddt), "s": buf("s", 2, 4, mem=smem, dtype=sdt)}
    assert copy_nest_to_call(copy_nest("d", "s", cast=cast), bufs) is None


def test_copy_nest_to_call_accepts_a_cast_matching_the_destination_dtype():
    bufs = {"d": buf("d", 2, 4, mem=ir.SBUF, dtype="bf16"), "s": buf("s", 2, 4, mem=ir.PSUM)}
    assert copy_nest_to_call(copy_nest("d", "s", cast="bf16"), bufs).instr == "ns.vector.tensor_copy"


def test_copy_nest_to_call_declines_computation():
    bufs = {"d": buf("d", 2, 4, mem=ir.SBUF), "s": buf("s", 2, 4, mem=ir.SBUF)}
    nest = ir.For("i", A(2), (ir.For("j", A(4), (ir.Assign("d", (i, j), ir.Mul(ir.Read("s", (i, j)), ir.Lit(2.0))),)),))
    assert copy_nest_to_call(nest, bufs) is None


def test_select_copies_converts_only_copy_nests():
    allocs = (ir.Alloc(buf("d", 2, 4, mem=ir.SBUF)), ir.Alloc(buf("s", 2, 4, mem=ir.PSUM)), ir.Alloc(buf("c", 2, 4)))
    compute = ir.For("i", A(2), (ir.For("j", A(4), (ir.Assign("c", (i, j), ir.Lit(1.0)),)),))
    out = select_copies(prog(*allocs, copy_nest("d", "s"), compute))
    assert [type(s).__name__ for s in out.body] == ["Alloc", "Alloc", "Alloc", "Call", "For"]
    assert out.body[3].instr == "ns.vector.tensor_copy"


# ------------------------------------------------------------------ loop kinds
def kinds(*stmts, args=()):
    p = infer_kinds(prog(*stmts, args=args))
    return {s.var: s.kind for s in ir.walk(p.body) if isinstance(s, ir.For)}


def test_independent_elementwise_loop_is_affine():
    assert kinds(ir.For("i", A(4), (ir.Assign("y", (i,), ir.Lit(1.0)),))) == {"i": "affine"}


def test_loop_writing_the_same_element_every_iteration_is_sequential():
    assert kinds(ir.For("i", A(4), (ir.Assign("y", (A(0),), ir.Read("x", (i,))),))) == {"i": "sequential"}


def test_scalar_accumulation_over_a_loop_is_sequential():
    ks = kinds(ir.For("i", A(4), (ir.For("j", A(3), (ir.Reduce("y", (i,), ir.Read("x", (j,))),)),)))
    assert ks == {"i": "affine", "j": "sequential"}


def test_overlapping_sweep_is_sequential_but_disjoint_sweep_is_affine():
    overlapping = ir.For("i", A(3), (ir.For("j", A(2), (ir.Assign("y", (i + j,), ir.Lit(1.0)),)),))
    disjoint = ir.For("i", A(2), (ir.For("j", A(2), (ir.Assign("y", (i * 2 + j,), ir.Lit(1.0)),)),))
    assert kinds(overlapping) == {"i": "sequential", "j": "affine"}
    assert kinds(disjoint) == {"i": "affine", "j": "affine"}


def test_read_of_a_neighbouring_iterations_write_is_sequential():
    nest = ir.For("i", A(3), (ir.Assign("y", (i + 1,), ir.Read("y", (i,))),))
    assert kinds(nest) == {"i": "sequential"}


def test_buffer_allocated_inside_the_loop_is_private():
    nest = ir.For("i", A(3), (ir.Alloc(buf("t", 1)), ir.Assign("t", (A(0),), ir.Lit(1.0)), ir.Assign("y", (i,), ir.Read("t", (A(0),)))))
    assert kinds(nest) == {"i": "affine"}


def test_psum_accumulation_loop_is_affine_via_matmul_accumulate_none():
    mm = ir.Call("ns.tensor.matmul", (("dst", win("acc", 4, 8)), ("stationary", win("s", 4, 4)), ("moving", win("v", 4, 8))),
                 (("accumulate", None),))
    nest = ir.For("k", A(3), (mm,))
    assert kinds(ir.Alloc(buf("acc", 4, 8, mem=ir.PSUM)), nest) == {"k": "affine"}


def test_window_stores_by_different_iterations_must_not_overlap():
    def store(stride):
        w = ir.Window("y", (i * stride,), (A(4),))
        return ir.For("i", A(3), (ir.Call("ns.sync.dma_copy", (("dst", w), ("src", win("s", 4)))),))
    assert kinds(store(4)) == {"i": "affine"}
    assert kinds(store(2)) == {"i": "sequential"}   # windows of 4 at stride 2 overlap


def test_explicit_kinds_are_kept():
    nest = ir.For("i", A(3), (ir.Assign("y", (A(0),), ir.Lit(1.0)),), kind="affine")
    assert kinds(nest) == {"i": "affine"}  # the user asked; mark() is what checks the claim
