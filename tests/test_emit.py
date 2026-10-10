import pytest

from nki_sched import NC_DEFAULT, HardwareConfig, ir
from nki_sched.emit import EmitError, dtype_src, emit_nki, ident, shape_src, window_src
from nki_sched.expr import Aff, sym, var

A = Aff

GOLDEN_L4_SOURCE = """\
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def mm(lhsT, rhs):
  K, M = lhsT.shape
  _, N = rhs.shape
  assert M % 128 == 0, "expected M to be a multiple of 128"
  assert N % 512 == 0, "expected N to be a multiple of 512"
  assert K % 128 == 0, "expected K to be a multiple of 128"
  C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for mo in nl.affine_range(M // 128):
    for no in nl.affine_range(N // 512):
      matmul = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
      for ko in nl.affine_range(K // 128):
        lhsT_sb = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_sb = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_sb, src=lhsT[128 * ko:128 * ko + 128, 128 * mo:128 * mo + 128])
        nisa.dma_copy(dst=rhs_sb, src=rhs[128 * ko:128 * ko + 128, 512 * no:512 * no + 512])
        nisa.nc_matmul(dst=matmul, stationary=lhsT_sb, moving=rhs_sb)
      C_sb = nl.ndarray((128, 512), dtype=lhsT.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=C_sb, src=matmul)
      nisa.dma_copy(dst=C[128 * mo:128 * mo + 128, 512 * no:512 * no + 512], src=C_sb)

  return C
"""


def test_golden_l4_emits_exactly_this_source(build_golden_l4):
    assert emit_nki(build_golden_l4(), NC_DEFAULT) == GOLDEN_L4_SOURCE


def test_emitted_source_is_valid_python(build_golden_l4):
    compile(emit_nki(build_golden_l4(), NC_DEFAULT), "<emitted>", "exec")


@pytest.mark.parametrize("name, expected", [
    ("lhsT_sb", "lhsT_sb"),
    ("matmul.k", "matmul_k"),
    ("C.init.m", "C_init_m"),
    ("2x", "v_2x"),
    ("for", "v_for"),
])
def test_ident_sanitizes_names(name, expected):
    assert ident(name) == expected


@pytest.mark.parametrize("dtype, expected", [
    ("f32", "nl.float32"), ("bf16", "nl.bfloat16"), ("f16", "nl.float16"), ("like:lhsT", "lhsT.dtype"),
])
def test_dtype_src(dtype, expected):
    assert dtype_src(dtype) == expected


@pytest.mark.parametrize("shape, expected", [
    ((A(4),), "(4,)"),
    ((A(128), A(512)), "(128, 512)"),
    ((A(128), sym("K") // 128, A(8)), "(128, K // 128, 8)"),
])
def test_shape_src(shape, expected):
    assert shape_src(shape) == expected


def test_window_src_full_slice_and_point_forms():
    b = ir.Buffer("b", (A(4), sym("K") // 4, A(8)), "f32", ir.SBUF)
    bufs = {"b": b}
    full = ir.Window("b", (A(0),) * 3, b.shape)
    assert window_src(full, bufs) == "b"
    sliced = ir.Window("b", (A(0), var("t"), A(0)), (A(4), A(1), A(8)), (1,))
    assert window_src(sliced, bufs) == "b[0:4, t, 0:8]"
    shifted = ir.Window("b", (var("o") * 4, A(0), A(0)), (A(4), b.shape[1], A(8)))
    assert window_src(shifted, bufs) == "b[4 * o:4 * o + 4, 0:K // 4, 0:8]"


def tiny_proc(*stmts, args=(), assumptions=(), params=()):
    base = (ir.Buffer("x", (sym("N"),), "like:x", ir.HBM, "arg"), ir.Buffer("y", (sym("N"),), "like:x", ir.HBM, "out"))
    return ir.Proc("k", base + tuple(args), (("N", "x", 0),), tuple(stmts), tuple(assumptions), tuple(params))


def copy_call(**attrs):
    w = ir.Window("x", (A(0),), (A(8),))
    return ir.Call("ns.sync.dma_copy", (("dst", ir.Window("y", (A(0),), (A(8),))), ("src", w)), tuple(attrs.items()))


def test_params_and_assumptions_become_python_statements():
    src = emit_nki(tiny_proc(copy_call(), params=(("H", sym("N") // 2),), assumptions=(ir.Assumption(sym("N"), 8),)))
    assert "N = x.shape" in src or "N, = x.shape" in src
    assert "H = N // 2" in src
    assert 'assert N % 8 == 0, "expected N to be a multiple of 8"' in src


def test_engine_and_accumulate_attributes_are_emitted_only_when_explicit():
    assert "engine" not in emit_nki(tiny_proc(copy_call()))
    assert "dma_copy(dst=y[0:8], src=x[0:8], engine=nisa.engine.vector)" in emit_nki(tiny_proc(copy_call(engine="vector")))
    assert "dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync" in emit_nki(tiny_proc(copy_call(dge="hwdge", engine="sync")))


def test_loop_kinds_map_to_nl_range_functions():
    def loop(kind):
        return ir.For("i", A(4), (ir.Alloc(ir.Buffer("t", (A(4),), "f32", ir.SBUF)),), kind=kind)
    assert "nl.static_range(4)" in emit_nki(tiny_proc(loop("static")))
    assert "nl.sequential_range(4)" in emit_nki(tiny_proc(loop("sequential")))
    assert "nl.affine_range(4)" in emit_nki(tiny_proc(loop("affine")))


def test_scalar_statements_that_were_not_lowered_are_an_error_naming_the_buffer():
    scalar = ir.For("i", A(4), (ir.Assign("y", (var("i"),), ir.Lit(1.0)),))
    with pytest.raises(EmitError, match="scalar statement left in the program .*y"):
        emit_nki(tiny_proc(scalar))


def test_remaining_pure_copy_nests_are_selected_automatically():
    tile = ir.Alloc(ir.Buffer("t", (A(8),), "like:x", ir.SBUF))
    nest = ir.For("i", A(8), (ir.Assign("t", (var("i"),), ir.Read("x", (var("i"),))),))
    assert "nisa.dma_copy(dst=t, src=x[0:8])" in emit_nki(tiny_proc(tile, nest))


def test_sbuf_footprint_is_asserted_in_the_kernel_when_symbolic_and_rejected_when_constant():
    sym_buf = ir.Buffer("t", (A(128), sym("N")), "f32", ir.SBUF)
    src = emit_nki(tiny_proc(ir.Alloc(sym_buf)))
    assert "assert 4 * N <= 196608" in src and "SBUF footprint" in src
    too_big = ir.Buffer("t", (A(128), A(100000)), "f32", ir.SBUF)
    with pytest.raises(EmitError, match="SBUF footprint"):
        emit_nki(tiny_proc(ir.Alloc(too_big)))
    emit_nki(tiny_proc(ir.Alloc(too_big)), HardwareConfig(sbuf_bytes_per_partition=10 ** 7))


def test_hardware_legality_is_checked_at_emission():
    from nki_sched.lower import HardwareError
    bad = ir.Buffer("t", (A(256), A(8)), "f32", ir.SBUF)
    with pytest.raises(HardwareError, match="partition dimension"):
        emit_nki(tiny_proc(ir.Alloc(bad)))


def test_hbm_temporaries_are_allocated_in_hbm():
    src = emit_nki(tiny_proc(ir.Alloc(ir.Buffer("tmp", (sym("N"),), "f32", ir.HBM))))
    assert "tmp = nl.ndarray((N,), dtype=nl.float32, buffer=nl.hbm)" in src
