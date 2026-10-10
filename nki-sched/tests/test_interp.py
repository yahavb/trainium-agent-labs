import numpy as np
import pytest

from nki_sched import interp, ir
from nki_sched.expr import Aff, sym, var

A = Aff
i, j = var("i"), var("j")


def buf(name, *shape, mem=ir.HBM, role="temp", dtype="f32"):
    return ir.Buffer(name, tuple(A(s) if isinstance(s, int) else s for s in shape), dtype, mem, role)


def proc(args, body, sizes=(), assumptions=()):
    return ir.Proc("t", tuple(args), tuple(sizes), tuple(body), tuple(assumptions))


def win(name, shape, lo=None, points=()):
    lo = lo or (A(0),) * len(shape)
    return ir.Window(name, tuple(lo), tuple(A(s) for s in shape), points)


# ------------------------------------------------------------------ quantize
@pytest.mark.parametrize("value, expected", [
    (1.0, 1.0),
    (1.0 + 2 ** -8, 1.0),                 # exactly halfway to the next bf16 -> ties to even (down)
    (1.0 + 3 * 2 ** -9, 1.0 + 2 ** -7),    # above halfway -> up
    (3.14159265, 3.140625),
    (-3.14159265, -3.140625),
    (0.0, 0.0),
])
def test_quantize_bf16(value, expected):
    assert interp.quantize("bf16", np.float32(value)) == np.float32(expected)


@pytest.mark.parametrize("dtype, value, expected", [
    ("f32", 0.1, np.float32(0.1)),
    ("f16", 0.1, np.float32(np.float16(0.1))),
    ("f16", 70000.0, np.float32(np.inf)),
])
def test_quantize_other_dtypes(dtype, value, expected):
    assert interp.quantize(dtype, np.float32(value)) == expected


def test_quantize_preserves_shape_and_scalar_rank():
    assert interp.quantize("bf16", np.float32(1.5)).ndim == 0
    assert interp.quantize("bf16", np.ones((2, 3), np.float32)).shape == (2, 3)


def test_quantize_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        interp.quantize("f8", np.zeros(1, np.float32))


# ------------------------------------------------------------------ statements
def test_assign_reduce_and_loops():
    # y[i] = sum_j x[i, j]
    x, y = buf("x", 2, 3, role="arg"), buf("y", 2, role="out")
    body = [ir.For("i", A(2), (
        ir.Assign("y", (i,), ir.Lit(0.0)),
        ir.For("j", A(3), (ir.Reduce("y", (i,), ir.Read("x", (i, j))),)),
    ))]
    xv = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = interp.run(proc([x, y], body), {"x": xv})
    np.testing.assert_array_equal(out["y"], xv.sum(1))


def test_sizes_are_read_from_argument_shapes_and_extents_use_them():
    n = sym("N")
    x, y = buf("x", n, role="arg"), buf("y", n, role="out")
    body = [ir.For("i", n, (ir.Assign("y", (i,), ir.Mul(ir.Read("x", (i,)), ir.Lit(2.0))),))]
    out = interp.run(proc([x, y], body, sizes=[("N", "x", 0)]), {"x": np.arange(5, dtype=np.float32)})
    np.testing.assert_array_equal(out["y"], 2 * np.arange(5))


def test_params_are_evaluated_from_sizes():
    n = sym("N")
    x, y = buf("x", n, role="arg"), buf("y", Aff.of(sym("H")), role="out")
    body = [ir.For("i", sym("H"), (ir.Assign("y", (i,), ir.Read("x", (i,))),))]
    p = proc([x, y], body, sizes=[("N", "x", 0)]).with_(params=(("H", n // 2),))
    out = interp.run(p, {"x": np.arange(8, dtype=np.float32)})
    np.testing.assert_array_equal(out["y"], np.arange(4))


@pytest.mark.parametrize("dtype, stored", [("f32", 0.1), ("f16", float(np.float16(0.1))), ("bf16", 0.10009765625)])
def test_stores_round_to_the_buffer_dtype(dtype, stored):
    y = buf("y", 1, role="out", dtype=dtype)
    out = interp.run(proc([y], [ir.Assign("y", (A(0),), ir.Lit(0.1))]), {})
    assert out["y"][0] == pytest.approx(stored, rel=1e-6)


def test_like_dtype_follows_the_input_dtype_override():
    x, y = buf("x", 1, role="arg", dtype="like:x"), buf("y", 1, role="out", dtype="like:x")
    p = proc([x, y], [ir.Assign("y", (A(0),), ir.Read("x", (A(0),)))])
    xv = np.array([1.0 + 2 ** -9], np.float32)
    assert interp.run(p, {"x": xv})["y"][0] == xv[0]
    assert interp.run(p, {"x": xv, "__dtypes__": {"x": "bf16"}})["y"][0] == 1.0  # rounded to bf16


def test_cast_expression_rounds():
    y = buf("y", 1, role="out")
    p = proc([y], [ir.Assign("y", (A(0),), ir.CastE("bf16", ir.Lit(3.14159265)))])
    assert interp.run(p, {})["y"][0] == pytest.approx(3.140625)


# ------------------------------------------------------------------ errors
def test_missing_input_is_reported():
    x, y = buf("x", 1, role="arg"), buf("y", 1, role="out")
    with pytest.raises(interp.InterpError, match="missing input x"):
        interp.run(proc([x, y], []), {})


def test_assumption_violation_is_reported():
    n = sym("N")
    x = buf("x", n, role="arg")
    p = proc([x], [], sizes=[("N", "x", 0)], assumptions=[ir.Assumption(n, 4)])
    with pytest.raises(interp.InterpError, match="assumption violated"):
        interp.run(p, {"x": np.zeros(6, np.float32)})
    interp.run(p, {"x": np.zeros(8, np.float32)})  # divisible: fine


def test_out_of_bounds_scalar_index():
    y = buf("y", 2, role="out")
    with pytest.raises(interp.InterpError, match="index out of bounds"):
        interp.run(proc([y], [ir.Assign("y", (A(2),), ir.Lit(1.0))]), {})


def test_read_of_uninitialised_buffer():
    y, t = buf("y", 1, role="out"), buf("t", 1)
    body = [ir.Alloc(t), ir.Assign("y", (A(0),), ir.Read("t", (A(0),)))]
    with pytest.raises(interp.InterpError, match="uninitialised"):
        interp.run(proc([y], body), {})


def test_reduce_into_uninitialised_buffer():
    y = buf("y", 1, role="out")
    with pytest.raises(interp.InterpError, match="uninitialised"):
        interp.run(proc([y], [ir.Reduce("y", (A(0),), ir.Lit(1.0))]), {})


def test_unknown_instruction():
    y = buf("y", 1, role="out")
    with pytest.raises(interp.InterpError, match="unknown instruction"):
        interp.run(proc([y], [ir.Call("ns.tensor.bogus", ())]), {})


# ------------------------------------------------------------------ instructions
def copy_proc(instr, dst_win, src_win, out_shape, dtype="f32", src_dtype="f32"):
    x = buf("x", 4, 8, role="arg", dtype=src_dtype)
    y = buf("y", *out_shape, role="out", dtype=dtype)
    return proc([x, y], [ir.Call(instr, (("dst", dst_win), ("src", src_win)))])


@pytest.mark.parametrize("instr", ["ns.sync.dma_copy", "ns.vector.tensor_copy", "ns.scalar.tensor_copy", "ns.gpsimd.tensor_copy"])
def test_copy_instructions_copy_a_window(instr):
    xv = np.arange(32, dtype=np.float32).reshape(4, 8)
    p = copy_proc(instr, win("y", (2, 4)), win("x", (2, 4), lo=(A(1), A(2))), (2, 4))
    np.testing.assert_array_equal(interp.run(p, {"x": xv})["y"], xv[1:3, 2:6])


def test_copy_with_point_dim_drops_the_axis():
    xv = np.arange(32, dtype=np.float32).reshape(4, 8)
    src = ir.Window("x", (A(0), A(3)), (A(4), A(1)), (1,))  # x[0:4, 3] -> 1-D of 4
    p = copy_proc("ns.sync.dma_copy", win("y", (4,)), src, (4,))
    np.testing.assert_array_equal(interp.run(p, {"x": xv})["y"], xv[:, 3])


def test_copy_casts_to_destination_dtype():
    xv = np.full((4, 8), 1.0 + 2 ** -9, np.float32)
    p = copy_proc("ns.vector.tensor_copy", win("y", (4, 8)), win("x", (4, 8)), (4, 8), dtype="bf16")
    assert (interp.run(p, {"x": xv})["y"] == 1.0).all()


def test_copy_shape_mismatch_and_out_of_bounds():
    xv = np.zeros((4, 8), np.float32)
    bad = copy_proc("ns.sync.dma_copy", win("y", (2, 4)), win("x", (4, 4)), (2, 4))
    with pytest.raises(interp.InterpError, match="shape mismatch"):
        interp.run(bad, {"x": xv})
    oob = copy_proc("ns.sync.dma_copy", win("y", (4, 4)), win("x", (4, 4), lo=(A(0), A(6))), (4, 4))
    with pytest.raises(interp.InterpError, match="out of bounds"):
        interp.run(oob, {"x": xv})


def matmul_proc_with(attrs, repeat=1, psum_shape=(3, 5)):
    """acc = repeat x (st.T @ mv) on freshly allocated PSUM, copied to y."""
    st, mv = buf("st", 4, 3, role="arg"), buf("mv", 4, 5, role="arg")
    y = buf("y", *psum_shape, role="out")
    acc = buf("acc", *psum_shape, mem=ir.PSUM)
    call = ir.Call("ns.tensor.matmul", (("dst", win("acc", psum_shape)), ("stationary", win("st", (4, 3))),
                                        ("moving", win("mv", (4, 5)))), attrs)
    return proc([st, mv, y], [ir.Alloc(acc)] + [call] * repeat
                + [ir.Call("ns.vector.tensor_copy", (("dst", win("y", psum_shape)), ("src", win("acc", psum_shape))))])


@pytest.fixture
def mm_inputs():
    r = np.random.default_rng(0)
    return {"st": r.standard_normal((4, 3)).astype(np.float32), "mv": r.standard_normal((4, 5)).astype(np.float32)}


@pytest.mark.parametrize("attrs, repeat, factor", [
    ((("accumulate", None),), 1, 1),   # first write to a fresh region overwrites
    ((("accumulate", None),), 3, 3),   # then later writes add
    ((("accumulate", False),), 3, 1),  # always overwrite
])
def test_matmul_accumulate_semantics(mm_inputs, attrs, repeat, factor):
    out = interp.run(matmul_proc_with(attrs, repeat), mm_inputs)["y"]
    np.testing.assert_allclose(out, factor * (mm_inputs["st"].T @ mm_inputs["mv"]), rtol=1e-5)


def test_matmul_accumulate_true_needs_initialised_psum(mm_inputs):
    with pytest.raises(interp.InterpError, match="uninitialised PSUM"):
        interp.run(matmul_proc_with((("accumulate", True),)), mm_inputs)


def test_matmul_accumulate_none_on_partially_written_region_is_rejected(mm_inputs):
    st, mv = buf("st", 4, 3, role="arg"), buf("mv", 4, 5, role="arg")
    y = buf("y", 3, 5, role="out")
    acc = buf("acc", 3, 5, mem=ir.PSUM)
    half = ir.Window("acc", (A(0), A(0)), (A(3), A(2)))
    first = ir.Call("ns.tensor.matmul", (("dst", half), ("stationary", win("st", (4, 3))),
                                         ("moving", ir.Window("mv", (A(0), A(0)), (A(4), A(2))))), (("accumulate", None),))
    whole = ir.Call("ns.tensor.matmul", (("dst", win("acc", (3, 5))), ("stationary", win("st", (4, 3))),
                                         ("moving", win("mv", (4, 5)))), (("accumulate", None),))
    with pytest.raises(interp.InterpError, match="partially-written"):
        interp.run(proc([st, mv, y], [ir.Alloc(acc), first, whole]), mm_inputs)


def test_matmul_shape_mismatch(mm_inputs):
    with pytest.raises(interp.InterpError, match="matmul: result"):
        interp.run(matmul_proc_with((), psum_shape=(3, 5)).with_(body=(
            ir.Alloc(buf("acc", 3, 4, mem=ir.PSUM)),
            ir.Call("ns.tensor.matmul", (("dst", win("acc", (3, 4))), ("stationary", win("st", (4, 3))),
                                         ("moving", win("mv", (4, 5)))), (("accumulate", False),)))), mm_inputs)


def test_matmul_on_a_point_indexed_3d_operand():
    st = buf("st", 4, 2, 3, role="arg")      # tile index on axis 1, dropped by a point window
    mv, y = buf("mv", 4, 5, role="arg"), buf("y", 3, 5, role="out")
    acc = buf("acc", 3, 5, mem=ir.PSUM)
    sta = ir.Window("st", (A(0), A(1), A(0)), (A(4), A(1), A(3)), (1,))
    call = ir.Call("ns.tensor.matmul", (("dst", win("acc", (3, 5))), ("stationary", sta), ("moving", win("mv", (4, 5)))),
                   (("accumulate", None),))
    p = proc([st, mv, y], [ir.Alloc(acc), call, ir.Call("ns.vector.tensor_copy", (("dst", win("y", (3, 5))), ("src", win("acc", (3, 5)))))])
    r = np.random.default_rng(1)
    inp = {"st": r.standard_normal((4, 2, 3)).astype(np.float32), "mv": r.standard_normal((4, 5)).astype(np.float32)}
    np.testing.assert_allclose(interp.run(p, inp)["y"], inp["st"][:, 1, :].T @ inp["mv"], rtol=1e-5)


# ------------------------------------------------------------------ loop order
def test_reverse_affine_only_reverses_affine_loops():
    y = buf("y", 3, role="out")
    # y[0] = 1; for i: y[i+1] = y[i] * 2  -- order matters
    def body(kind):
        return [ir.Assign("y", (A(0),), ir.Lit(1.0)),
                ir.For("i", A(2), (ir.Assign("y", (i + 1,), ir.Mul(ir.Read("y", (i,)), ir.Lit(2.0))),), kind=kind)]
    assert list(interp.run(proc([y], body("sequential")), {}, reverse_affine=True)["y"]) == [1.0, 2.0, 4.0]
    with pytest.raises(interp.InterpError):
        interp.run(proc([y], body("affine")), {}, reverse_affine=True)
    assert list(interp.run(proc([y], body("affine")), {}, reverse_affine=False)["y"]) == [1.0, 2.0, 4.0]
