import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nki_sched import interp, ir, verify  # noqa: E402
from nki_sched.expr import Aff, var  # noqa: E402

A = Aff


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_hostile_inputs_have_the_requested_shapes_and_a_large_element(dtype):
    inp = verify.hostile_inputs({"lhsT": (4, 3), "rhs": (4, 5)}, seed=1, dtype=dtype)
    assert inp["lhsT"].shape == (4, 3) and inp["rhs"].shape == (4, 5)
    assert inp["lhsT"].flat[0] == pytest.approx(1e3, rel=1e-2)
    if dtype == "bf16":
        assert inp["__dtypes__"] == {"lhsT": "bf16", "rhs": "bf16"}
        assert (interp.quantize("bf16", inp["lhsT"]) == inp["lhsT"]).all()   # already bf16-representable
    else:
        assert "__dtypes__" not in inp


def test_hostile_inputs_are_deterministic_per_seed():
    a = verify.hostile_inputs({"x": (3, 3)}, seed=7)["x"]
    b = verify.hostile_inputs({"x": (3, 3)}, seed=7)["x"]
    c = verify.hostile_inputs({"x": (3, 3)}, seed=8)["x"]
    assert (a == b).all() and not (a == c).all()


def test_torch_oracle_takes_arguments_in_the_given_order():
    inputs = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones((3, 2), np.float32)}
    out = verify.torch_oracle(lambda a, b: a @ b, arg_names=("a", "b"))(inputs)
    np.testing.assert_allclose(out, inputs["a"] @ inputs["b"])


def test_torch_oracle_runs_in_bf16_when_the_inputs_say_so():
    x = interp.quantize("bf16", np.full((2, 2), 1.0 + 2 ** -7, np.float32))
    inputs = {"lhsT": x, "rhs": x, "__dtypes__": {"lhsT": "bf16", "rhs": "bf16"}}
    out = verify.torch_oracle(lambda a, b: a @ b)(inputs)
    assert out.dtype == np.float32 and out[0, 0] == pytest.approx(2 * (1.0 + 2 ** -7) ** 2, rel=1e-2)


def test_make_checker_accepts_a_correct_program(naive_proc, np_ref):
    verify.make_checker(np_ref, [dict(K=4, M=3, N=5)])(naive_proc)


def test_make_checker_reports_where_a_wrong_program_differs(naive_proc, np_ref):
    wrong = naive_proc.with_(body=tuple(ir.map_stmts(
        naive_proc.body, lambda s: ir.Reduce(s.buf, s.idx, ir.Mul(ir.Lit(2.0), s.rhs)) if isinstance(s, ir.Reduce) else None)))
    with pytest.raises(verify.Mismatch, match=r"shape \{'K': 4, 'M': 3, 'N': 5\}: first mismatch at \(0, 0\)"):
        verify.make_checker(np_ref, [dict(K=4, M=3, N=5)])(wrong)


def test_make_checker_reports_shape_mismatches(naive_proc):
    with pytest.raises(verify.Mismatch, match="first mismatch"):
        verify.make_checker(lambda inp: np.zeros((1, 1), np.float32), [dict(K=4, M=3, N=5)])(naive_proc)


def test_reverse_run_exposes_an_affine_claim_that_in_order_execution_hides():
    i = var("i")
    # y[i] = y[i-1]-style recurrence expressed as x -> y with a carried dependence, forced affine
    body = (ir.Assign("C", (A(0), A(0)), ir.Lit(1.0)),
            ir.For("i", A(2), (ir.Assign("C", (A(0), i + 1), ir.Mul(ir.Read("C", (A(0), i)), ir.Lit(2.0))),), kind="affine"))
    proc = ir.Proc("t", (ir.Buffer("lhsT", (A(1), A(1)), "f32", ir.HBM, "arg"), ir.Buffer("rhs", (A(1), A(1)), "f32", ir.HBM, "arg"),
                         ir.Buffer("C", (A(1), A(3)), "f32", ir.HBM, "out")), (), body)
    oracle = lambda inp: np.array([[1.0, 2.0, 4.0]], np.float32)  # noqa: E731
    # in order it is right ...
    verify.make_checker(oracle, [dict(K=1, M=1, N=1)], reverse_affine=False)(proc)
    # ... but with affine loops reversed the interpreter cannot even read the not-yet-written value
    with pytest.raises(interp.InterpError):
        verify.make_checker(oracle, [dict(K=1, M=1, N=1)], reverse_affine=True)(proc)
