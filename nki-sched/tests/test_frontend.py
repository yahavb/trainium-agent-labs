import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nki_sched import arg, interp, ir, trace  # noqa: E402
from nki_sched.expr import Aff, sym, var  # noqa: E402
from nki_sched.frontend import TraceError  # noqa: E402

K, M, N = sym("K"), sym("M"), sym("N")


def nest(stage, loops, stmt):
    for v, ext in reversed(loops):
        stmt = ir.For(v, ext, (stmt,), stage)
    return stmt


def expected_matmul_ir(out_rhs):
    """The naive IR for C = lhsT.T @ rhs, written out by hand."""
    v = var
    acc = ir.Buffer("matmul", (M, N), "f32", ir.HBM, "temp")
    init = nest("matmul.init", [("matmul.init.m", M), ("matmul.init.n", N)],
                ir.Assign("matmul", (v("matmul.init.m"), v("matmul.init.n")), ir.Lit(0.0)))
    update = nest("matmul", [("matmul.m", M), ("matmul.n", N), ("matmul.k", K)],
                  ir.Reduce("matmul", (v("matmul.m"), v("matmul.n")),
                            ir.Mul(ir.Read("lhsT", (v("matmul.k"), v("matmul.m"))), ir.Read("rhs", (v("matmul.k"), v("matmul.n"))))))
    idx = (v("C.m"), v("C.n"))
    out = nest("C", [("C.m", M), ("C.n", N)], ir.Assign("C", idx, out_rhs(ir.Read("matmul", idx))))
    return (ir.Alloc(acc), init, update, out)


def trace2(fn, dtype="f32", dims=(("K", "M"), ("K", "N")), shapes=((8, 4), (8, 8)), names=("lhsT", "rhs")):
    args = {n: arg(d, dtype, example=s) for n, d, s in zip(names, dims, shapes)}
    return trace(fn, args, name="k")


@pytest.mark.parametrize("dtype, out_rhs", [
    ("f32", lambda r: r),                                   # f32 accumulate == f32 result: no cast needed
    ("bf16", lambda r: ir.CastE("like:lhsT", r)),            # torch returns bf16: cast the f32 accumulator
])
def test_trace_matmul_matches_the_hand_written_naive_ir(dtype, out_rhs):
    p = trace2(lambda lhsT, rhs: lhsT.T @ rhs, dtype)
    assert p.body == expected_matmul_ir(out_rhs)
    assert [(b.name, b.role, b.dtype) for b in p.args] == [
        ("lhsT", "arg", "like:lhsT"), ("rhs", "arg", "like:rhs"), ("C", "out", "like:lhsT")]
    assert p.sizes == (("K", "lhsT", 0), ("M", "lhsT", 1), ("N", "rhs", 1))
    assert p.name == "k"


def test_widening_casts_on_operands_are_elided():
    explicit = trace2(lambda lhsT, rhs: (lhsT.float().T @ rhs.float()).to(lhsT.dtype), "bf16")
    assert explicit.body == expected_matmul_ir(lambda r: ir.CastE("like:lhsT", r))


def test_narrowing_cast_is_kept_and_output_dtype_is_concrete():
    p = trace2(lambda lhsT, rhs: (lhsT.T @ rhs).to(torch.bfloat16), "f32")
    assert p.args[-1].dtype == "bf16"
    assert p.body[-1].body[0].body[0].rhs == ir.CastE("bf16", ir.Read("matmul", (var("C.m"), var("C.n"))))


def test_output_name_and_stage_names_follow_arguments():
    p = trace(lambda lhsT, rhs: lhsT.T @ rhs, {"lhsT": arg(("K", "M"), example=(8, 4)), "rhs": arg(("K", "N"), example=(8, 8))},
              name="k", output="out")
    assert p.args[-1].name == "out" and ir.find_loop(p.body, "out.m") is not None


# ---------------------------------------------------------------- generality: many specs, torch is the oracle
SPECS = {
    "a.T @ b": (lambda a, b: a.T @ b, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
    "a @ b": (lambda a, b: a @ b, (("M", "K"), ("K", "N")), ((4, 8), (8, 6))),
    "a.T @ b.T": (lambda a, b: a.T @ b.T, (("K", "M"), ("N", "K")), ((8, 4), (6, 8))),
    "permute": (lambda a, b: a.permute(1, 0) @ b, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
    "transposed result": (lambda a, b: (a.T @ b).T, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
    "transpose(0,1)": (lambda a, b: torch.transpose(a, 0, 1) @ b, (("K", "M"), ("K", "N")), ((8, 4), (8, 6))),
}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_trace_is_correct_for_many_specs(name):
    fn, dims, shapes = SPECS[name]
    p = trace2(fn, dims=dims, shapes=shapes, names=("a", "b"))
    r = np.random.default_rng(0)
    a, b = (r.standard_normal(s).astype(np.float32) for s in shapes)
    got = interp.run(p, {"a": a, "b": b})["C"]
    np.testing.assert_allclose(got, fn(torch.from_numpy(a), torch.from_numpy(b)).numpy(), rtol=1e-4, atol=1e-4)


def test_two_chained_matmuls_materialize_two_stages():
    fn = lambda a, b, c: (a.T @ b) @ c  # noqa: E731
    p = trace(fn, {"a": arg(("K", "M"), example=(8, 4)), "b": arg(("K", "N"), example=(8, 6)), "c": arg(("N", "P"), example=(6, 3))}, name="k")
    stages = {s.stage for s in ir.walk(p.body) if isinstance(s, ir.For)}
    assert stages == {"matmul.init", "matmul", "matmul_1.init", "matmul_1", "C"}
    r = np.random.default_rng(1)
    a, b, c = (r.standard_normal(s).astype(np.float32) for s in ((8, 4), (8, 6), (6, 3)))
    got = interp.run(p, {"a": a, "b": b, "c": c})["C"]
    np.testing.assert_allclose(got, (a.T @ b) @ c, rtol=1e-4, atol=1e-4)
    assert [s[0] for s in p.sizes] == ["K", "M", "N", "P"]


def test_bf16_trace_matches_torch_bf16_after_rounding():
    fn = lambda lhsT, rhs: lhsT.T @ rhs  # noqa: E731
    p = trace2(fn, "bf16")
    r = np.random.default_rng(2)
    a, b = (interp.quantize("bf16", r.standard_normal(s).astype(np.float32)) for s in ((8, 4), (8, 8)))
    got = interp.run(p, {"lhsT": a, "rhs": b, "__dtypes__": {"lhsT": "bf16", "rhs": "bf16"}})["C"]
    want = fn(torch.from_numpy(a).to(torch.bfloat16), torch.from_numpy(b).to(torch.bfloat16)).float().numpy()
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("fn, dims, shapes, message", [
    (lambda lhsT, rhs: torch.relu(lhsT.T @ rhs), (("K", "M"), ("K", "N")), ((8, 4), (8, 8)), "unsupported torch op in spec: aten.relu"),
    (lambda lhsT, rhs: lhsT + rhs, (("A", "B"), ("A", "B")), ((4, 4), (4, 4)), "unsupported torch op in spec: aten.add"),
    # numerically consistent example shapes, but the contraction dims are different symbols
    (lambda lhsT, rhs: lhsT.T @ rhs, (("K", "M"), ("J", "N")), ((8, 4), (8, 8)), "contraction dims differ"),
])
def test_trace_errors_name_the_problem(fn, dims, shapes, message):
    with pytest.raises(TraceError, match=message):
        trace2(fn, dims=dims, shapes=shapes)


def test_trace_needs_example_shapes():
    with pytest.raises(TraceError, match="example"):
        trace(lambda a, b: a @ b, {"a": arg(("M", "K")), "b": arg(("K", "N"))})


def test_trace_rejects_unsupported_dtype():
    with pytest.raises(TraceError, match="unsupported dtype"):
        trace(lambda a, b: (a.T @ b).to(torch.float64), {"a": arg(("K", "M"), example=(4, 4)), "b": arg(("K", "N"), example=(4, 4))})
