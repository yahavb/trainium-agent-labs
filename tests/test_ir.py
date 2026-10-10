import pytest

from nki_sched import ir
from nki_sched.expr import Aff, sym, var

A = Aff
i, j = var("i"), var("j")


def buf(name, *shape, mem=ir.HBM, role="temp", dtype="f32"):
    return ir.Buffer(name, tuple(A(s) for s in shape), dtype, mem, role)


@pytest.fixture
def prog():
    """for i in 0..2: for j in 0..3: y[i,j] = x[j,i]*x[j,i];  with a scratch alloc inside the nest."""
    x, y, t = buf("x", 3, 2, role="arg"), buf("y", 2, 3, role="out"), buf("t", 3, mem=ir.SBUF)
    body = (ir.For("i", A(2), (
        ir.Alloc(t),
        ir.For("j", A(3), (
            ir.Assign("y", (i, j), ir.Mul(ir.Read("x", (j, i)), ir.Read("x", (j, i)))),
            ir.Reduce("t", (j,), ir.Read("x", (j, i))),
        ), stage="s"),
    )),)
    return ir.Proc("p", (x, y), (), body)


def test_walk_is_preorder_and_recursive(prog):
    kinds = [type(s).__name__ for s in ir.walk(prog.body)]
    assert kinds == ["For", "Alloc", "For", "Assign", "Reduce"]


def test_find_loop_and_loop_vars(prog):
    assert ir.find_loop(prog.body, "j").stage == "s"
    assert ir.find_loop(prog.body, "zz") is None
    assert ir.loop_vars(prog.body) == ["i", "j"]


def test_buffers_of_collects_args_and_allocs(prog):
    assert set(ir.buffers_of(prog)) == {"x", "y", "t"}


def test_buffers_of_rejects_duplicate_alloc(prog):
    dup = prog.with_(body=prog.body + (ir.Alloc(buf("t", 3)),))
    with pytest.raises(ValueError, match="duplicate allocation"):
        ir.buffers_of(dup)


def test_subst_vars_rewrites_indices_extents_and_windows():
    w = ir.Window("b", (i * 4, A(0)), (A(4), A(8)))
    s = ir.For("i", i + 2, (ir.Assign("a", (i,), ir.Lit(0.0)), ir.Call("ns.sync.dma_copy", (("dst", w), ("src", w)))))
    out = ir.subst_vars(s, {"i": j * 2})
    assert out.extent == j * 2 + 2
    assert out.body[0].idx == (j * 2,)
    assert out.body[1].args[0][1].lo == (j * 8, A(0))


def test_remap_buf_redirects_reads_writes_and_window_bases(prog):
    shift = lambda idx: tuple(e + 1 for e in idx)  # noqa: E731
    out = ir.remap_buf(prog.body[0], "x", "x2", shift)
    reads = [r for s in ir.walk((out,)) if isinstance(s, (ir.Assign, ir.Reduce)) for r in ir.expr_reads(s.rhs)]
    assert {r.buf for r in reads} == {"x2"} and all(r.idx[0] == j + 1 for r in reads)
    out_y = ir.remap_buf(prog.body[0], "y", "y2", shift)
    assert [s.buf for s in ir.walk((out_y,)) if isinstance(s, ir.Assign)] == ["y2"]


def test_remap_buf_keeps_window_points_and_size():
    w = ir.Window("b", (A(0), i, A(0)), (A(4), A(1), A(8)), (1,))
    c = ir.Call("ns.sync.dma_copy", (("dst", w), ("src", w)))
    out = ir.remap_buf(c, "b", "b2", lambda idx: idx)
    assert out.args[0][1].buf == "b2" and out.args[0][1].points == (1,) and out.args[0][1].size == w.size


def test_map_stmts_splices_and_keeps(prog):
    out = ir.map_stmts(prog.body, lambda s: (s, s) if isinstance(s, ir.Reduce) else None)
    assert sum(isinstance(s, ir.Reduce) for s in ir.walk(out)) == 2
    assert sum(isinstance(s, ir.Alloc) for s in ir.walk(out)) == 1


@pytest.mark.parametrize("stmt, expected", [
    (ir.Assign("a", (i,), ir.Read("b", (i,))), [("r", "b"), ("w", "a")]),
    (ir.Reduce("a", (i,), ir.Read("b", (i,))), [("r", "b"), ("rw", "a")]),
])
def test_accesses_scalar_modes(stmt, expected):
    assert [(m, b) for m, b, _ in ir.accesses(stmt)] == expected


@pytest.mark.parametrize("attrs, dst_mode", [((), "rw"), ((("accumulate", None),), "rw"), ((("accumulate", False),), "w")])
def test_matmul_dst_access_mode_follows_accumulate(attrs, dst_mode):
    win = lambda name: ir.Window(name, (A(0), A(0)), (A(4), A(8)))  # noqa: E731
    c = ir.Call("ns.tensor.matmul", (("dst", win("d")), ("stationary", win("s")), ("moving", win("m"))), attrs)
    modes = {b: m for m, b, _ in ir.accesses(c)}
    assert modes == {"d": dst_mode, "s": "r", "m": "r"}


def test_window_full_and_shape():
    b = buf("b", 4, 2, 8)
    assert ir.Window("b", (A(0),) * 3, b.shape).is_full({"b": b})
    assert not ir.Window("b", (A(0), i, A(0)), (A(4), A(1), A(8))).is_full({"b": b})
    pt = ir.Window("b", (A(0), A(1), A(0)), (A(4), A(1), A(8)), (1,))
    assert pt.shape() == (A(4), A(8)) and not pt.is_full({"b": b})


def test_call_arg_and_attr_lookup():
    w = ir.Window("d", (A(0),), (A(4),))
    c = ir.Call("x", (("dst", w),), (("accumulate", True),))
    assert c.arg("dst") is w and c.attr("accumulate") is True and c.attr("missing", 7) == 7
    with pytest.raises(KeyError):
        c.arg("src")


def test_printer_shows_loops_windows_points_and_assumptions(prog):
    w = ir.Window("b", (A(0), i, A(0)), (A(4), A(1), A(8)), (1,))
    p = prog.with_(body=prog.body + (ir.Call("ns.sync.dma_copy", (("dst", w), ("src", w))),),
                   assumptions=(ir.Assumption(sym("M"), 4),))
    text = str(p)
    assert "assume M % 4 == 0" in text
    assert "for i in 0..2:" in text and "t[j] += x[j, i]" in text
    assert "ns.sync.dma_copy(dst=b[0:+4, i, 0:+8]" in text
