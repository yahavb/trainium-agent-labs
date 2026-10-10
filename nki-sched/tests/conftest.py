"""Shared fixtures. Tests use only numpy, torch and this repo (no Neuron SDK, no nkibench)."""

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nki_sched import NC_TINY, Sched, arg, ir, trace, verify  # noqa: E402
from nki_sched.expr import sym, var  # noqa: E402

# small shapes that exercise 1..3 tiles in every dimension under NC_TINY (tiles 4 x 8, contraction 4)
CHECK_SHAPES = [dict(K=4, M=4, N=8), dict(K=8, M=8, N=16), dict(K=12, M=4, N=24)]


@pytest.fixture
def tiny_hw():
    return NC_TINY


@pytest.fixture
def check_shapes():
    return CHECK_SHAPES


@pytest.fixture
def make_proc():
    """factory: traced naive matmul program (C = lhsT.T @ rhs) with symbolic K, M, N."""

    def make(dtype="f32", name="mm"):
        return trace(
            lambda lhsT, rhs: lhsT.T @ rhs,
            {"lhsT": arg(("K", "M"), dtype, example=(8, 4)), "rhs": arg(("K", "N"), dtype, example=(8, 8))},
            name=name,
        )

    return make


@pytest.fixture
def naive_proc(make_proc):
    return make_proc()


@pytest.fixture
def make_inputs():
    """factory: random float32 inputs {"lhsT": [K,M], "rhs": [K,N]}."""

    def make(K, M, N, seed=0):
        r = np.random.default_rng(seed)
        return {"lhsT": r.standard_normal((K, M)).astype(np.float32),
                "rhs": r.standard_normal((K, N)).astype(np.float32)}

    return make


@pytest.fixture
def np_ref():
    """numpy reference for the matmul spec."""

    def ref(inputs):
        return inputs["lhsT"].T.astype(np.float32) @ inputs["rhs"].astype(np.float32)

    return ref


@pytest.fixture
def new_sched(make_proc, tiny_hw, np_ref):
    """factory: a Sched on the traced matmul with tiny tiles; check=True runs the differential
    oracle (incl. reverse-affine) after every primitive."""

    def make(check=True, proc=None, hw=None, shapes=CHECK_SHAPES):
        chk = verify.make_checker(np_ref, shapes) if check else None
        return Sched(proc if proc is not None else make_proc(), hw or tiny_hw, check=chk)

    return make


@pytest.fixture(scope="session")
def load_example():
    """factory: import examples/<name>.py as a module (the examples dir is put on sys.path)."""
    sys.path.insert(0, os.path.join(ROOT, "examples"))

    def load(name):
        spec = importlib.util.spec_from_file_location(f"examples_{name}", os.path.join(ROOT, "examples", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return load


@pytest.fixture
def build_golden_l4():
    """factory: the level-4 program written out by hand (what the l4 schedule should produce)."""

    def build(name="mm", tm=128, tn=512, tk=128):
        K, M, N = sym("K"), sym("M"), sym("N")
        A = ir.Aff
        lhsT = ir.Buffer("lhsT", (K, M), "like:lhsT", ir.HBM, "arg")
        rhs = ir.Buffer("rhs", (K, N), "like:rhs", ir.HBM, "arg")
        C = ir.Buffer("C", (M, N), "like:lhsT", ir.HBM, "out")
        acc = ir.Buffer("matmul", (A(tm), A(tn)), "f32", ir.PSUM)
        lhsT_sb = ir.Buffer("lhsT_sb", (A(tk), A(tm)), "like:lhsT", ir.SBUF)
        rhs_sb = ir.Buffer("rhs_sb", (A(tk), A(tn)), "like:rhs", ir.SBUF)
        C_sb = ir.Buffer("C_sb", (A(tm), A(tn)), "like:lhsT", ir.SBUF)
        mo, no, ko = var("mo"), var("no"), var("ko")
        full = lambda b: ir.Window(b.name, (A(0), A(0)), b.shape)  # noqa: E731
        dma_l = ir.Call("ns.sync.dma_copy", (("dst", full(lhsT_sb)),
                                             ("src", ir.Window("lhsT", (ko * tk, mo * tm), (A(tk), A(tm))))))
        dma_r = ir.Call("ns.sync.dma_copy", (("dst", full(rhs_sb)),
                                             ("src", ir.Window("rhs", (ko * tk, no * tn), (A(tk), A(tn))))))
        mm = ir.Call("ns.tensor.matmul", (("dst", full(acc)), ("stationary", full(lhsT_sb)), ("moving", full(rhs_sb))))
        cp = ir.Call("ns.vector.tensor_copy", (("dst", full(C_sb)), ("src", full(acc))))
        st = ir.Call("ns.sync.dma_copy", (("dst", ir.Window("C", (mo * tm, no * tn), (A(tm), A(tn)))), ("src", full(C_sb))))
        k_loop = ir.For("ko", K // tk, (ir.Alloc(lhsT_sb), ir.Alloc(rhs_sb), dma_l, dma_r, mm))
        n_loop = ir.For("no", N // tn, (ir.Alloc(acc), k_loop, ir.Alloc(C_sb), cp, st))
        return ir.Proc(
            name=name, args=(lhsT, rhs, C),
            sizes=(("K", "lhsT", 0), ("M", "lhsT", 1), ("N", "rhs", 1)),
            body=(ir.For("mo", M // tm, (n_loop,)),),
            assumptions=(ir.Assumption(M, tm), ir.Assumption(N, tn), ir.Assumption(K, tk)),
        )

    return build
