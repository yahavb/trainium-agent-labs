"""The examples are code: they must keep running, and the schedules in them must keep producing
correct, hardware-legal kernels."""

import numpy as np
import pytest

from nki_sched import NC_DEFAULT, NC_TINY, interp, ir
from nki_sched.analysis import dma_bytes
from nki_sched.emit import emit_nki


@pytest.fixture(scope="module")
def ladder(load_example):
    return load_example("matmul_ladder")


# ------------------------------------------------------------------ the small examples
@pytest.mark.parametrize("module, name, expected", [
    ("loops", "example_naive", "matmul.init.m"),
    ("loops", "example_split", "assume K % 128 == 0"),
    ("loops", "example_reorder_reduction", "for matmul.k in 0..K"),
    ("loops", "example_reorder_rejected", None),
    ("loops", "example_compute_at", "alloc matmul : f32[128, 512] @ HBM"),
    ("memory", "example_stage_in", "ns.sync.dma_copy(dst=lhsT_sb"),
    ("memory", "example_set_memory_rejected", None),
    ("memory", "example_tensorize_one_tile", "nisa.nc_matmul(dst=matmul, stationary=lhsT_sb, moving=rhs_sb)"),
])
def test_example_runs_and_prints_what_it_documents(load_example, capsys, module, name, expected):
    getattr(load_example(module), name)()
    out = capsys.readouterr().out
    assert out.strip()
    if expected:
        assert expected in out


def test_examples_have_docstrings_that_explain_them(load_example):
    for module in ("loops", "memory"):
        mod = load_example(module)
        fns = [f for n, f in vars(mod).items() if n.startswith("example_")]
        assert fns and all(f.__doc__ and len(f.__doc__) > 40 for f in fns)


def test_tensorize_example_prints_the_emitted_nki_kernel(load_example, capsys):
    load_example("memory").example_tensorize_one_tile()
    out = capsys.readouterr().out
    assert "-- emitted NKI source:" in out and "@nki.jit" in out and "nl.affine_range" in out


# ------------------------------------------------------------------ the matmul ladder
@pytest.mark.parametrize("level", ["l4", "l5", "l6", "l7"])
@pytest.mark.parametrize("K, M, N", [(4, 4, 8), (8, 8, 16), (12, 4, 24), (8, 12, 8)])
def test_ladder_schedules_compute_the_right_matmul(ladder, make_inputs, np_ref, level, K, M, N):
    s = ladder.build(level, NC_TINY)
    inp = make_inputs(K, M, N, seed=K + M + N)
    np.testing.assert_allclose(interp.run(s.proc, inp)["C"], np_ref(inp), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("level", ["l4", "l5", "l6", "l7"])
def test_ladder_schedules_emit_legal_kernels_with_the_expected_entry_points(ladder, level):
    src = ladder.build(level, NC_DEFAULT).source()
    assert f"def {ladder.ENTRY[level]}(lhsT, rhs):" in src and "@nki.jit" in src
    assert src.count("nisa.nc_matmul(") == 1
    compile(src, level, "exec")


def test_l4_matches_the_hand_written_golden_kernel_up_to_statement_order(ladder, build_golden_l4):
    """The only differences are the order of independent allocations and loads."""
    got = ladder.build("l4", NC_DEFAULT).source().replace(ladder.ENTRY["l4"], "mm")
    want = emit_nki(build_golden_l4(), NC_DEFAULT)
    assert sorted(got.splitlines()) == sorted(want.splitlines())


def test_only_the_hoisting_schedules_need_a_footprint_assertion(ladder):
    assert "SBUF footprint" not in ladder.build("l4").source()
    for level in ("l5", "l6"):
        assert "SBUF footprint" in ladder.build(level).source()


# the byte counts below were measured by nkibench in the NKI simulator on a Trainium node
# (l4: 1.00 1.56 1.00 2.00; l5: 1.00 1.11 1.00 1.14; l6/l7: 1.00 on (K,M,N) = the four shapes)
NKIBENCH_SHAPES = [(128, 128, 512), (256, 256, 1024), (512, 128, 512), (256, 512, 1024)]


@pytest.mark.parametrize("level, expected_ratios", [
    ("l4", [1.00, 1.56, 1.00, 2.00]),
    ("l5", [1.00, 1.11, 1.00, 1.14]),
    ("l6", [1.00, 1.00, 1.00, 1.00]),
    ("l7", [1.00, 1.00, 1.00, 1.00]),
])
def test_static_hbm_traffic_matches_what_nkibench_measured(ladder, level, expected_ratios):
    proc = ladder.build(level, NC_DEFAULT).proc
    for (K, M, N), want in zip(NKIBENCH_SHAPES, expected_ratios):
        floor = 4 * (K * M + K * N + M * N)
        got = dma_bytes(proc, {"K": K, "M": M, "N": N}, itemsize=4) / floor
        assert got == pytest.approx(want, abs=0.006), (K, M, N)


def test_hoisting_never_increases_traffic(ladder):
    sizes = {"K": 256, "M": 512, "N": 1024}
    bytes_ = {lv: dma_bytes(ladder.build(lv).proc, sizes) for lv in ("l4", "l5", "l6")}
    assert bytes_["l4"] > bytes_["l5"] > bytes_["l6"]


def test_cli_writes_the_kernel_and_ir_only_mode_does_not(ladder, tmp_path, capsys):
    out = tmp_path / "k.py"
    ladder.main(["l4", str(out), "--ir"])
    assert not out.exists() and "ns.tensor.matmul" in capsys.readouterr().out
    ladder.main(["l4", str(out)])
    assert "def nki_matmul_tiled_" in out.read_text()


def test_ladder_programs_contain_no_scalar_statements_after_emission_selection(ladder):
    from nki_sched.lower import select_copies
    proc = select_copies(ladder.build("l4").proc)
    assert not [s for s in ir.walk(proc.body) if isinstance(s, (ir.Assign, ir.Reduce))]
