import pytest

from nki_sched import ir
from nki_sched.analysis import AnalysisError, const_int, dim_interval, injective, loop_ranges, region
from nki_sched.expr import Aff, sym, var

A = Aff
i, j, k = var("i"), var("j"), var("k")
M = sym("M")


@pytest.mark.parametrize("expr, ranges, lo, size", [
    (i, {"i": A(4)}, A(0), A(4)),                       # plain sweep
    (i + 5, {"i": A(4)}, A(5), A(4)),
    (i * 2, {"i": A(4)}, A(0), A(7)),                   # stride 2 touches 0,2,4,6 -> span 7
    (i * 4 + j, {"i": A(3), "j": A(4)}, A(0), A(12)),   # mixed radix: contiguous
    (-i + 3, {"i": A(4)}, A(0), A(4)),                  # negative coefficient: lowest is 3-3
    (var("o") * 128 + i, {"i": A(128)}, var("o") * 128, A(128)),  # outer var stays in the base
    (i, {"i": M // 128}, A(0), M // 128),               # symbolic extent
    (A(7), {"i": A(4)}, A(7), A(1)),                    # independent of ranged vars
])
def test_dim_interval(expr, ranges, lo, size):
    assert dim_interval(expr, ranges) == (lo, size)


def test_dim_interval_rejects_nonaffine_dependence_on_a_ranged_var():
    with pytest.raises(AnalysisError, match="non-affine"):
        dim_interval((i + 1) // 2, {"i": A(4)})


def test_region_of_a_single_access_is_its_bounding_box():
    lo, size = region([(var("o") * 128 + i, j)], {"i": A(128), "j": A(8)})
    assert lo == (var("o") * 128, A(0)) and size == (A(128), A(8))


def test_region_unions_accesses_that_differ_by_constants():
    lo, size = region([(i,), (i + 2,)], {"i": A(4)})
    assert lo == (A(0),) and size == (A(6),)
    lo, size = region([(i + 2,), (i,)], {"i": A(4)})
    assert lo == (A(0),) and size == (A(6),)


def test_region_rejects_accesses_with_symbolic_offset_difference():
    with pytest.raises(AnalysisError, match="symbolic offset"):
        region([(i + var("a"),), (i + var("b"),)], {"i": A(4)})


def test_const_int():
    assert const_int(A(5), "x") == 5
    with pytest.raises(AnalysisError, match="not a compile-time constant"):
        const_int(M, "the tile")


def test_loop_ranges_collects_nested_loops():
    body = (ir.For("i", A(2), (ir.For("j", M, (ir.Assign("a", (i, j), ir.Lit(0.0)),)),)),)
    assert loop_ranges(body) == {"i": A(2), "j": M}


@pytest.mark.parametrize("idx, loops, expected", [
    ((i, j), {"i": A(2), "j": A(3)}, True),                    # one loop per dimension
    ((i * 4 + j,), {"i": A(2), "j": A(4)}, True),              # mixed radix, no overlap
    ((i * 3 + j,), {"i": A(2), "j": A(4)}, False),             # stride 3 < extent 4: overlap
    ((i + j,), {"i": A(2), "j": A(2)}, False),                 # classic aliasing
    ((i,), {"i": A(2), "j": A(2)}, False),                     # j does not index -> not injective in (i, j)
    ((i, i), {"i": A(2)}, False),                              # same loop in two dims
    ((i * 4 + j,), {"i": M, "j": A(4)}, True),                 # symbolic outer extent is fine
])
def test_injective(idx, loops, expected):
    assert injective(idx, loops) is expected
