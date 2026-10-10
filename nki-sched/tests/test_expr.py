import pytest

from nki_sched.expr import Aff, Sym, Var, sym, to_py, var

i, j, M = var("i"), var("j"), sym("M")


@pytest.mark.parametrize("expr, text", [
    (i + 3, "i + 3"),
    (i + i, "2*i"),
    (i - i, "0"),
    (2 * i + 3 * j - i, "i + 3*j"),
    (-i, "-i"),
    (i * 4 + 4 * i, "8*i"),
    (Aff(5), "5"),
    (i + 0, "i"),
])
def test_affine_arithmetic_is_canonical(expr, text):
    assert str(expr) == text


def test_equality_ignores_construction_order():
    assert i + j + 1 == 1 + j + i
    assert hash(i + j) == hash(j + i)


@pytest.mark.parametrize("a, b", [(i, j), (i, i + 1), (M, i)])
def test_distinct_expressions_differ(a, b):
    assert a != b


@pytest.mark.parametrize("expr, d, text", [
    (i * 8 + 16, 4, "2*i + 4"),     # divisible: exact division
    (Aff(12), 4, "3"),
    (M, 128, "(M // 128)"),         # not divisible: stays symbolic
    (i * 6, 4, "(6*i // 4)"),
])
def test_floordiv(expr, d, text):
    assert str(expr // d) == text


def test_floordiv_by_one_is_identity():
    assert M // 1 == M


@pytest.mark.parametrize("expr, d, is_zero", [(i * 8, 4, True), (Aff(8), 4, True), (i, 4, False), (i * 6, 4, False)])
def test_mod(expr, d, is_zero):
    assert ((expr % d) == Aff(0)) is is_zero


def test_multiplying_two_symbols_gives_an_opaque_atom():
    p = i * j
    assert not p.is_const and p.free_vars() == {"i", "j"}
    assert i * j == j * i  # canonical operand order


def test_min_folds_constants_and_dedups():
    assert Aff.min(3, 5) == Aff(3)
    assert Aff.min(i, i) == i
    assert Aff.min(i, j) == Aff.min(j, i)


@pytest.mark.parametrize("expr, env, value", [
    (2 * i + j + 1, {"i": 3, "j": 4}, 11),
    (M // 128, {"M": 512}, 4),
    (M % 128, {"M": 130}, 2),
    (Aff.min(M, i), {"M": 7, "i": 9}, 7),
    (i * j, {"i": 3, "j": 5}, 15),
])
def test_eval(expr, env, value):
    assert expr.eval(env) == value


def test_subs_substitutes_through_opaque_atoms():
    e = (i + 1) // 4 + i * 2
    out = e.subs({"i": 4 * j})
    assert out.eval({"j": 3}) == e.eval({"i": 12})


def test_subs_leaves_other_vars_alone():
    assert (i + j).subs({"i": 5}) == j + 5


def test_free_vars_and_syms_and_coef():
    e = 3 * i + (M // 128) + (j * 0)
    assert e.free_vars() == {"i"} and e.free_syms() == {"M"} and e.coef("i") == 3 and e.coef("j") == 0


def test_var_and_sym_are_different_atoms():
    assert var("x") != sym("x") and Var("x") != Sym("x")


@pytest.mark.parametrize("expr, src", [
    (i, "i"),
    (2 * i + 3, "2 * i + 3"),
    (i - j, "i - j"),
    (M // 128, "M // 128"),
    (128 * (M // 128), "128 * (M // 128)"),
    (M // 128 + 1, "(M // 128) + 1"),
    ((i + 1) // 2, "(i + 1) // 2"),
    (Aff.min(M, 4), "min(4, M)"),
])
def test_to_py_is_precedence_safe(expr, src):
    assert to_py(expr) == src
    # and evaluates the same as the symbolic expression
    env = {"i": 5, "j": 2, "M": 512}
    assert eval(src, {"min": min}, env) == expr.eval(env)


def test_to_py_applies_rename():
    assert to_py(var("a.b") + 1, lambda n: n.replace(".", "_")) == "a_b + 1"


def test_of_rejects_unknown_types():
    with pytest.raises(TypeError):
        Aff.of(1.5)
