import types
from nkiknow import api_lookup as al


def make_mods():
    nl = types.ModuleType("nki.language"); nisa = types.ModuleType("nki.isa"); nki = types.ModuleType("nki")
    def nc_matmul(dst, stationary, moving, accumulate=None):
        """Matrix multiply on the tensor engine.

        dst must be in psum.
        Second line.
        """
    nisa.nc_matmul = nc_matmul
    nl.sbuf = object()          # a value, not a function
    nl.ndarray = lambda shape, dtype, buffer=None: None
    return {"nl": nl, "nisa": nisa, "nki": nki}


def test_error_names_bare_function(monkeypatch):
    monkeypatch.setattr(al, "_modules", make_mods)
    out = al.signatures_for("nc_matmul() got an unexpected keyword argument 'lhs'", "")
    assert "nisa.nc_matmul(dst, stationary, moving, accumulate=None)" in out
    assert "dst must be in psum" in out


def test_non_callable_and_missing(monkeypatch):
    monkeypatch.setattr(al, "_modules", make_mods)
    code = "import nki\nx = nl.sbuf((2,2))\ny = nl.ndaray((2,), 1)\n"
    out = al.signatures_for("TypeError: 'MemoryRegion' object is not callable", code)
    assert "`nl.sbuf` is a object value" in out or "not a function" in out
    assert "nl.ndaray` does not exist" in out and "nl.ndarray" in out.split("Close real names")[1]


def test_cap_and_max_three(monkeypatch):
    mods = make_mods()
    for i in range(10):
        setattr(mods["nisa"], f"f{i}", (lambda a: None))
        mods["nisa"].__dict__[f"f{i}"].__doc__ = "x" * 500
    monkeypatch.setattr(al, "_modules", lambda: mods)
    out = al.signatures_for("", "\n".join(f"nisa.f{i}(1)" for i in range(10)))
    assert out.count("`nisa.f") <= 3 and len(out) <= al.MAX_CHARS + 60


def test_no_modules(monkeypatch):
    monkeypatch.setattr(al, "_modules", lambda: {})
    assert al.signatures_for("x", "y") == ""
