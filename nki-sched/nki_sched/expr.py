"""Symbolic integer expressions used for loop extents, indices, window offsets and sizes.

An `Aff` is `const + sum(coef * atom)`. Atoms are loop variables (`Var`), size/parameter symbols
(`Sym`), and a few opaque non-affine forms (`FloorDiv`, `Mod`, `MinE`, `Prod`) that are treated as
fresh symbols by the analyses. Everything is immutable, hashable and canonical (terms sorted,
zero coefficients dropped) so structural equality is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Var:
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True, order=True)
class Sym:
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class FloorDiv:
    num: "Aff"
    den: int

    def __str__(self):
        return f"({self.num} // {self.den})"


@dataclass(frozen=True)
class Mod:
    num: "Aff"
    den: int

    def __str__(self):
        return f"({self.num} % {self.den})"


@dataclass(frozen=True)
class MinE:
    a: "Aff"
    b: "Aff"

    def __str__(self):
        return f"min({self.a}, {self.b})"


@dataclass(frozen=True)
class Prod:
    a: "Aff"
    b: "Aff"

    def __str__(self):
        return f"({self.a} * {self.b})"


def _key(atom):
    return (type(atom).__name__, str(atom))


@dataclass(frozen=True)
class Aff:
    const: int = 0
    terms: tuple = ()  # ((atom, coef), ...) sorted by _key, coef != 0

    # ---- construction -------------------------------------------------
    @staticmethod
    def of(x) -> "Aff":
        if isinstance(x, Aff):
            return x
        if isinstance(x, int):
            return Aff(x, ())
        if isinstance(x, (Var, Sym, FloorDiv, Mod, MinE, Prod)):
            return Aff(0, ((x, 1),))
        if isinstance(x, str):
            return Aff(0, ((Sym(x), 1),))
        raise TypeError(f"cannot make an index expression from {x!r}")

    @staticmethod
    def _make(const, d):
        terms = tuple(sorted(((a, c) for a, c in d.items() if c != 0), key=lambda t: _key(t[0])))
        return Aff(const, terms)

    def _dict(self):
        return dict(self.terms)

    # ---- arithmetic ---------------------------------------------------
    def __add__(self, o):
        o = Aff.of(o)
        d = self._dict()
        for a, c in o.terms:
            d[a] = d.get(a, 0) + c
        return Aff._make(self.const + o.const, d)

    __radd__ = __add__

    def __neg__(self):
        return Aff._make(-self.const, {a: -c for a, c in self.terms})

    def __sub__(self, o):
        return self + (-Aff.of(o))

    def __rsub__(self, o):
        return Aff.of(o) - self

    def __mul__(self, o):
        o = Aff.of(o)
        if o.is_const:
            k = o.const
            return Aff._make(self.const * k, {a: c * k for a, c in self.terms})
        if self.is_const:
            return o * self.const
        a, b = sorted((self, o), key=str)
        return Aff.of(Prod(a, b))

    __rmul__ = __mul__

    def __floordiv__(self, d: int):
        assert isinstance(d, int) and d > 0
        if d == 1:
            return self
        if self.const % d == 0 and all(c % d == 0 for _, c in self.terms):
            return Aff._make(self.const // d, {a: c // d for a, c in self.terms})
        return Aff.of(FloorDiv(self, d))

    def __mod__(self, d: int):
        assert isinstance(d, int) and d > 0
        if self.const % d == 0 and all(c % d == 0 for _, c in self.terms):
            return Aff(0, ())
        return Aff.of(Mod(self, d))

    @staticmethod
    def min(a, b) -> "Aff":
        a, b = Aff.of(a), Aff.of(b)
        if a.is_const and b.is_const:
            return Aff(min(a.const, b.const))
        if a == b:
            return a
        x, y = sorted((a, b), key=str)
        return Aff.of(MinE(x, y))

    # ---- queries --------------------------------------------------------
    @property
    def is_const(self) -> bool:
        return not self.terms

    def coef(self, name: str) -> int:
        for a, c in self.terms:
            if isinstance(a, Var) and a.name == name:
                return c
        return 0

    def free_vars(self) -> set:
        """Names of all loop variables appearing anywhere (incl. inside opaque atoms)."""
        out = set()
        for a, _ in self.terms:
            if isinstance(a, Var):
                out.add(a.name)
            elif isinstance(a, (FloorDiv, Mod)):
                out |= a.num.free_vars()
            elif isinstance(a, (MinE, Prod)):
                out |= a.a.free_vars() | a.b.free_vars()
        return out

    def free_syms(self) -> set:
        out = set()
        for a, _ in self.terms:
            if isinstance(a, Sym):
                out.add(a.name)
            elif isinstance(a, (FloorDiv, Mod)):
                out |= a.num.free_syms()
            elif isinstance(a, (MinE, Prod)):
                out |= a.a.free_syms() | a.b.free_syms()
        return out

    # ---- substitution / evaluation ------------------------------------
    def subs(self, m: dict) -> "Aff":
        """Substitute loop variables / symbols by name -> Aff|int."""
        if not m:
            return self
        res = Aff(self.const)
        for a, c in self.terms:
            res = res + _subs_atom(a, m) * c
        return res

    def eval(self, env: dict) -> int:
        v = self.const
        for a, c in self.terms:
            v += c * _eval_atom(a, env)
        return v

    # ---- printing -----------------------------------------------------
    def __str__(self):
        parts = []
        for a, c in self.terms:
            s = str(a)
            if c == 1:
                parts.append(s)
            elif c == -1:
                parts.append(f"-{s}")
            else:
                parts.append(f"{c}*{s}")
        if self.const or not parts:
            parts.append(str(self.const))
        out = " + ".join(parts).replace("+ -", "- ")
        return out

    __repr__ = __str__


def _subs_atom(a, m):
    if isinstance(a, (Var, Sym)):
        if a.name in m:
            return Aff.of(m[a.name])
        return Aff.of(a)
    if isinstance(a, FloorDiv):
        return a.num.subs(m) // a.den
    if isinstance(a, Mod):
        return a.num.subs(m) % a.den
    if isinstance(a, MinE):
        return Aff.min(a.a.subs(m), a.b.subs(m))
    if isinstance(a, Prod):
        return a.a.subs(m) * a.b.subs(m)
    raise TypeError(a)


def _eval_atom(a, env):
    if isinstance(a, (Var, Sym)):
        return env[a.name]
    if isinstance(a, FloorDiv):
        return a.num.eval(env) // a.den
    if isinstance(a, Mod):
        return a.num.eval(env) % a.den
    if isinstance(a, MinE):
        return min(a.a.eval(env), a.b.eval(env))
    if isinstance(a, Prod):
        return a.a.eval(env) * a.b.eval(env)
    raise TypeError(a)


def var(name: str) -> Aff:
    return Aff.of(Var(name))


def sym(name: str) -> Aff:
    return Aff.of(Sym(name))


def to_py(e: Aff, rename=lambda n: n) -> str:
    """Python source for an index expression (for the NKI emitter)."""
    parts = []
    combined = len(e.terms) > 1 or e.const != 0
    for a, c in e.terms:
        s = _atom_py(a, rename)
        if isinstance(a, (FloorDiv, Mod, Prod)) and (c != 1 or combined):
            s = f"({s})"
        if c == 1:
            parts.append(s)
        elif c == -1:
            parts.append(f"-{s}")
        else:
            parts.append(f"{c} * {s}")
    if e.const or not parts:
        parts.append(str(e.const))
    return " + ".join(parts).replace("+ -", "- ")


def _paren(src: str) -> str:
    return f"({src})" if " " in src else src


def _atom_py(a, rename):
    if isinstance(a, (Var, Sym)):
        return rename(a.name)
    if isinstance(a, FloorDiv):
        return f"{_paren(to_py(a.num, rename))} // {a.den}"
    if isinstance(a, Mod):
        return f"{_paren(to_py(a.num, rename))} % {a.den}"
    if isinstance(a, MinE):
        return f"min({to_py(a.a, rename)}, {to_py(a.b, rename)})"
    if isinstance(a, Prod):
        return f"{_paren(to_py(a.a, rename))} * {_paren(to_py(a.b, rename))}"
    raise TypeError(a)
