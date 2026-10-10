"""Affine region / injectivity analysis (Halide-style interval inference over loop extents)."""

from __future__ import annotations

from . import ir
from .expr import Aff, Var, FloorDiv, Mod, MinE, Prod


class AnalysisError(Exception):
    pass


def loop_ranges(stmts) -> dict:
    """var -> extent for every For in stmts (recursively)."""
    return {s.var: s.extent for s in ir.walk(stmts) if isinstance(s, ir.For)}


def _split_ranged(e: Aff, ranges: dict):
    """-> (fixed part: Aff with ranged vars set to 0, [(coef, var)] for ranged vars)."""
    ranged = []
    fixed = Aff(e.const)
    for atom, c in e.terms:
        if isinstance(atom, Var) and atom.name in ranges:
            ranged.append((c, atom.name))
        elif isinstance(atom, (FloorDiv, Mod, MinE, Prod)) and (e_vars := Aff.of(atom).free_vars()) & set(ranges):
            raise AnalysisError(f"non-affine dependence on an inner loop variable in index {e}")
        else:
            fixed = fixed + Aff.of(atom) * c
    return fixed, ranged


def dim_interval(e: Aff, ranges: dict):
    """(lo, size) of index expression e as the ranged loop vars sweep [0, extent)."""
    fixed, ranged = _split_ranged(e, ranges)
    lo, size = fixed, Aff(1)
    for c, v in ranged:
        ext = ranges[v]
        span = (ext - 1) * abs(c)
        size = size + span
        if c < 0:
            lo = lo + (ext - 1) * c
    return lo, size


def region(idxs: list, ranges: dict):
    """Bounding box of several index tuples over the ranged loops: (lo tuple, size tuple).
    Different accesses must have bases differing by constants (else we cannot pick a symbolic min)."""
    ndim = len(idxs[0])
    los, sizes = [], []
    for d in range(ndim):
        ivs = [dim_interval(ix[d], ranges) for ix in idxs]
        base_lo, base_size = ivs[0]
        lo, hi = base_lo, base_lo + base_size
        for l, z in ivs[1:]:
            diff = l - lo
            if not diff.is_const:
                raise AnalysisError(f"cannot union accesses with symbolic offset difference {diff}")
            if diff.const < 0:
                lo = l
            end = l + z
            diff_hi = end - hi
            if not diff_hi.is_const:
                raise AnalysisError(f"cannot union accesses with symbolic extent difference {diff_hi}")
            if diff_hi.const > 0:
                hi = end
        los.append(lo)
        sizes.append(hi - lo)
    return tuple(los), tuple(sizes)


def const_int(e: Aff, what: str) -> int:
    if not e.is_const:
        raise AnalysisError(f"{what} is not a compile-time constant: {e}")
    return e.const


def injective(idx: tuple, loops: dict) -> bool:
    """Is (loops values) -> idx location injective? `loops`: var -> extent for the loops under
    consideration. Sufficient test: every loop var appears in exactly one dim, and within each dim
    the loops' coefficients form a non-overlapping mixed-radix system."""
    seen = {}
    for d, e in enumerate(idx):
        terms = []
        for atom, c in e.terms:
            if isinstance(atom, Var) and atom.name in loops:
                terms.append((abs(c), atom.name))
            elif isinstance(atom, (FloorDiv, Mod, MinE, Prod)) and Aff.of(atom).free_vars() & set(loops):
                return False
        for _, v in terms:
            if v in seen:
                return False
            seen[v] = d
        terms.sort()
        for (c1, v1), (c2, v2) in zip(terms, terms[1:]):
            ext = loops[v1]
            if not ext.is_const or c2 < c1 * ext.const:
                return False
    return set(seen) == set(loops)


def dma_bytes(proc, sizes: dict, itemsize: int = 4) -> int:
    """Bytes moved by ns.sync.dma_copy instructions when the kernel runs on the given sizes
    (e.g. {"K": 256, "M": 256, "N": 1024}): window elements x trip counts x itemsize. This is the
    static counterpart of what nkibench counts at simulation time."""
    env = dict(sizes)
    for name, expr in proc.params:
        env[name] = expr.eval(env)

    def rec(stmts, mult):
        total = 0
        for s in stmts:
            if isinstance(s, ir.For):
                total += rec(s.body, mult * s.extent.eval(env))
            elif isinstance(s, ir.Call) and s.instr == "ns.sync.dma_copy":
                elems = 1
                for z in s.arg("dst").shape():
                    elems *= z.eval(env)
                total += mult * elems * itemsize
        return total

    return rec(proc.body, 1)
