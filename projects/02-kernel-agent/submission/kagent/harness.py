"""Verification harness: rule check + numerical check + a failure description the agent can act on.

TOLERANCE = a proven float32 error bound, not a chosen constant. A case passes iff, per element,
    |got - ref| <= gamma_k * scale,    gamma_k = k*u / (1 - k*u),  u = 2**-24 (float32)
where ref is float64 on the exact float32 inputs, `scale` is the level's magnitude from backward
error analysis and k counts the roundings on the path to one output (Higham, Accuracy and
Stability of Numerical Algorithms, ch. 3-4). Examples:
  * relu(a*x + b): mul, add, store -> k = 3, scale = |a||x| + |b|
  * row sum of N terms, any order: |s_hat - s| <= gamma_(N-1) * sum|x|, plus the store -> k = N
  * row max: exact in floating point -> k = 0, the kernel must be bit-exact
So any kernel that is correct in float32 arithmetic passes, whatever its summation order, and a
float64 accumulator passes with room to spare. Bugs (dropped partial tile, wrong identity,
overflow, missing max-subtraction, E[x^2]-E[x]^2 cancellation) give errors >= 1e-3 * scale, far
above gamma_k * scale for our sizes (gamma_2053 = 1.2e-4).
`bound use` = max |got-ref| / bound is reported for every passing kernel: how much of its
allowed error it actually spends.
Non-finite output where the reference is finite always fails.
"""
import argparse
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import guard, rules, static
from .levels import LEVELS, TINY, Case, Level

U32 = 2.0 ** -24
# errors below this (x scale) are rounding, not logic bugs; those are >= 1e-3, see TOLERANCE
ROUNDING_CEILING = 1e-3


def gamma(k):
    k = np.asarray(k, dtype=np.float64)
    return k * U32 / (1 - k * U32)
CASE_TIMEOUT_S = 20.0


@dataclass
class Violation:
    line: int | None
    kind: str
    what: str
    source_line: str = ""

    def fix(self):
        return rules.fix_for(self.kind, self.what)

    def text(self):
        loc = f"line {self.line} `{self.source_line}`" if self.line else "kernel"
        return f"{loc}: {self.what} [{self.kind}]"


@dataclass
class CaseResult:
    case: Case
    status: str                 # pass | wrong | nonfinite | shape | error | timeout
    msg: str = ""
    worst: float = 0.0          # max |got-ref| / scale
    bound_use: float = 0.0      # max |got-ref| / bound; <= 1 passes
    facts: tuple = ()           # location/value facts, compared across cases to find the pattern


@dataclass
class Report:
    level: Level
    violations: list = field(default_factory=list)
    results: list = field(default_factory=list)
    load_error: str | None = None
    cost: dict | None = None    # efficiency on the largest dev case

    @property
    def n_pass(self):
        return sum(r.status == "pass" for r in self.results)

    @property
    def numerics_ok(self):
        return self.load_error is None and bool(self.results) and self.n_pass == len(self.results)

    @property
    def passed(self):
        return self.numerics_ok and not self.violations

    def failure_classes(self):
        """Coarse labels for the taxonomy; the agent refines them."""
        c = sorted({v.kind for v in self.violations})
        if self.load_error:
            c.append("load-error")
        c += sorted({r.status for r in self.results if r.status != "pass"})
        return c

    def summary(self):
        lv = self.level
        if self.load_error:
            return f"L{lv.num} {lv.name}: FAIL, kernel did not load ({self.load_error})"
        verdict = "PASS" if self.passed else "FAIL"
        return (f"L{lv.num} {lv.name}: {verdict}, {len(self.violations)} rule violation(s), "
                f"{self.n_pass}/{len(self.results)} cases correct")

    def numeric_feedback(self, max_cases=2):
        """Distilled numerical failure: the cross-case pattern first, then the worst cases."""
        if self.load_error:
            return self.load_error
        fails = [r for r in self.results if r.status != "pass"]
        if not fails:
            return (f"All {len(self.results)} cases within the float32 error bound "
                    f"(worst uses {self.bound_use():.0%} of it).")
        lines = [f"{len(fails)}/{len(self.results)} cases wrong."]
        for pat in filter(None, [_pattern(self.results), _common_facts(fails)]):
            lines.append(f"Pattern: {pat}.")
        errs = {}  # same exception type at the same line is one crash, whatever the shapes
        for r in fails:
            if r.status in ("error", "timeout"):
                where = r.msg[r.msg.rfind(" at line "):] if " at line " in r.msg else ""
                errs.setdefault((r.msg.split(":")[0], where), []).append(r)
        for rs in errs.values():
            lines.append(f"Crash in {len(rs)} case(s) (e.g. '{rs[0].case.name}'): {rs[0].msg}")
        shown = [r for r in sorted(fails, key=lambda r: -r.worst) if r.status not in ("error", "timeout")]
        for r in _diverse(shown, max_cases):
            lines.append(f"- '{r.case.name}': {r.msg}")
        return "\n".join(lines)

    def bound_use(self):
        return max((r.bound_use for r in self.results), default=0.0)

    def cost_text(self):
        c = self.cost
        if not c:
            return ""
        flops = f"{c['flops'] / c['min_flops']:.2f}x the minimal" if c["min_flops"] else f"{c['flops']}"
        return (f"Cost on '{c['case']}': {flops} FLOPs, "
                f"HBM reads {c['hbm_read'] / c['min_read']:.2f}x and writes "
                f"{c['hbm_write'] / c['min_write']:.2f}x the minimum, {c['ops']} instructions "
                f"averaging {c['flops'] / max(c['ops'], 1):.0f} elements (a full tile is "
                f"{rules.TILE_R * rules.TILE_C}).")

    def text(self, max_cases=2):
        out = [self.summary()]
        if self.violations:
            out.append("Rule violations (a violating kernel scores zero):")
            out += [f"- {v.text()}" for v in self.violations]
        out.append(self.numeric_feedback(max_cases))
        if self.passed:
            out.append(self.cost_text())
        return "\n".join(out)


def _diverse(results, k):
    """Worst-first, but prefer cases that differ in tags so two examples aren't the same bug twice."""
    picked, seen = [], set()
    for r in results:
        key = frozenset(r.case.tags)
        if key not in seen:
            picked.append(r)
            seen.add(key)
        if len(picked) == k:
            break
    return picked


def pattern_tag(results):
    """The case tag that best separates failing from passing cases: (tag, exclusive) or None.
    A tag shared with many passing cases is a coincidence, not a pattern; saying nothing beats
    pointing the model at the wrong cause."""
    fails = [set(r.case.tags) for r in results if r.status != "pass"]
    passes = [set(r.case.tags) for r in results if r.status == "pass"]
    if not fails or not passes:
        return None
    best = None
    for t in sorted(set().union(*fails)):
        nf = sum(t in f for f in fails)
        np_ = sum(t in p for p in passes)
        if nf == len(fails) and np_ == 0:
            return t, True
        score = (nf / len(fails)) * (nf / (nf + np_))
        if best is None or score > best[0]:
            best = (score, t, nf, np_)
    _, t, nf, np_ = best
    if nf / len(fails) >= 0.9 and nf / (nf + np_) >= 0.75:
        return t, False
    return None


def _pattern(results):
    fails = [r for r in results if r.status != "pass"]
    if fails and len(fails) == len(results):
        return "every case fails, including small single-tile ones"
    pt = pattern_tag(results)
    if pt is None:
        return None
    t, exclusive = pt
    if exclusive:
        return f"every failing case is '{t}' and no passing case is"
    nf = sum(t in r.case.tags for r in fails)
    np_ = sum(t in r.case.tags for r in results if r.status == "pass")
    return f"{nf}/{len(fails)} failing cases are '{t}' (only {np_} passing case(s) are)"


def common_facts(fails):
    """Facts true of (nearly) every numerically wrong case, e.g. 'every wrong output is exactly 0'."""
    measured = [set(r.facts) for r in fails if r.status in ("wrong", "nonfinite")]
    if len(measured) < 2:
        return [], len(measured)
    strong, support = {}, {}
    for fs in measured:
        for f in fs:
            base = f.lstrip("~")
            support[base] = support.get(base, 0) + 1
            if not f.startswith("~"):
                strong[base] = strong.get(base, 0) + 1
    # a constant like "exactly 0" must hold with ONE value across cases, so it's keyed by value
    return [f for f in sorted(strong) if support[f] >= 0.9 * len(measured)], len(measured)


def fact_support(fails, needle):
    """(cases with a fact containing `needle`, numerically wrong cases). Matching by substring lets
    facts that carry per-case numbers (offsets, cut points) count together."""
    measured = [r.facts for r in fails if r.status in ("wrong", "nonfinite")]
    k = sum(any(needle in f and not f.startswith("~") for f in fs) for fs in measured)
    return k, len(measured)


def _common_facts(fails):
    common, n = common_facts(fails)
    if not common:
        return None
    return f"in all or nearly all {n} wrong cases, " + "; ".join(common)


def _ranges(coords, limit=3):
    coords = list(coords)
    runs, start = [], coords[0]
    for a, b in zip(coords, coords[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}..{a}")
            if b is not None:
                start = b
    more = f" +{len(runs) - limit} more" if len(runs) > limit else ""
    return ", ".join(runs[:limit]) + more


def _locate(bad, tiles, names):
    """Returns (text, facts). Facts are phrased to read well when shared across cases. A fact
    prefixed '~' is merely consistent (a 1-row output can't show 'only the last row'), so it
    supports a cross-case fact without being evidence for it."""
    parts, facts = [], []
    for ax, T in enumerate(tiles):
        D = bad.shape[ax]
        others = tuple(i for i in range(bad.ndim) if i != ax)
        coords = np.nonzero(bad.any(axis=others) if others else bad)[0]
        nm = names[ax]
        ntiles = -(-D // T)
        if len(coords) == D:
            parts.append(f"every {nm} (all {D})")
            if D == 1:
                facts.append(f"~only the last {nm} is wrong")
            if ntiles == 1 and D % T:
                facts.append(f"~wrong outputs lie only in the last partial {nm} tile")
            continue
        tiles_bad = sorted({int(c) // T for c in coords})
        last_start = (ntiles - 1) * T
        if list(coords) == [D - 1]:
            parts.append(f"only the last {nm} ({nm} {D - 1} of {D})")
            facts.append(f"only the last {nm} is wrong")
            if D % T:
                facts.append(f"{'' if ntiles > 1 else '~'}wrong outputs lie only in the last "
                             f"partial {nm} tile")
        elif ntiles == 1:
            parts.append(f"{len(coords)}/{D} {nm}s ({_ranges(coords)})")
            if D % T:
                facts.append(f"~wrong outputs lie only in the last partial {nm} tile")
        elif tiles_bad == [ntiles - 1] and D % T:
            parts.append(f"only {nm}s {_ranges(coords)}, all inside the last partial {nm} tile "
                         f"({nm}s {last_start}..{D - 1}, width {D - last_start})")
            facts.append(f"wrong outputs lie only in the last partial {nm} tile")
        else:
            parts.append(f"{len(coords)}/{D} {nm}s ({_ranges(coords)}), in {nm} tile(s) "
                         f"{tiles_bad} of {ntiles} (tile = {T} {nm}s)")
    return "; ".join(parts), facts


def _value_facts(g, ref, bad):
    gb, rb = g[bad], ref[bad]
    weak = "~" if gb.size == 1 else ""  # one wrong output is trivially "all equal"
    facts = []
    if gb.size and np.isfinite(gb[0]) and (gb == gb[0]).all():
        facts.append(f"{weak}every wrong output is exactly {gb[0]:g}")
    if rb.size and (rb < 0).all():
        facts.append(f"{weak}every wrong output has a negative expected value")
    elif rb.size and (rb > 0).all() and (gb < rb).all():
        facts.append(f"{weak}every wrong output has a positive expected value and is too small")
    return facts


def compare(level, case, got, ref64, scale, bound, args64=None):
    if not isinstance(got, np.ndarray):
        return CaseResult(case, "shape", f"returned {type(got).__name__}, expected an array of "
                                         f"shape {ref64.shape}", np.inf)
    if got.shape != ref64.shape:
        return CaseResult(case, "shape", f"returned shape {got.shape}, expected {ref64.shape}", np.inf)
    g = got.astype(np.float64)
    scale = np.broadcast_to(np.maximum(scale, TINY), ref64.shape)
    bound = np.broadcast_to(bound, ref64.shape)
    with np.errstate(all="ignore"):
        abserr = np.abs(g - ref64)
        abserr[g == ref64] = 0.0  # equal infinities
        abserr[np.isnan(abserr)] = np.inf
        err = abserr / scale
        use = np.where(abserr == 0, 0.0, abserr / bound)
    bad = abserr > bound
    worst = float(err.max()) if err.size else 0.0
    bound_use = float(use.max()) if use.size else 0.0
    if not bad.any():
        return CaseResult(case, "pass", "", worst, bound_use)
    nonfinite = ~np.isfinite(g) & np.isfinite(ref64)
    idx = np.unravel_index(int(np.argmax(err)), err.shape)
    d = (g - ref64)[bad & np.isfinite(g)]
    direction = ("too large" if (d > 0).all() else "too small" if (d < 0).all() else "mixed signs") if d.size else "non-finite"
    istr = ", ".join(map(str, idx))
    msg = (f"out[{istr}] expected {ref64[idx]:.6g} got {g[idx]:.6g} "
           f"(error {err[idx]:.2g} x scale, "
           + (f"{use[idx]:.3g}x the float32 error bound); " if bound[idx] > 0
              else "and this op is exact, so the answer must match bit for bit); ")
           + f"{bad.mean():.0%} of outputs wrong, "
           f"{direction}")
    if nonfinite.any():
        msg += f", {int(nonfinite.sum())} non-finite (nan/inf) where the answer is finite"
    where, facts = _locate(bad, level.out_tile, level.out_axes)
    facts += _value_facts(g, ref64, bad)
    if args64 is not None and level.explain is not None:
        facts += level.explain(args64, g, ref64, bad)
    msg += f"; wrong at: {where}"
    shown = [f.lstrip("~") for f in facts
             if not f.startswith(("only the last", "wrong outputs lie", "~only the last", "~wrong outputs lie"))]
    if shown:
        msg += "; " + "; ".join(shown)
    if not nonfinite.any() and worst < ROUNDING_CEILING:
        msg += (f". Error is rounding-sized (< {ROUNDING_CEILING:g} x scale), not a logic bug: "
                f"accumulate in float64 (np.zeros(n, dtype=np.float64)) and cast only the final result")
    return CaseResult(case, "nonfinite" if nonfinite.any() else "wrong", msg, worst, bound_use,
                      tuple(facts))


def _crash_msg(exc, source_lines):
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if f.filename == guard.KERNEL_FILENAME]
    where = ""
    if frames:
        ln = frames[-1].lineno
        src = source_lines[ln - 1].strip() if 0 < ln <= len(source_lines) else ""
        where = f" at line {ln} `{src}`"
    return f"{type(exc).__name__}: {exc}{where}"


def verify(level, source, holdout=False, timeout_s=None):
    lines = source.splitlines()
    src = lambda ln: lines[ln - 1].strip() if ln and 0 < ln <= len(lines) else ""
    rep = Report(level)
    viol = {}
    for ln, kind, what in static.check(source):
        viol[(ln, kind)] = Violation(ln, kind, what, src(ln))
    if any(k in ("syntax", "no-kernel") for _, k in viol):
        rep.violations = list(viol.values())
        rep.load_error = "; ".join(v.fix() for v in rep.violations if v.kind in ("syntax", "no-kernel"))
        return rep
    try:
        fn, rec = guard.load_kernel(source)
    except Exception as e:  # kernel module raised at import time
        rep.violations = list(viol.values())
        rep.load_error = _crash_msg(e, lines)
        return rep

    biggest = max(level.get_cases(holdout), key=lambda c: sum(getattr(a, "size", 0) for a in c.args))
    for case in level.get_cases(holdout):
        args64 = tuple(a.astype(np.float64) if isinstance(a, np.ndarray) else a for a in case.args)
        ref = np.asarray(level.ref(*args64), dtype=np.float64)
        scale = level.scale(args64, ref)
        bound = gamma(level.bound_k(args64)) * scale + level.abs_floor
        try:
            got = guard.call_kernel(fn, rec, case.args, timeout_s or CASE_TIMEOUT_S)
        except guard.KernelTimeout:
            rep.results.append(CaseResult(case, "timeout", f"timed out after {timeout_s or CASE_TIMEOUT_S:g}s", np.inf))
            continue
        except Exception as e:
            rep.results.append(CaseResult(case, "error", _crash_msg(e, lines), np.inf))
            continue
        rep.results.append(compare(level, case, got, ref, scale, bound, args64))
        if case is biggest:
            rep.cost = dict(case=case.name, ops=rec.ops, flops=rec.flops,
                            min_flops=level.min_flops(args64), hbm_read=rec.hbm_read,
                            hbm_write=rec.hbm_write,
                            min_read=sum(a.nbytes for a in case.args if isinstance(a, np.ndarray)),
                            min_write=ref.size * 4)

    for (ln, kind), what in rec.violations.items():
        viol.setdefault((ln, kind), Violation(ln, kind, what, src(ln)))
    rep.violations = _dedupe(viol.values())
    return rep


# on one line, a specific violation explains the generic one (np.sum also trips "add.reduce")
_SUBSUMES = {"sum-like": {"reduction"}, "max-like": {"reduction"}, "linalg": {"matmul-op", "reduction"},
             "fancy-index": {"whole-array"}, "newaxis": {"broadcast"}, "layout": {"whole-array"}}


def _dedupe(violations):
    by_line = {}
    for v in violations:
        by_line.setdefault(v.line, []).append(v)
    out = []
    for vs in by_line.values():
        kinds = {v.kind for v in vs}
        drop = set().union(*(_SUBSUMES.get(k, set()) for k in kinds))
        out += [v for v in vs if v.kind not in drop or v.kind in _SUBSUMES]
    return sorted(out, key=lambda v: (v.line or 0, v.kind))


def main(argv=None):
    p = argparse.ArgumentParser(description="Verify a kernel file against a ladder level.")
    p.add_argument("level", type=int)
    p.add_argument("kernel_file")
    p.add_argument("-v", "--verbose", action="store_true", help="print every case")
    p.add_argument("--fixes", action="store_true", help="print the repair instruction per violation")
    a = p.parse_args(argv)
    rep = verify(LEVELS[a.level], open(a.kernel_file).read())
    print(rep.text())
    if a.fixes:
        for v in rep.violations:
            print(f"FIX line {v.line}: {v.fix()}")
    if a.verbose:
        for r in rep.results:
            print(f"  {r.status:9s} err/scale={r.worst:.2e} bound_use={r.bound_use:.2f}  {r.case.name}")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
