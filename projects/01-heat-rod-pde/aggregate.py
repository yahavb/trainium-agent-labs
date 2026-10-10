#!/usr/bin/env python3
"""aggregate.py -- build next-round prompts from ALL candidates of a round: dedupe, parts-aware ranking, a per-term
table of what each candidate got right, and four prompt variants. Never prints a target value."""
import collections
import json
import sys

import numpy as np
import sympy as sp

import pdecheck
import level0_heatrod as L0
import level1_heatrod as L1
from pdecheck import x, t

LEVELS = {0: L0, 1: L1}
LABELS = "ABCDEFGH"


def term_report(p, u, n_max=8, rel=0.02):
    """Per-wave verdicts with the checker's own projections and thresholds (what pdecheck.term_report
    would return). Direction only: 'ok', 'too large', 'too small', 'wrong sign', 'missing',
    'should not be there'; rate: 'ok', 'too fast', 'too slow', or None when the term is absent."""
    L, k = float(p["L"]), p["k"]
    grid = np.linspace(0, L, 801)
    f0 = pdecheck._num(p["f"], grid, 0 * grid)
    ug = pdecheck._num(u, grid, 0 * grid)
    lam1 = float(p["lam"](1))
    t1 = 0.5 / max(float(k) * lam1 * lam1, 1e-12)
    u1 = pdecheck._num(u, grid, t1 + 0 * grid)
    out = []
    for m in range(1, n_max + 1):
        b = pdecheck._num(p["basis"](m), grid, 0 * grid)
        norm = float(np.trapezoid(b * b, grid)) or 1.0
        got, want = float(np.trapezoid(ug * b, grid)) / norm, float(np.trapezoid(f0 * b, grid)) / norm
        if abs(got - want) <= rel * max(abs(want), 1e-12) + 1e-9:
            coef = "ok" if abs(want) > 1e-9 else None
        elif abs(want) < 1e-9:
            coef = "should not be there"
        elif abs(got) < 1e-9:
            coef = "missing"
        elif got * want < 0:
            coef = "wrong sign"
        else:
            coef = "too large" if abs(got) > abs(want) else "too small"
        rate = None
        c1 = float(np.trapezoid(u1 * b, grid)) / norm
        if abs(got) > 1e-6 and c1 / got > 0:
            r_got, r_want = -np.log(c1 / got) / t1, float(k) * float(p["lam"](m)) ** 2
            rate = "ok" if abs(r_got - r_want) <= 0.02 * r_want else ("too fast" if r_got > r_want else "too slow")
        if coef or rate:
            out.append(dict(n=m, wave=sp.sstr(p["basis"](m)), coef=coef, rate=rate))
    return out


def fingerprint(expr, p):
    """Numeric identity: equal on a fixed grid to 4 significant digits."""
    X, T = np.meshgrid(np.linspace(0, float(p["L"]), 7), np.linspace(0, 0.05, 3))
    v = pdecheck._num(expr, X.ravel(), T.ravel())
    return tuple(float(f"{z:.4g}") for z in v)


def rank_key(g):
    """Parts-aware: reward, then checks passed, then start-shape error, then fewer flagged terms."""
    flagged = sum((tr["coef"] not in ("ok", None)) + (tr["rate"] not in ("ok", None)) for tr in g.get("terms", []))
    se = g["start_error"] if g["start_error"] is not None else 9e9
    return (g["reward"], sum(g["parts"].values()), -se, -flagged)


def grade(p, answer):
    g = pdecheck.check(p, answer)
    g["terms"] = []
    if g["expr"] and g["parts"]:
        try:
            g["terms"] = term_report(p, pdecheck.parse(g["expr"]))
        except Exception:
            pass
    return g


def groups_of(p, graded):
    """Dedupe numerically identical answers; drop unparseable ones into a count."""
    groups, broken = collections.OrderedDict(), []
    for i, g in enumerate(graded):
        if not g["parts"]:
            broken.append(g)
            continue
        fp = fingerprint(pdecheck.parse(g["expr"]), p)
        groups.setdefault(fp, []).append(g)
    reps = sorted(((gs[0], len(gs)) for gs in groups.values()), key=lambda r: rank_key(r[0]), reverse=True)
    return reps, broken


def describe(g):
    ok = [k for k, v in g["parts"].items() if v]
    bad = [k for k, v in g["parts"].items() if not v]
    names = dict(equation="the equation", left_bc="the left end", right_bc="the right end",
                 start_shape="the starting shape")
    s = f"{', '.join(names[k] for k in ok) or 'nothing'} pass{'es' if len(ok) == 1 else ''}"
    if bad:
        s += f"; {', '.join(names[k] for k in bad)} fail{'s' if len(bad) == 1 else ''}"
    if g["start_error"] is not None:
        s += f" (starting shape off by {g['start_error'] * 100:.2f}%)"
    return s


def term_table(reps):
    """wave -> which candidates had the coefficient right / the rate right."""
    tab = collections.OrderedDict()
    for (g, _), lab in zip(reps, LABELS):
        for tr in g["terms"]:
            row = tab.setdefault(tr["wave"], dict(coef=[], rate=[], n=tr["n"]))
            if tr["coef"] == "ok":
                row["coef"].append(lab)
            if tr["rate"] == "ok":
                row["rate"].append(lab)
    return sorted(tab.items(), key=lambda kv: kv[1]["n"])


def render(base_prompt, reps, broken, memory=None, variant="V1", max_shown=3):
    lines = [base_prompt, "", f"{sum(c for _, c in reps) + len(broken)} attempts were checked. What each got right and wrong:"]
    for (g, count), lab in list(zip(reps, LABELS))[:max_shown]:
        dup = f", given by {count} attempts" if count > 1 else ""
        lines += ["", f"Attempt {lab} (reward {g['reward']:.1f}{dup}): {describe(g)}.",
                  f"  u(x, t) = {g['expr'][:400]}", f"  Checker: {g['feedback'][:600]}"]
    if broken:
        lines += ["", f"{len(broken)} attempt(s) gave no usable final line"
                  + (" (cut off before the end: keep the working short)." if any(b.get("cut") for b in broken) else ".")]
    tab = term_table(reps[:max_shown])
    useful = [(w, r) for w, r in tab if r["coef"] or r["rate"]]
    if useful and variant in ("V1", "V2"):
        lines += ["", "Term by term, across the attempts:"]
        for w, r in useful:
            parts = []
            if r["coef"]:
                parts.append(f"coefficient right in {', '.join(r['coef'])}")
            if r["rate"]:
                parts.append(f"decay rate right in {', '.join(r['rate'])}")
            lines.append(f"  {w}: {'; '.join(parts)}")
        coef_src = collections.Counter(l for _, r in useful for l in r["coef"]).most_common(1)
        rate_src = collections.Counter(l for _, r in useful for l in r["rate"]).most_common(1)
        if coef_src and rate_src and coef_src[0][0] != rate_src[0][0]:
            lines.append(f"So: keep the coefficients from {coef_src[0][0]}, each on its own sine, and build "
                         f"every exponent the way {rate_src[0][0]} did.")
    if variant == "V2":
        lines.append("Change only what the checker flagged. Copy everything else exactly.")
    lines.append("Write the corrected answer.")
    return "\n".join(lines)


def splice(p, reps):
    """Optional controller splice: per wave, a coefficient from a candidate whose coefficient was right
    and a rate from a candidate whose rate was right. Uses only the per-term verdicts.
    The caller grades it and KEEPS IT ONLY IF IT SCORES 1.0: a partial splice can score 0.8 with the
    starting shape 100% off (round 2 of the logged 1.3 run), the same reward trap as before."""
    pieces = {}
    for g, _ in reps:
        for term in sp.Add.make_args(sp.expand(pdecheck.parse(g["expr"]))):
            sins = [f for f in sp.Mul.make_args(term) if isinstance(f, sp.sin)]
            exps = [f for f in sp.Mul.make_args(term) if isinstance(f, sp.exp)]
            if len(sins) != 1:
                continue
            wave = sp.sstr(sins[0])
            verdict = next((tr for tr in g["terms"] if tr["wave"] == wave), None)
            if verdict is None:
                continue
            c = sp.simplify(term / sins[0] / sp.Mul(*exps))
            r = sp.simplify(-sum(e.args[0] for e in exps) / t) if exps else 0
            slot = pieces.setdefault(wave, dict(sin=sins[0]))
            if verdict["coef"] == "ok" and "c" not in slot:
                slot["c"] = c
            if verdict["rate"] == "ok" and "r" not in slot:
                slot["r"] = r
    full = [s for s in pieces.values() if "c" in s and "r" in s]
    if not full:
        return None
    return sp.Add(*(s["c"] * sp.exp(-s["r"] * t) * s["sin"] for s in full))




EXAMPLES = [pdecheck.EXAMPLE,                                                              # k=2, L=1
            "4*exp(-3*pi**2*t/25)*sin(pi*x/5) + 0.2*exp(-12*pi**2*t/25)*sin(2*pi*x/5)",    # k=3, L=5
            "1.5*exp(-5*pi**2*t/49)*sin(pi*x/7) - 0.7*exp(-20*pi**2*t/49)*sin(2*pi*x/7)"]  # k=5, L=7
EXAMPLE_K = [2, 3, 5]
V4_FIX = False          # Change B: set by agent.py --v4-fix


def other_example(p):
    """A worked example whose k differs from the problem's. Its waves match no level-0/1 problem."""
    for ex, k in zip(EXAMPLES, EXAMPLE_K):
        if sp.Integer(k) != p["k"] and (ex != EXAMPLES[0] or not V4_FIX):
            return ex
    return EXAMPLES[1]


def _with_terms(p, g):
    if "terms" not in g:
        g["terms"] = []
        if g.get("expr") and g.get("parts"):
            try:
                g["terms"] = term_report(p, pdecheck.parse(g["expr"]))
            except Exception:
                pass
    return g


def _insert_before_last(prompt, block):
    """Put a block before the final instruction line."""
    if not block:
        return prompt
    head, _, last = prompt.rpartition("\n")
    return f"{head}{block}\n{last}"


def build_round_prompts(a, problem, base_prompt, graded, best_ever):
    """Four different prompts from all of this round's candidates plus the best-ever one.
    V1 full aggregate, V2 aggregate + change-only-what-was-flagged, V3 best-ever only with a keep
    line, V4 a fresh start with another worked example at a higher temperature."""
    p = problem
    pool = [_with_terms(p, g) for g in graded]
    if best_ever is not None and all(best_ever is not g for g in graded):
        pool.append(_with_terms(p, best_ever))
    reps, broken = groups_of(p, pool)
    plan = []
    v1 = render(base_prompt, reps, broken, variant="V1")
    v2 = render(base_prompt, reps, broken, variant="V2")
    ver = ""
    if getattr(a, "verify_rates", False):
        for g in pool:
            g.setdefault("verify", verify_rates(p, g))
        ver = render_verification(p, reps)
        v1 = _insert_before_last(v1, ver)
        v2 = _insert_before_last(v2, ver)
    b = best_ever
    keep = ""
    if b["parts"].get("start_shape") and not b["parts"].get("equation"):
        keep = ("\nIts starting shape already matches: keep every coefficient and every sine exactly "
                "as written. Only the exponents are wrong.")
    elif b["parts"].get("equation") and not b["parts"].get("start_shape"):
        keep = ("\nEvery term already satisfies the equation: keep each exponent paired with its own "
                "sine. Only the coefficients, and which waves appear, need to change.")
    v3 = (f"{base_prompt}\n\nYour best attempt so far (reward {b['reward']:.1f}):\n  u(x, t) = {b['expr']}\n"
          f"A checker found this problem with it: {b['feedback']}{keep}"
          + (render_verification(p, [(b, 1)]) if ver else "") + "\nFix it.")
    v4 = pdecheck.prompt_of(p, example=other_example(p))
    variants = {"V1": (v1, None), "V2": (v2, None), "V3": (v3, None), "V4": (v4, 0.9)}
    order = ["V1", "V2", "V3", "V4"]
    for i in range(a.samples):
        name = order[i % len(order)]
        plan.append((name, *variants[name]))
    return plan


def try_splice(problem, graded, best_ever):
    """A candidate built from per-term verdicts. Returned only if it scores 1.0; otherwise None
    (a partial splice can score 0.8 with the starting shape 100% off)."""
    p = problem
    pool = [_with_terms(p, g) for g in graded if g.get("parts")]
    if best_ever is not None and best_ever.get("parts"):
        pool.append(_with_terms(p, best_ever))
    if not pool:
        return None
    try:
        reps, _ = groups_of(p, pool)
        s = splice(p, reps)
    except Exception:
        return None
    if s is None:
        return None
    g = pdecheck.check(p, "u(x, t) = " + sp.sstr(s))
    return g if g["reward"] == 1.0 else None


def build_spec_prompts(a, problem, base_prompt, history, best_ever):
    """Idea 2 for the spec modes. Short JSON replies barely vary with temperature (measured: four
    byte-identical specs at 0.6-1.0), so diversity has to come from the prompts.
    history: every graded attempt so far, each with its spec and feedback."""
    tried = collections.OrderedDict()
    for g in history:
        fam = (g.get("spec") or {}).get("eigenfunction")
        if fam and fam not in tried:
            failed = [k for k, v in (g.get("parts") or {}).items() if not v]
            tried[fam] = (g["reward"], failed, g["feedback"])
    names = dict(equation="the equation", left_bc="the left end", right_bc="the right end",
                 start_shape="the starting shape")
    lines = [f"  {fam}: reward {r:.1f}; fails {', '.join(names[k] for k in failed) or 'nothing'}"
             for fam, (r, failed, _) in tried.items()]
    b = best_ever
    v1 = (f"{base_prompt}\n\nYour best attempt so far (reward {b['reward']:.1f}) chose:\n  "
          f"{json.dumps(b.get('spec'))}\nThe checker's verdict on the solver's answer:\n"
          f"{__import__('agent').compact_feedback(b)}\nFix the spec.")
    v2 = (f"{base_prompt}\n\nEigenfunction families already tried, and what failed:\n" + "\n".join(lines)
          + "\nChoose a family that is not in this list.")
    v3 = (f"{base_prompt}\n\nBefore you answer, check your eigenfunction X_n(x) against both end "
          f"conditions: work out X_n at every end where u = 0 and X_n' at every end where u_x = 0, "
          f"for every n = 1, 2, 3, ... Both must be 0.")
    variants = [("S1", v1, None), ("S2", v2, None), ("S3", v3, None), ("S4", base_prompt, 1.0)]
    return [variants[i % len(variants)] for i in range(a.samples)]


# ---------------------------------------------------------------- Change A: rate verification
def _pretty_freq(w):
    """pi/2 rather than 1.5707963..."""
    try:
        q = sp.nsimplify(sp.N(w) / sp.pi, rational=True, tolerance=1e-6)
        if q.q <= 64:
            return sp.sstr(q * sp.pi)
    except Exception:
        pass
    return sp.sstr(w)


def verify_rates(problem, g):
    """Decay-rate consistency for every term of one candidate: is (rate in the exponent) equal to
    k * (spatial frequency of that term's sine or cosine)**2 ? Reads a parsed copy of the
    candidate's expression; never edits it and never states the correct exponent.
    Returns [{term, freq, rate, implied_freq, ok}] or [] when the answer did not parse."""
    if not g.get("expr") or not g.get("parts"):
        return []
    try:
        u = sp.expand(pdecheck.parse(g["expr"]))
    except Exception:
        return []
    k = problem["k"]
    out = []
    for term in sp.Add.make_args(u):
        factors = sp.Mul.make_args(term)
        waves = [f for f in factors if isinstance(f, (sp.sin, sp.cos))]
        exps = [f for f in factors if isinstance(f, sp.exp)]
        if len(waves) != 1:
            continue
        w = sp.simplify(waves[0].args[0] / x)
        if w.has(x) or w.has(t):
            continue
        r = sp.simplify(-sum(e.args[0] for e in exps) / t) if exps else sp.Integer(0)
        if r.has(x) or r.has(t):
            continue
        want = k * w ** 2
        try:
            ok = abs(float(r) - float(want)) <= 1e-3 * max(abs(float(want)), 1e-12)
        except TypeError:
            continue
        implied = sp.sqrt(r / k) if float(r) > 0 else sp.Integer(0)
        out.append(dict(term=sp.sstr(waves[0]), freq=_pretty_freq(w), rate=sp.sstr(sp.nsimplify(r, [sp.pi]) if r.is_Float else r),
                        implied_freq=_pretty_freq(implied), ok=bool(ok)))
    return out


def render_verification(problem, reps):
    """Targeted feedback for every distinct candidate (not only the best)."""
    k = sp.sstr(problem["k"])
    lines = []
    for (g, count), lab in zip(reps, LABELS):
        bad = [v for v in g.get("verify", []) if not v["ok"]]
        if not bad:
            continue
        lines.append(f"Attempt {lab}: {len(bad)} term(s) fail the decay-rate check "
                     f"(decay rate == k * spatial_frequency**2, with k = {k}):")
        for v in bad:
            lines.append(f"  {v['term']}: spatial frequency {v['freq']}; decay rate in your exponent "
                         f"{v['rate']}, which corresponds to frequency {v['implied_freq']}, not {v['freq']}. "
                         f"FAILED.")
    if not lines:
        return ""
    return ("\n\nDecay-rate verification of the attempts above (computed from your own expressions):\n"
            + "\n".join(lines)
            + "\nRecalculate each failing exponent from its own sine's frequency. Keep each coefficient "
              "and each sine as they are.")
