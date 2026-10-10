#!/usr/bin/env python3
"""
stress.py -- a property-based stress test for the nkibench ladder, built on Hypothesis.

nkibench grades a kernel on a handful of fixed shapes filled with standard-normal values. That is
the right thing for the score the agent sees every round: the same inputs every time, so two rounds
are comparable. It is the wrong thing for deciding a level is SOLVED, because a kernel can fit the
fixed shapes and still be wrong on inputs the level promises to handle.

This file generates those inputs instead: shapes anywhere inside the level's contract, and hostile
values (zeros, huge and tiny magnitudes, large means, constant rows). When it finds a failure,
Hypothesis SHRINKS it to the smallest input that still fails, and that smallest input is the
feedback -- "fails at C,H,W=(1, 2, 3) pool=2" says where to look; "3 of 4 shapes passed" does not.

    python stress.py --selftest                              # prove the stress test first
    python stress.py --level 4 --check reference_level4.py   # should pass
    python stress.py --level 1 --check my_pool.py --examples 20

Three choices that matter:

  * DETERMINISTIC. derandomize=True, no example database: the same kernel always meets the same
    inputs, so two runs of this file can be compared like two runs of nkibench.
  * A SIMULATION BUDGET. Each case is a CPU simulation, about 0.1 s on a seat for these levels,
    and shrinking can ask for many. --max-sims caps the total; past the cap new cases are not run, so shrinking stops at the
    smallest failure found so far. Results are cached, so the final replay never re-simulates.
  * THE CONTRACT IS PER LEVEL, AND IT IS A JUDGEMENT. The reference kernels only promise what their
    asserts allow -- level 3 is one fixed shape, level 4 needs multiples of the tile sizes -- so
    generating outside that would fail correct kernels. CONTRACTS below says what each level
    promises and why. --beyond-contract opens levels 3 and 4 to ragged shapes, which the shipped
    references are EXPECTED to fail: that is a harder level, not a bug.
"""

import argparse
import sys
import textwrap
import time

import numpy as np

try:
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover
    sys.exit("stress.py needs Hypothesis:  pip install hypothesis")

import nkibench

# ---------------------------------------------------------------- hostile values
#
# The order matters: Hypothesis shrinks a sampled_from towards its FIRST element, so a failure that
# also happens on ordinary values is reported on ordinary values, and a failure reported on "offset"
# really needs the large mean.

VALUE_KINDS = ("normal", "zeros", "constant_rows", "negative", "large", "tiny", "offset")


def fill(shape, kind, seed):
    r = np.random.default_rng(seed)
    if kind == "zeros":
        return np.zeros(shape, np.float32)
    if kind == "constant_rows":       # every row one value: max == min, variance 0
        row = r.standard_normal(shape[0]).astype(np.float32)
        return np.broadcast_to(row.reshape((-1,) + (1,) * (len(shape) - 1)), shape).copy()
    x = r.standard_normal(shape).astype(np.float32)
    if kind == "negative":
        return -np.abs(x)
    if kind == "large":
        return (x * 1e4).astype(np.float32)
    if kind == "tiny":
        return (x * 1e-4).astype(np.float32)
    if kind == "offset":              # a large mean on small variation: where cancellation bites
        return (x + 100.0).astype(np.float32)
    return x


# ---------------------------------------------------------------- what each level promises

@st.composite
def _pool_specs(draw):
    # The NumPy reference truncates when H or W does not divide by the pool size
    # (x[:, :H // p * p, ...]), so ragged spatial sizes ARE in the contract -- and none of the four
    # fixed shapes exercises them. C <= 128 because channels sit on the partition axis.
    p = draw(st.integers(1, 4))
    return dict(shape=(draw(st.integers(1, 128)), draw(st.integers(p, 32)),
                       draw(st.integers(p, 32))), pool_size=p)


@st.composite
def _transpose_specs(draw):
    # Partition axis untouched, so P <= 128. F1 x F2 kept small: the reference issues one copy per
    # element, so simulation time grows with F1 * F2.
    f1, f2 = draw(st.integers(1, 8)), draw(st.integers(1, 8))
    return dict(shape=(draw(st.integers(1, 128)), f1 * f2), shape2D=(f1, f2))


@st.composite
def _matmul_tiled_specs(draw):
    # reference_level4.py asserts K, M multiples of 128 and N of 512. Up to 4 tiles each way, so
    # odd tile counts (3) appear -- the fixed shapes only ever use 1, 2 and 4.
    return dict(K=128 * draw(st.integers(1, 4)), M=128 * draw(st.integers(1, 4)),
                N=512 * draw(st.integers(1, 2)))


@st.composite
def _matmul_ragged_specs(draw, kmax, mmax, nmax):
    return dict(K=draw(st.integers(1, kmax)), M=draw(st.integers(1, mmax)),
                N=draw(st.integers(1, nmax)))


def _pool_args(spec, kind, seed):
    return (fill(spec["shape"], kind, seed), spec["pool_size"])


def _transpose_args(spec, kind, seed):
    return (fill(spec["shape"], kind, seed), spec["shape2D"])


def _matmul_args(spec, kind, seed):
    return (fill((spec["K"], spec["M"]), kind, seed),
            fill((spec["K"], spec["N"]), kind, seed + 1))


CONTRACTS = {
    1: dict(specs=_pool_specs(), args=_pool_args,
            why="any C <= 128, H and W up to 32 including sizes the pool does not divide, "
                "pool 1-4"),
    2: dict(specs=_transpose_specs(), args=_transpose_args,
            why="any P <= 128, F1 and F2 from 1 to 8"),
    3: dict(specs=st.just(dict(K=128, M=64, N=512)), args=_matmul_args,
            why="the one shape the level defines; only the values vary",
            beyond=_matmul_ragged_specs(128, 128, 512)),
    4: dict(specs=_matmul_tiled_specs(), args=_matmul_args,
            why="K and M multiples of 128 up to 512, N 512 or 1024",
            beyond=_matmul_ragged_specs(512, 512, 1024)),
}


# ---------------------------------------------------------------- one case, graded like agent.py

def check_case(kernel, level_n, args):
    """The same checks agent.grade() applies to a fixed shape. Returns a message, or None."""
    spec = nkibench.LEVELS[level_n]
    before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
    want = spec["ref"](*args)
    try:
        got, counted = nkibench.simulate_and_count(kernel, args)
    except nkibench.NkiMissing:
        raise
    except Exception as e:
        return f"RAISED during simulation: {type(e).__name__}: {str(e)[:300]}"
    m = (nkibench.check_inputs_untouched(before, args)
         or nkibench.describe_mismatch(got, want)
         or nkibench.check_traffic_bar(level_n, counted, args, want))
    hazards = [w for w in counted.get("warnings", []) if "incorrect results on hardware" in w]
    if hazards and not m:
        m = "CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
    return m


def fixed_shapes(kernel, level_n):
    """What nkibench sees: (passed, total, first failure)."""
    spec = nkibench.LEVELS[level_n]
    passed, first = 0, None
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level_n)
        m = check_case(kernel, level_n, args)
        if m is None:
            passed += 1
        elif first is None:
            first = (nkibench.label(case, level_n), m)
    return passed, len(spec["shapes"]), first


def what_is_unusual(level_n, spec):
    """Name what the failing input has that the fixed shapes never do.

    Measured on the selftest mutant: shrunk to C,H,W=(1, 2, 3) pool=2, describe_mismatch said
    "most elements are wrong, so this is the core arithmetic ... not an edge case" -- true of that
    tiny output and exactly backwards about the cause, which IS the ragged width. The fixed shapes
    are the agent's only experience, so the useful sentence is how this input differs from them.
    """
    notes = []
    if level_n == 1:
        C, H, W = spec["shape"]
        p = spec["pool_size"]
        for name, v in (("H", H), ("W", W)):
            if v % p:
                notes.append(f"{name}={v} is not a multiple of the pool size {p}, so the last "
                             f"{v % p} {'row' if name == 'H' else 'column'}(s) must be dropped and "
                             f"the window strides still use the full {name}={v}; every fixed "
                             f"shape divides evenly")
        if p == 1:
            notes.append("pool size 1: the output equals the input")
        if C == 1:
            notes.append("a single channel, C=1")
    elif level_n == 2:
        f1, f2 = spec["shape2D"]
        if 1 in (f1, f2):
            notes.append(f"shape2D=({f1}, {f2}) has a dimension of 1")
        if spec["shape"][0] == 1:
            notes.append("a single partition, P=1")
    elif level_n in (3, 4):
        tiles = dict(K=nkibench.PMAX, M=nkibench.GEMM_STATIONARY_FMAX,
                     N=nkibench.GEMM_MOVING_FMAX)
        for d, t in tiles.items():
            v = spec[d]
            if v % t:
                notes.append(f"{d}={v} is not a multiple of the {t} tile, so the last tile along "
                             f"{d} is partial ({v % t})")
            elif v // t == 3:
                notes.append(f"{d}={v} is 3 tiles; the fixed shapes only ever use 1, 2 or 4")
    return notes


def _label(level_n, spec, kind):
    lbl = nkibench.label(spec, level_n) if level_n in (1, 2) else \
        f"K={spec['K']} M={spec['M']} N={spec['N']}"
    return f"{lbl}, values={kind}"


# ---------------------------------------------------------------- the stress test

def stress(kernel, level_n, examples=50, max_sims=300, beyond=False, seed=0):
    """Returns a dict: ok, message (the agent-facing text), sims, cases, seconds, minimal."""
    contract = CONTRACTS[level_n]
    specs = contract["beyond"] if beyond else contract["specs"]
    cache, stats = {}, dict(sims=0, cases=0, last_fail=None)
    t0 = time.time()

    @settings(max_examples=examples, derandomize=True, database=None, deadline=None,
              phases=(Phase.generate, Phase.shrink), report_multiple_bugs=False,
              print_blob=False,
              suppress_health_check=list(HealthCheck))
    @given(spec=specs, kind=st.sampled_from(VALUE_KINDS), vseed=st.integers(0, 3))
    def prop(spec, kind, vseed):
        key = (repr(sorted(spec.items())), kind, vseed)
        stats["cases"] += 1
        if key not in cache:
            if stats["sims"] >= max_sims:
                return            # budget spent: stop exploring, keep the smallest failure so far
            stats["sims"] += 1
            args = contract["args"](spec, kind, seed * 1000 + vseed)
            cache[key] = check_case(kernel, level_n, args)
        m = cache[key]
        if m is not None:
            stats["last_fail"] = (spec, kind, m)
            raise AssertionError(m)

    try:
        prop()
        ok = True
    except AssertionError:
        ok = False
    secs = time.time() - t0

    scope = ("BEYOND the level's contract (ragged shapes)" if beyond
             else f"within the level's contract: {contract['why']}")
    if ok:
        msg = (f"STRESS TEST level {level_n}: PASSED. {stats['sims']} generated inputs, {scope}, "
               f"with hostile values ({', '.join(VALUE_KINDS)}).")
        return dict(ok=True, message=msg, sims=stats["sims"], cases=stats["cases"],
                    seconds=secs, minimal=None)

    spec, kind, m = stats["last_fail"]
    msg = (f"STRESS TEST level {level_n}: FAILED. Among {stats['sims']} simulated inputs, {scope}, "
           f"the SMALLEST failing one found is\n  {_label(level_n, spec, kind)}")
    unusual = what_is_unusual(level_n, spec)
    if unusual:
        msg += ("\nWhat this input has that the fixed test shapes do not -- start here:\n"
                + "\n".join(f"  - {u}" for u in unusual))
    msg += f"\nOn it, the checker says:\n{textwrap.indent(m, '  ')}"
    if kind != VALUE_KINDS[0]:
        msg += (f"\n  It needs values={kind} to fail -- ordinary random values pass here -- so look "
                f"at how the kernel handles that kind of data, not at the tiling.")
    return dict(ok=False, message=msg, sims=stats["sims"], cases=stats["cases"],
                seconds=secs, minimal=(spec, kind))


# ---------------------------------------------------------------- selftest

# A realistic level-1 bug: compute the pool view's strides from W_out * pool instead of W. They are
# equal whenever the pool divides W -- which all four fixed shapes do -- so nkibench scores this
# kernel 1.0, and it is wrong on every ragged width.
_MUTANT_L1 = [("[sz_pool * sz_win, sz_hin // sz_pool]",
               "[sz_pool * (sz_win // sz_pool) * sz_pool, sz_hin // sz_pool]"),
              ("[sz_win, sz_pool],", "[(sz_win // sz_pool) * sz_pool, sz_pool],")]


def _mutant_level1(path="/tmp/_stress_mutant_level1.py"):
    src = open("reference_level1.py").read()
    for old, new in _MUTANT_L1:
        assert old in src, f"mutation anchor not found: {old}"
        src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    return path


def selftest(examples, max_sims):
    print("Proving the stress test before trusting it.\n")
    rc = 0

    # The value generators must produce what they claim.
    ok = (np.all(fill((3, 4), "zeros", 0) == 0)
          and np.all(np.ptp(fill((5, 6), "constant_rows", 0), axis=1) == 0)
          and np.all(fill((4, 4), "negative", 0) <= 0)
          and abs(float(fill((64, 64), "offset", 0).mean()) - 100) < 1)
    print(f"  value generators                          -> {'ok' if ok else 'FAIL'}")
    rc |= 0 if ok else 1

    # No false alarms: every shipped reference must pass inside its contract.
    print()
    for n in sorted(CONTRACTS):
        kernel = nkibench.load_kernel(f"reference_level{n}.py", nkibench.LEVELS[n]["entry"])
        r = stress(kernel, n, examples, max_sims)
        print(f"  reference level {n} passes its contract    -> "
              f"{'ok' if r['ok'] else 'FAIL'}  ({r['sims']} sims, {r['seconds']:.0f} s)")
        if not r["ok"]:
            print(textwrap.indent(r["message"], "      "))
        rc |= 0 if r["ok"] else 1

    # The point: a kernel nkibench scores 1.0 that the stress test catches, shrunk to a small input.
    print()
    kernel = nkibench.load_kernel(_mutant_level1(), nkibench.LEVELS[1]["entry"])
    passed, total, _ = fixed_shapes(kernel, 1)
    ok = passed == total
    print(f"  ragged-width mutant passes nkibench       -> {passed}/{total} fixed shapes "
          f"{'(as intended: the fixed shapes miss it)' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
    r = stress(kernel, 1, examples, max_sims)
    ok = not r["ok"]
    print(f"  ...and the stress test catches it         -> {'ok' if ok else 'FAIL'}  "
          f"({r['sims']} sims, {r['seconds']:.0f} s)")
    rc |= 0 if ok else 1
    if r["minimal"]:
        (spec, kind) = r["minimal"]
        C, H, W = spec["shape"]
        ragged = W % spec["pool_size"] != 0
        print(f"  ...shrunk to a ragged width               -> "
              f"{'ok' if ragged else 'FAIL'}  ({_label(1, spec, kind)})")
        rc |= 0 if ragged else 1
        print("\n  what the agent would be told:\n")
        print(textwrap.indent(r["message"], "    "))

    print("\nSTRESS SELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--level", type=int, choices=sorted(CONTRACTS))
    ap.add_argument("--check", metavar="FILE.py")
    ap.add_argument("--examples", type=int, default=50, help="generated cases before shrinking")
    ap.add_argument("--max-sims", type=int, default=300, help="cap on simulations, shrinking included")
    ap.add_argument("--beyond-contract", action="store_true",
                    help="levels 3-4: ragged shapes the shipped references do not handle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest(a.examples, a.max_sims))
    if not (a.level and a.check):
        ap.error("give --selftest, or --level N --check FILE.py")
    if a.beyond_contract and "beyond" not in CONTRACTS[a.level]:
        ap.error(f"level {a.level} has no beyond-contract space defined")

    kernel = nkibench.load_kernel(a.check, nkibench.LEVELS[a.level]["entry"])
    passed, total, first = fixed_shapes(kernel, a.level)
    print(f"nkibench fixed shapes: {passed}/{total} passed")
    if first:
        print(f"  first failure, on {first[0]}:\n{textwrap.indent(first[1], '    ')}")
    r = stress(kernel, a.level, a.examples, a.max_sims, a.beyond_contract, a.seed)
    print(f"\n{r['message']}\n\n({r['sims']} simulations, {r['cases']} cases including "
          f"shrinking, {r['seconds']:.0f} s)")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
