"""Would the kernels the agent "solved" survive tests it never saw?

Judges grade correctness on held-back shapes and hostile values, not on the 1-4 shapes the checker
shows the agent. This re-tests every distinct solved kernel in the attempt logs on extra cases:

  * SHAPES the level's contract allows but the checker never shows: sizes the reference truncates,
    pool size 1, one channel, non-square inputs, more tiles in K, M and N.
  * VALUES the checker never uses (it always draws standard normals): large magnitudes, a large
    offset, all negative, all zeros, a constant.

A case is kept only if THEIR reference kernel passes it ("fair"): judges are unlikely to test what
their own tutorial kernel cannot do. Cases the reference fails are reported separately.

    PYTHONDONTWRITEBYTECODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python hidden_eval.py attempts_*.jsonl
"""
import json, os, sys, collections
import numpy as np
import nkibench

# ------------------------------------------------------------------ the extra cases

SHAPES = {
    1: [dict(shape=(16, 17, 19), pool_size=2),     # H, W not divisible: the reference truncates
        dict(shape=(8, 8, 8), pool_size=1),        # pool 1 is the identity
        dict(shape=(1, 16, 16), pool_size=4),      # a single channel
        dict(shape=(16, 8, 24), pool_size=2),      # non-square
        dict(shape=(96, 12, 12), pool_size=3)],
    2: [dict(shape=(1, 12), shape2D=(3, 4)),       # one partition
        dict(shape=(16, 9), shape2D=(1, 9)),       # F1 = 1
        dict(shape=(16, 9), shape2D=(9, 1)),       # F2 = 1
        dict(shape=(128, 96), shape2D=(12, 8)),    # full partitions, non-square
        dict(shape=(5, 77), shape2D=(7, 11))],     # odd everything
    3: [dict(K=128, M=64, N=512)],                 # the reference asserts this one shape
    4: [dict(K=384, M=128, N=512),                 # three K chunks
        dict(K=128, M=384, N=512),                 # three M blocks
        dict(K=128, M=128, N=1536),                # three N blocks
        dict(K=256, M=256, N=512)],
}

VALUES = {                        # applied to every array input; the scale keeps fp32 sums finite
    "normal":   lambda a, r: a,
    "large":    lambda a, r: a * 1e3,
    "offset":   lambda a, r: a + 1e3,
    "negative": lambda a, r: -np.abs(a) - 1.0,
    "zeros":    lambda a, r: np.zeros_like(a),
    "constant": lambda a, r: np.full_like(a, 7.0),
}


def make_case(level, spec, vname, seed=123):
    r = np.random.default_rng(seed + level)
    args = nkibench.LEVELS[level]["make_args"](spec, r)
    return tuple(VALUES[vname](a, r).astype(a.dtype) if isinstance(a, np.ndarray) else a
                 for a in args)


def cases_for(level):
    """Extra shapes with normal values, plus every value pattern on the level's own first shape
    and on the extra shapes."""
    out = []
    shown = nkibench.LEVELS[level]["shapes"]
    for spec in SHAPES[level]:
        out.append((spec, "normal"))
    for spec in [shown[0]] + SHAPES[level]:
        for v in VALUES:
            if v != "normal":
                out.append((spec, v))
    return out


# ------------------------------------------------------------------ grading one kernel on one case

def run_case(kernel, level, spec, vname):
    args = make_case(level, spec, vname)
    before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    want = nkibench.LEVELS[level]["ref"](*args)
    try:
        got, _ = nkibench.simulate_and_count(kernel, args)
    except Exception as e:  # noqa: BLE001
        return f"raised {type(e).__name__}: {str(e)[:120]}"
    return (nkibench.check_inputs_untouched(before, args)
            or nkibench.describe_mismatch(got, want))


_n = [0]


def load(source, level):
    _n[0] += 1
    path = f"/tmp/_hidden_{os.getpid()}_{_n[0]}.py"
    with open(path, "w") as f:
        f.write(source)
    return nkibench.load_kernel(path, nkibench.LEVELS[level]["entry"])


def case_label(level, spec, vname):
    return f"{nkibench.label(spec, level)} [{vname}]"


# ------------------------------------------------------------------ main

def main(paths):
    solved = collections.defaultdict(dict)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["reward"] >= 1 - 1e-9 and r["code"].strip():
                solved[r["level"]].setdefault(r["code"], os.path.basename(p))
    report = {}
    for level in sorted(solved):
        ref_src = open(f"reference_level{level}.py").read()
        ref = load(ref_src, level)
        fair, unfair = [], []
        for spec, v in cases_for(level):
            (unfair if run_case(ref, level, spec, v) else fair).append((spec, v))
        print(f"\n===== level {level}: {len(solved[level])} distinct solved kernels; "
              f"{len(fair)} fair extra cases, {len(unfair)} the reference itself fails")
        for spec, v in unfair:
            print(f"   reference fails: {case_label(level, spec, v)}")
        fails_by_case = collections.Counter()
        per_kernel = []
        for code, src in solved[level].items():
            try:
                k = load(code, level)
            except Exception as e:  # noqa: BLE001
                per_kernel.append((src, None, [f"load: {e}"]))
                continue
            bad = []
            for spec, v in fair:
                m = run_case(k, level, spec, v)
                if m:
                    bad.append(case_label(level, spec, v) + ": " + m.splitlines()[0][:110])
                    fails_by_case[case_label(level, spec, v)] += 1
            per_kernel.append((src, len(fair) - len(bad), bad))
        survivors = sum(1 for _, ok, bad in per_kernel if ok is not None and not bad)
        print(f"   survive every fair case: {survivors} of {len(per_kernel)}")
        for c, n in fails_by_case.most_common():
            print(f"   {n:3d} kernels fail  {c}")
        for src, ok, bad in per_kernel:
            if bad:
                print(f"   - from {src}: passes {ok}/{len(fair)}; first failure: {bad[0]}")
        report[level] = dict(kernels=len(per_kernel), survivors=survivors, fair=len(fair),
                             unfair=[case_label(level, s, v) for s, v in unfair],
                             fails_by_case=dict(fails_by_case))
    json.dump(report, open("hidden_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
