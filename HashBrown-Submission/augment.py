#!/usr/bin/env python3
"""
augment.py -- synthesizes additional, varied test inputs for nkibench.py, used only when
explicitly enabled (the --augment flag). The original checker's fixed shapes and seed=0
values stay reachable with no flag, per rule 7 in HARDENING.md.

Tiers (cumulative):

  A1   fresh random values each run, via a logged seed (see fresh_seed()). Still fully
       reproducible -- rule 8 -- because the seed actually used is always printed.
  A2a  extra shapes that still divide evenly, inside the level's declared shape class.
  A2b  ragged shapes: a prime dimension, a dimension of exactly 1, or (where the level's own
       declared class allows it) a size that doesn't divide the tile evenly. Still inside the
       declared class -- rule 9. Some levels' classes do not admit a ragged shape at all
       (level 4's reference literally asserts M/N/K are tile multiples); extra_shapes()
       returns [] for those, and that is itself logged as a finding, not patched around.
  A3   hostile values: large magnitude, large mean, zeros, negatives, TINY values, and one
       row of identical values -- see hostile_args() for why "tiny" catches
       cheats/c6_almost_right.py and "large magnitude" does not.
  A4   run the kernel twice on two DIFFERENT random instantiations of the SAME shape and fail
       if the two outputs are identical. The harness does not own the output buffer (every
       kernel allocates and returns its own, see HARDENING.md Phase 1 "Output allocation"),
       so there is no buffer to pre-fill with NaN and check for leftovers -- this is the only
       of the two A4 designs the plan described that actually applies here.
"""

import numpy as np

import nkibench


def fresh_seed():
    """A1: a seed that varies run to run, but is always logged so the run is reproducible
    after the fact from the printed value -- rule 8."""
    import time
    return int(time.time_ns()) % 1_000_000


# ---------------------------------------------------------------- A2a / A2b: extra shapes

def extra_shapes(level_n, tier):
    """tier: 'a2a' (divides evenly, more coverage) or 'a2b' (ragged: prime dim, dim of 1, or
    a non-tile-multiple size), both strictly inside the level's own declared shape class.

    Returns a list of shape-spec dicts in the same form as nkibench.LEVELS[n]['shapes'].
    """
    if level_n == 1:  # avgpool: shape=(C, H, W), pool_size. H and W must divide pool_size's
                      # window if the "ragged" case is to test tile boundaries rather than
                      # the reference's own truncate-the-remainder behaviour (see Findings log).
        if tier == "a2a":
            return [dict(shape=(16, 20, 20), pool_size=5)]
        if tier == "a2b":
            return [dict(shape=(1, 8, 8), pool_size=2)]  # C=1: a dimension of exactly 1
        return []
    if level_n == 2:  # transpose: shape=(sz_p, F), shape2D=(F1, F2)
        if tier == "a2a":
            return [dict(shape=(16, 24), shape2D=(6, 4))]
        if tier == "a2b":
            return [dict(shape=(37, 12), shape2D=(3, 4)),   # sz_p=37 is prime
                    dict(shape=(1, 12), shape2D=(3, 4)),     # sz_p=1: a dimension of exactly 1
                    dict(shape=(32, 37), shape2D=(37, 1))]   # F1=37 is prime; F2=1
        return []
    if level_n == 3:  # matmul single tile: K<=128, M<=128, N<=512 (no tiling, so "ragged" just
                      # means a prime/degenerate size, not a partial-tile boundary)
        if tier == "a2a":
            return [dict(K=96, M=48, N=384)]
        if tier == "a2b":
            return [dict(K=37, M=1, N=17)]  # K, N prime; M=1
        return []
    if level_n == 4:  # matmul tiled: K and M must be multiples of 128, N a multiple of 512 --
                      # reference_level4.py itself asserts this (lines 48-53). The declared
                      # class EXCLUDES non-multiples, so no in-class ragged shape exists.
        if tier == "a2a":
            return [dict(K=384, M=384, N=512)]
        if tier == "a2b":
            return []  # deliberately empty -- see module docstring and Findings log
        return []
    return []


# ---------------------------------------------------------------- A3: hostile values

HOSTILE_KINDS = ("large_magnitude", "large_mean", "zeros", "negatives", "tiny", "identical_row")


def _hostile_array(a, kind, rng):
    shape, dtype = a.shape, a.dtype
    if kind == "large_magnitude":
        return (rng.choice([-1.0, 1.0], size=shape) * (1e4 + rng.standard_normal(shape))).astype(dtype)
    if kind == "large_mean":
        return (1e4 + 0.01 * rng.standard_normal(shape)).astype(dtype)
    if kind == "zeros":
        return np.zeros(shape, dtype)
    if kind == "negatives":
        return (-np.abs(rng.standard_normal(shape))).astype(dtype)
    if kind == "tiny":
        # 1e-6 to 1e-3: deliberately overlapping cheats/c6_almost_right.py's fixed +1e-3
        # offset. At this magnitude the SAME absolute offset becomes a huge relative error --
        # see HARDENING.md Findings log for why "large magnitude" does the opposite.
        mag = rng.uniform(1e-6, 1e-3, size=shape)
        sign = rng.choice([-1.0, 1.0], size=shape)
        return (mag * sign).astype(dtype)
    if kind == "identical_row":
        out = rng.standard_normal(shape).astype(dtype)
        out[0] = out[0].flat[0]  # the whole first partition row becomes one repeated value
        return out
    raise ValueError(kind)


def hostile_args(args, seed):
    """args: an argument tuple as built by nkibench.make_inputs for one shape. Returns
    {kind: new_args} -- every ndarray in args replaced by a same-shape, same-dtype hostile
    array; non-array args (pool_size, shape2D, ...) pass through unchanged."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in HOSTILE_KINDS:
        out[kind] = tuple(_hostile_array(a, kind, rng) if isinstance(a, np.ndarray) else a
                           for a in args)
    return out


# ---------------------------------------------------------------- A4: output-sensitivity

def output_changes_with_input(kernel, level_n, case, seed):
    """Run the kernel on two DIFFERENT random instantiations of the SAME shape. Returns
    (changed: bool, out1, out2). If not changed, the kernel's output does not depend on its
    input -- do_nothing or constant_output, whichever it is.

    np.allclose(..., equal_nan=True) means two all-NaN outputs (an UNWRITTEN
    nl.shared_hbm buffer, per cheats/c1_do_nothing.py's measured behaviour) also count as
    "did not change" -- correctly, since neither run computed anything from its input.
    """
    args1, _ = nkibench.make_inputs(case, level_n, seed)
    args2, _ = nkibench.make_inputs(case, level_n, seed + 10_007)
    out1, _ = nkibench.simulate_and_count(kernel, args1)
    out2, _ = nkibench.simulate_and_count(kernel, args2)
    same = np.allclose(np.asarray(out1, np.float64), np.asarray(out2, np.float64),
                        equal_nan=True)
    return (not same), out1, out2
