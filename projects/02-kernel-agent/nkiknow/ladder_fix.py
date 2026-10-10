"""
Opt-in corrected registration of nkibench levels 5-7.  nkibench.py is NOT edited.

    import sys; sys.path.insert(0, "projects/02-kernel-agent")
    from nkiknow import ladder_fix; ladder_fix.apply()      # before using nkibench / agent

Why: on the organizer's test shapes (all K,M,N <= 512 x 512 x 1024) AWS's own tutorial kernels cannot
meet the level bars (hoist 1.9x > 1.6x; block-free-dim asserts on 2 of 4 shapes and is 1.3x > 1.25x;
fully-optimized asserts on every shape). Blocking only pays when there is something to reuse.

What apply() does
  * keeps the organizer's four small shapes on levels 5-7 for CORRECTNESS ONLY (no traffic bar);
  * adds two large shapes where blocking pays, and applies a PER-SHAPE traffic bar on those only,
    by wrapping nkibench.check_traffic_bar (nkibench's own check applies one bar to every shape);
  * sets LEVELS[n]["max_waste"] to the largest per-shape limit, for display.

Limits = deterministic traffic ratio of the AWS tutorial design on that shape (measured on seat-135,
nki 0.6.0 simulator, float32, 2026-10-10) x MARGIN.  Byte counts from the simulator are exact and
repeatable, so the margin is not for noise; it lets an equivalent-but-not-identical kernel (a few
extra small copies) pass.  5% is below the smallest gap between adjacent levels on the large shapes
(L4/L5 = 1.105 on the (2048,2048,1024) shape), so a lower level's tutorial kernel still fails.
"""
import copy

MARGIN = 1.05

SMALL_SHAPES = [dict(K=128, M=128, N=512), dict(K=256, M=256, N=1024),
                dict(K=512, M=128, N=512), dict(K=256, M=512, N=1024)]   # organizer's _MM_SHAPES
LARGE_SHAPES = [dict(K=2048, M=2048, N=2048), dict(K=2048, M=2048, N=1024)]

# Measured tutorial ratios (bytes moved / byte floor) on the large shapes, keyed (K, M, N).
MEASURED = {
    5: {(2048, 2048, 2048): 6.000, (2048, 2048, 1024): 4.750},   # nki_matmul_hoist_load_
    6: {(2048, 2048, 2048): 3.333, (2048, 2048, 1024): 2.750},   # nki_matmul_block_free_dimension_
    7: {(2048, 2048, 2048): 1.333, (2048, 2048, 1024): 1.000},   # nki_matmul_fully_optimized_ (blocks 16/2/8 capped by shape)
}
LIMITS = {n: {s: r * MARGIN for s, r in d.items()} for n, d in MEASURED.items()}

ORIGINAL = {}          # n -> deep copy of the organizer's LEVELS[n], filled on first apply()
_orig_check = None


def apply():
    """Idempotent. Returns the patched nkibench module."""
    global _orig_check
    import nkibench
    if _orig_check is None:
        _orig_check = nkibench.check_traffic_bar
        for n in (5, 6, 7):
            ORIGINAL[n] = copy.deepcopy(nkibench.LEVELS[n])
    for n in (5, 6, 7):
        nkibench.LEVELS[n]["shapes"] = [dict(s) for s in SMALL_SHAPES + LARGE_SHAPES]
        nkibench.LEVELS[n]["max_waste"] = max(LIMITS[n].values())

    def check_traffic_bar(level_n, counted, args, want):
        if level_n not in LIMITS:
            return _orig_check(level_n, counted, args, want)
        key = (args[0].shape[0], args[0].shape[1], args[1].shape[1])    # (K, M, N)
        limit = LIMITS[level_n].get(key)
        if limit is None or not counted.get("bytes"):
            return None                      # small shapes: correctness only
        old = nkibench.LEVELS[level_n]["max_waste"]
        nkibench.LEVELS[level_n]["max_waste"] = limit
        try:
            return _orig_check(level_n, counted, args, want)
        finally:
            nkibench.LEVELS[level_n]["max_waste"] = old

    nkibench.check_traffic_bar = check_traffic_bar
    return nkibench


def restore():
    """Undo apply(): original shapes, limits and checker."""
    import nkibench
    if _orig_check is not None:
        nkibench.check_traffic_bar = _orig_check
        for n, lv in ORIGINAL.items():
            nkibench.LEVELS[n].update(copy.deepcopy(lv))
