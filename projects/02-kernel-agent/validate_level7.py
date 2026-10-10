"""Check the level-7 reference, and measure what actually separates levels 5, 6 and 7.

Requires the pod's NKI simulator. Calls no model and makes no claim about device latency.

This validator reports BYTES and TRANSFERS side by side because the registered bars only
constrain bytes, and bytes have already bottomed out by level 6. It records that as a finding
instead of asserting a previous-level-fails check that is not true here.
"""
from pathlib import Path

import nkibench

LEVELS_UNDER_TEST = (5, 6, 7)


def validate():
    folder = Path(__file__).resolve().parent
    kernels = {
        level: nkibench.load_kernel(str(folder / f"reference_level{level}.py"),
                                    nkibench.LEVELS[level]["entry"])
        for level in LEVELS_UNDER_TEST
    }

    for level in LEVELS_UNDER_TEST:
        source = (folder / f"reference_level{level}.py").read_text()
        violations = nkibench.check_rules(source, level)
        if violations:
            raise AssertionError((level, violations))

    bar7 = nkibench.LEVELS[7]["max_waste"]
    worst_ratio = {level: 0.0 for level in LEVELS_UNDER_TEST}
    total_transfers = {level: 0 for level in LEVELS_UNDER_TEST}

    print(f"{'K':>4} {'M':>4} {'N':>5} | {'L5 ratio':>8} {'L6 ratio':>8} {'L7 ratio':>8} | "
          f"{'L5 xfer':>7} {'L6 xfer':>7} {'L7 xfer':>7}")
    for case in nkibench.LEVELS[7]["shapes"]:
        ratios, xfers = {}, {}
        for seed in (0, 1, 2):
            for level, kernel in kernels.items():
                args, _ = nkibench.make_inputs(case, 7, seed=seed)
                before = [value.copy() for value in args]
                want = nkibench.LEVELS[7]["ref"](*args)
                got, counted = nkibench.simulate_and_count(kernel, args)

                failures = [nkibench.describe_mismatch(got, want),
                            nkibench.check_inputs_untouched(before, args)]
                failures += [w for w in counted.get("warnings", [])
                             if "incorrect results on hardware" in w]
                failures = [f for f in failures if f]
                if failures:
                    raise AssertionError((level, case, seed, failures))

                floor = nkibench.minimum_hbm_bytes(args, want)
                ratios[level] = counted["bytes"] / floor
                xfers[level] = counted["transfers"]
                worst_ratio[level] = max(worst_ratio[level], ratios[level])
                total_transfers[level] += counted["transfers"]

            # Level 7 must meet its own registered bar, and must not regress on either metric.
            traffic_failure = nkibench.check_traffic_bar(7, counted, args, want)
            if traffic_failure:
                raise AssertionError((case, seed, traffic_failure))
            if ratios[7] > ratios[6] + 1e-9:
                raise AssertionError((case, "level 7 moved more bytes than level 6", ratios))
            if xfers[7] > xfers[6]:
                raise AssertionError((case, "level 7 used more transfers than level 6", xfers))

        print(f"{case['K']:>4} {case['M']:>4} {case['N']:>5} | "
              f"{ratios[5]:>7.3f}x {ratios[6]:>7.3f}x {ratios[7]:>7.3f}x | "
              f"{xfers[5]:>7} {xfers[6]:>7} {xfers[7]:>7}")

    print(f"\nPASS: {len(nkibench.LEVELS[7]['shapes'])} shapes x 3 seeds; numerics match the "
          f"reference, inputs preserved, level-7 bar ({bar7}x) met, and level 7 never costs more "
          f"bytes or more transfers than level 6.")
    print(f"worst-case byte ratio: L5 {worst_ratio[5]:.3f}x, L6 {worst_ratio[6]:.3f}x, "
          f"L7 {worst_ratio[7]:.3f}x (floor = 1.000x)")
    print(f"total transfers across all shapes/seeds: L5 {total_transfers[5]}, "
          f"L6 {total_transfers[6]}, L7 {total_transfers[7]}")

    print("\nFINDING on the ladder, from the measurements above:")
    if worst_ratio[6] <= bar7:
        print(f"  level 6's kernel already measures {worst_ratio[6]:.3f}x, inside level 7's "
              f"{bar7}x bar, so the\n  registered byte bar does NOT separate level 7 from level 6. "
              f"Bytes bottom out at the\n  floor (1.000x) and cannot be improved further.")
    if total_transfers[7] < total_transfers[6]:
        cut = 100 * (1 - total_transfers[7] / total_transfers[6])
        print(f"  What level 7 does improve is DMA transfer COUNT: {total_transfers[7]} vs "
              f"{total_transfers[6]}, a {cut:.0f}% cut,\n  for identical bytes. Blocking M lets a "
              f"whole block of lhsT tiles move in one transfer.")
        print("  RECOMMENDATION: to make level 7 a real rung, gate it on transfer count rather "
              "than on\n  bytes. check_traffic_bar() in nkibench.py only inspects "
              "counted['bytes'], though\n  counted['transfers'] is already measured and available.")


if __name__ == "__main__":
    validate()
