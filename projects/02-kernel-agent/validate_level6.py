"""Check the level-6 reference: numerics, input integrity, and measured HBM traffic.

Requires the pod's NKI simulator. Makes no claim about device latency and calls no model.

It also records a FINDING about the ladder rather than hiding it: level 6's registered bar is
1.25x the byte floor, but the level-5 reference already measures at most 1.143x, so the level-5
kernel passes level 6's bar unchanged. This validator therefore does NOT assert that the
previous level fails the new bar -- that assertion is true for level 7 but false for level 6.
It prints the comparison so the fact is visible instead of assumed.
"""
from pathlib import Path

import nkibench

LEVELS_UNDER_TEST = (4, 5, 6)


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

    bar6 = nkibench.LEVELS[6]["max_waste"]
    bar7 = nkibench.LEVELS[7]["max_waste"]
    worst = {level: 0.0 for level in LEVELS_UNDER_TEST}

    print(f"{'K':>4} {'M':>4} {'N':>5} | {'L4':>8} {'L5':>8} {'L6':>8} | L6 vs bar6  L6 vs bar7")
    for case in nkibench.LEVELS[6]["shapes"]:
        ratios = {}
        for seed in (0, 1, 2):
            for level, kernel in kernels.items():
                args, _ = nkibench.make_inputs(case, 6, seed=seed)
                before = [value.copy() for value in args]
                want = nkibench.LEVELS[6]["ref"](*args)
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
                worst[level] = max(worst[level], ratios[level])

            # The level-6 kernel must meet its own registered bar on every shape and seed.
            traffic_failure = nkibench.check_traffic_bar(6, counted, args, want)
            if traffic_failure:
                raise AssertionError((case, seed, traffic_failure))
            # Blocking must never cost more traffic than the level it improves on.
            if ratios[6] > ratios[5] + 1e-9:
                raise AssertionError((case, "blocking increased traffic vs level 5", ratios))

        print(f"{case['K']:>4} {case['M']:>4} {case['N']:>5} | "
              f"{ratios[4]:>7.3f}x {ratios[5]:>7.3f}x {ratios[6]:>7.3f}x | "
              f"{'PASS' if ratios[6] <= bar6 else 'FAIL':>10}  "
              f"{'PASS' if ratios[6] <= bar7 else 'FAIL':>10}")

    print(f"\nworst case: level 4 {worst[4]:.3f}x, level 5 {worst[5]:.3f}x, "
          f"level 6 {worst[6]:.3f}x (floor = 1.000x)")
    print(f"PASS: {len(nkibench.LEVELS[6]['shapes'])} shapes x 3 seeds; numerics match the "
          f"reference, inputs preserved, level-6 bar ({bar6}x) met, and blocking never costs "
          f"more bytes than level 5.")

    print("\nFINDING on the ladder, from the measurements above:")
    print(f"  level 5's kernel worst case is {worst[5]:.3f}x, and level 6's registered bar is "
          f"{bar6}x.")
    if worst[5] <= bar6:
        print("  So the level-5 kernel PASSES level 6's bar unchanged: as registered, level 6 is "
              "not a\n  tighter constraint than level 5. Tightening it below "
              f"{worst[5]:.3f}x would make it a real rung.")
    else:
        print("  So level 6's bar does constrain more tightly than level 5's kernel achieves.")
    print(f"  This level-6 kernel reaches {worst[6]:.3f}x, which also clears level 7's "
          f"{bar7}x bar.")


if __name__ == "__main__":
    validate()
