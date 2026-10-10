"""Check the level-5 reference and its measured traffic against level 4.

Requires the pod's NKI simulator. Does not call a model or claim device latency.
Unlike nkibench.py's standalone --check path, this explicitly enforces the
traffic bar and input-integrity checks used by agent.grade().
"""

from pathlib import Path

import nkibench


def validate():
    folder = Path(__file__).resolve().parent
    kernels = {
        level: nkibench.load_kernel(str(folder / f"reference_level{level}.py"),
                                    nkibench.LEVELS[level]["entry"])
        for level in (4, 5)
    }
    for level in (4, 5):
        source = (folder / f"reference_level{level}.py").read_text()
        violations = nkibench.check_rules(source, level)
        if violations:
            raise AssertionError(violations)

    old_rejected = False
    print("K M N | level-4 bytes | level-5 bytes | level-5 / floor")
    for case in nkibench.LEVELS[5]["shapes"]:
        K, M, N = case["K"], case["M"], case["N"]
        measured = {}
        for seed in (0, 1, 2):
            for level, kernel in kernels.items():
                args, _ = nkibench.make_inputs(case, 5, seed=seed)
                before = [value.copy() for value in args]
                want = nkibench.LEVELS[5]["ref"](*args)
                got, counted = nkibench.simulate_and_count(kernel, args)
                failures = [
                    nkibench.describe_mismatch(got, want),
                    nkibench.check_inputs_untouched(before, args),
                ]
                failures += [warning for warning in counted.get("warnings", [])
                             if "incorrect results on hardware" in warning]
                if any(failures):
                    raise AssertionError((level, case, seed, failures))

                traffic_failure = nkibench.check_traffic_bar(5, counted, args, want)
                if level == 5:
                    if traffic_failure:
                        raise AssertionError((case, seed, traffic_failure))
                    # RHS read once, LHS read once per N slab, output written once.
                    expected = (K * M * (N // 512) + K * N + M * N) * args[0].itemsize
                    expected_transfers = ((N // 512) * (K // 128)
                                          + (N // 512) * (M // 128) * (K // 128)
                                          + (M // 128) * (N // 512))
                    if counted["bytes"] != expected or counted["transfers"] != expected_transfers:
                        raise AssertionError((case, counted, expected, expected_transfers))
                elif traffic_failure:
                    old_rejected = True
                measured[level] = counted["bytes"]
                floor = nkibench.minimum_hbm_bytes(args, want)
            if measured[5] > measured[4]:
                raise AssertionError((case, "hoisting increased traffic", measured))
        print(f"{K} {M} {N} | {measured[4]:,} | {measured[5]:,} | {measured[5] / floor:.3f}x")

    if not old_rejected:
        raise AssertionError("The level-4 control should fail the level-5 traffic bar.")
    print("PASS: four shapes, three seeds; inputs preserved; traffic formula and level-5 bar met.")
    print("The unchanged level-4 control fails the level-5 traffic bar on at least one shape.")


if __name__ == "__main__":
    validate()
