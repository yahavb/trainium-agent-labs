"""Structured evaluation for the minimum-traffic experiment.

One decision path for the CLI and the agent loop: `nkibench.accept_case` is the shared
acceptance function, and this module wraps it into a structured result with independent
fields instead of a single boolean. Bytes stay integers. The resource estimate is a static
estimate only; compiler or device results are what upgrade its status.

    from traffic_eval import evaluate_file
    r = evaluate_file("candidate.py", 5)
    print(r["accepted"], r["failure_kind"], r["feedback"])

    python3 traffic_eval.py --level 5 --check candidate.py
"""
import argparse
import json

import numpy as np

import nkibench


def _new_result(total):
    return dict(
        accepted=False, rules_ok=False, numerics_ok=False, inputs_ok=False,
        traffic_ok=False, hazards_ok=False, per_case=[], failure_kind=None,
        feedback="", passed=0, total=total, resource_status="unestimated",
        bytes_total=0, worst_waste=None,
    )


def evaluate_file(path, level_n, tol=2e-2, seed=0):
    """Evaluate one candidate file against one level. Never raises for a bad candidate."""
    spec = nkibench.LEVELS[level_n]
    result = _new_result(len(spec["shapes"]))

    try:
        src = open(path).read()
    except OSError as e:
        result["failure_kind"] = "environment"
        result["feedback"] = f"cannot read {path}: {e}"
        return result

    violations = nkibench.check_rules(src, level_n)
    if violations:
        result["failure_kind"] = "rules"
        result["feedback"] = ("Rule violations, which score zero however fast the kernel is. "
                              "Fix exactly these: " + " ".join(violations))
        return result
    result["rules_ok"] = True

    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        try:
            import nki  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        if sdk_present:
            # The SDK is here, so a missing module is the candidate's fault, not the environment's.
            result["failure_kind"] = "load"
            result["feedback"] = (f"there is no module named {e.name!r}. The only imports that "
                                  f"exist are `import nki`, `import nki.language as nl`, and "
                                  f"`import nki.isa as nisa`. Use exactly those three.")
        else:
            result["failure_kind"] = "environment"
            result["feedback"] = (f"cannot import {e.name!r}, and nki itself is not importable "
                                  f"here. Run this where the Neuron SDK exists.")
        return result
    except Exception as e:
        result["failure_kind"] = "load"
        result["feedback"] = (f"the file imports but {spec['entry']} could not be loaded: "
                              f"{type(e).__name__}: {e}")
        return result

    ok_inputs = ok_numerics = ok_traffic = ok_hazards = True
    worst = 0.0
    saw_floor = False
    for case in spec["shapes"]:
        label = nkibench.label(case, level_n)
        entry = dict(case=label, ok=False, bytes=None, floor=None, waste=None,
                     transfers=None, unmeasured=0, failure=None, checks=None)
        args, _ = nkibench.make_inputs(case, level_n, seed)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            entry["failure"] = f"CANNOT SIMULATE: {e}"
            result["per_case"].append(entry)
            result["failure_kind"] = "environment"
            result["feedback"] = entry["failure"]
            return result
        except Exception as e:
            entry["failure"] = f"raised {type(e).__name__}: {e}"
            result["per_case"].append(entry)
            ok_inputs = ok_numerics = ok_traffic = ok_hazards = False
            continue

        entry["bytes"] = int(counted.get("bytes") or 0)
        entry["floor"] = int(nkibench.minimum_hbm_bytes(args, want) or 0)
        entry["transfers"] = int(counted.get("transfers") or 0)
        entry["unmeasured"] = int(counted.get("unmeasured") or 0)
        if entry["floor"]:
            saw_floor = True
            entry["waste"] = round(entry["bytes"] / entry["floor"], 4)
            worst = max(worst, entry["waste"])

        ok, m, checks = nkibench.accept_case(level_n, got, counted, args, before, want, tol)
        entry["ok"] = ok
        entry["failure"] = m
        entry["checks"] = checks
        ok_inputs &= checks["inputs_ok"]
        ok_numerics &= checks["numerics_ok"]
        ok_traffic &= checks["traffic_ok"]
        ok_hazards &= checks["hazard_ok"]

        result["per_case"].append(entry)
        result["bytes_total"] += entry["bytes"]
        if ok:
            result["passed"] += 1

    result["inputs_ok"] = ok_inputs
    result["numerics_ok"] = ok_numerics
    result["traffic_ok"] = ok_traffic
    result["hazards_ok"] = ok_hazards
    result["worst_waste"] = worst if saw_floor else None
    result["accepted"] = result["passed"] == result["total"] and result["total"] > 0

    if result["accepted"]:
        result["feedback"] = ("Correct on every shape, inputs preserved, traffic gate met. "
                              "Resource status remains an estimate until compiled on device.")
        return result

    failed = [c for c in result["per_case"] if c["failure"]]
    first = failed[0] if failed else None
    if first is not None and first["checks"]:
        result["failure_kind"] = (
            "inputs" if not first["checks"]["inputs_ok"] else
            "numerics" if not first["checks"]["numerics_ok"] else
            "traffic" if not first["checks"]["traffic_ok"] else "hazard")
        result["feedback"] = (f"{result['passed']} of {result['total']} shapes passed. "
                              f"On {first['case']}: {first['failure']}")
    else:
        result["failure_kind"] = "runtime"
        result["feedback"] = (f"{result['passed']} of {result['total']} shapes passed. "
                              + (first["failure"] if first else "no case was evaluated."))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--check", metavar="FILE.py", required=True)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = evaluate_file(a.check, a.level, a.tol, a.seed)
    print(json.dumps(r, indent=2, sort_keys=True))
    raise SystemExit(0 if r["accepted"] else 1)


if __name__ == "__main__":
    main()
