"""Real local classifier + existing prompt builder + scripted CPU repair fixtures.

Not a generated NKI-kernel, complete-agent, accuracy, or speedup benchmark.
No recorded/model-generated code is executed; cloud resources are not accessed.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import httpx
import numpy as np


def require(condition, message):
    # Unlike assert, validation remains enabled under python -O.
    if not condition:
        raise AssertionError(message)


def checked(candidate, inputs, reference):
    want = reference(*inputs)
    try:
        got = np.asarray(candidate(*inputs))
    except (AttributeError, ValueError, TypeError) as error:
        return False, f"{type(error).__name__}: {error}"
    if got.shape != want.shape:
        return False, (f"Output shape {got.shape}; expected shape {want.shape}. "
                       "Correct transpose orientation.")
    if not np.isfinite(got).all() or not np.allclose(got, want, rtol=1e-5, atol=1e-6):
        return False, "Numerical checker rejects the output: values do not match the reference."
    return True, "Numerical checker passed."


def bad_api(a):
    return np.row_softmax(a)


def repaired_softmax(a):
    values = np.exp(a - np.max(a, axis=1, keepdims=True))
    return values / np.sum(values, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    project = args.repo.resolve() / "projects/20-kernel-agent"
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "local_decider"))
    import agent
    import decider
    import nkibench

    rng = np.random.default_rng(96)
    q, k = [rng.normal(size=(12, 8)).astype(np.float32) for _ in range(2)]
    x = rng.normal(size=(6, 8)).astype(np.float32)
    cases = [
        ("attention_score_orientation", 11, (q, k), nkibench.ref_attention_scores,
         lambda a, b: (a.T @ b).astype(a.dtype),
         lambda a, b: ((a @ b.T) / (a.shape[1] ** 0.5)).astype(a.dtype),
         "repair_layout", "def candidate(q, k): return q.T @ k"),
        ("transpose_output_shape", 9, (x,), nkibench.ref_transpose_tile,
         lambda a: a.copy(), lambda a: np.ascontiguousarray(a.T),
         "repair_layout", "def candidate(x): return x.copy()"),
        ("softmax_invalid_api", 10, (x,), nkibench.ref_row_softmax,
         bad_api, repaired_softmax, "repair_api",
         "def candidate(x): return np.row_softmax(x)"),
    ]
    results = []
    started = time.perf_counter()
    for name, level, inputs, reference, broken, repaired, expected, code in cases:
        passed_before, feedback = checked(broken, inputs, reference)
        require(not passed_before, f"{name}: fixture must fail before routing")
        state = decider.bounded_state({"level": level, "code": code, "feedback": feedback})
        decision = decider.decide(state)
        require(decision["strategy"] in decider.OPTIONS, "Unknown advisory strategy")
        base_prompt = agent.repair_prompt(level, code, feedback)
        prompt = base_prompt + "\nOptional advisory focus: " + decider.OPTIONS[decision["strategy"]]
        require(base_prompt in prompt and code in prompt and feedback in prompt,
                "Existing repair prompt or checker feedback was lost")
        # Trusted scripted handler selected by strategy; no model-output execution.
        candidate = repaired if decision["strategy"] == expected else broken
        passed, _ = checked(candidate, inputs, reference)
        require(passed, f"{name}: route {decision['strategy']} did not repair the fixture")
        require(not checked(broken, inputs, reference)[0], "Checker accepted broken fixture")
        results.append({"case": name, "initial_check_passed": False,
                        "strategy": decision["strategy"], "source": decision["source"],
                        "classifier_seconds": decision["seconds"], "repair_prompt_preserved": True,
                        "scripted_cpu_repair_passed": passed, "bad_result_still_rejected": True})

    require(any(row["source"] == "decider" for row in results),
            "Real classifier was never used: check the local service")
    # Outage simulated at client boundary; the running service is not stopped.
    with patch.object(decider, "client", side_effect=httpx.ConnectError("deliberate test outage")):
        fallback = decider.decide(decider.bounded_state({
            "level": 11, "feedback": "Contraction axis mismatch; correct transpose orientation."}))
    require(fallback["source"] == "fallback" and fallback["strategy"] == "repair_layout",
            "Outage did not retain deterministic fallback")
    require(checked(cases[0][5], cases[0][2], cases[0][3])[0], "Fallback repair failed")
    # An inappropriate strategy has no applicable handler; checker still rejects it.
    require(not checked(cases[0][4], cases[0][2], cases[0][3])[0],
            "Wrong strategy bypassed the numerical check")
    print(json.dumps({
        "fixture_cases": len(results), "fixture_passes": len(results),
        "real_model_accepted_decisions": sum(r["source"] == "decider" for r in results),
        "simulated_outage_fallback_passed": True, "wrong_strategy_cannot_bypass_checker": True,
        "seconds": round(time.perf_counter() - started, 4),
        "repairs_are_scripted_numpy_fixtures": True, "main_agent_loop_modified": False,
        "generated_nki_kernel_tested": False, "trainium_tested": False,
        "speedup_measured": False, "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()
