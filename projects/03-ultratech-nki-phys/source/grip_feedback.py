"""Deterministic public-case guidance; never substitutes for physics scoring."""


def explain_check(check):
    if check.get("passed") is True:
        return "All physics gates passed for this output. Preserve accuracy when changing the solver."
    force = check.get("force_check", {})
    numerical = force.get("numerical", {})
    if "reason" in numerical:
        return ("Invalid output: " + numerical["reason"].replace("impulses", "forces") +
                ". Return the required force-vector shape with finite FP32 values.")
    messages = []
    if numerical.get("feasible") is False:
        messages.append("Negative pyramid-edge forces violate feasibility. Enforce nonnegativity; do not clamp physical tangential components, which may be signed.")
    metrics = (
        (numerical, "residual", 1e-4, "Projected residual",
         "The output has not met the optimality tolerance. Investigate update correctness, step size, convergence or FP32 stagnation; more iterations alone may not fix rounding."),
        (numerical, "objective_error", 1e-5, "Objective discrepancy",
         "Check that the solver minimizes the specified regularized QP, with the correct matrix orientation and bias sign."),
        (force, "force_error", 1e-4, "Edge-force error",
         "Edge values differ too much from the trusted solution. Residual-only stopping may be insufficient; investigate conditioning or a more accurate solving method."),
        (force, "acceleration_error", 1e-4, "Acceleration error",
         "The force errors produce excessive generalized-acceleration error. Reduce force error rather than accepting a small objective error alone."),
        (check, "wrench_error", 1e-4, "Contact-wrench error",
         "Reconstructed normal/friction forces are inaccurate. Return pyramid-edge coefficients, not direct normal/tangential components, and solve their coupled problem."),
        (check, "cone_violation", 1e-6, "Friction-cone violation",
         "The reconstructed friction exceeds the pyramidal bound. Preserve nonnegative edge coefficients and the specified friction model."),
    )
    for report, key, limit, label, guidance in metrics:
        value = report.get(key)
        if value is not None and value > limit:
            messages.append(f"{label} {value:.6g} exceeds {limit:.6g}. {guidance}")
    return " ".join(messages) or "A physics gate failed. Inspect the complete numerical report; no specific cause is established."


def next_prompt(scored):
    failures, cases, invalid = [], {}, set()
    gates = (
        ("residual", 1e-4, "projected residual"),
        ("objective_error", 1e-5, "objective discrepancy"),
        ("force_error", 1e-4, "edge-force error"),
        ("acceleration_error", 1e-4, "acceleration error"),
        ("wrench_error", 1e-4, "contact-wrench error"),
        ("cone_violation", 1e-6, "friction-cone violation"),
    )
    observed = {key: {} for key, _, _ in gates}
    infeasible = set()
    for attempt in scored:
        if attempt.get("score") == 1:
            continue
        if attempt.get("error"):
            failures.append("Execution failure: " + attempt["error"][-500:])
        if attempt.get("harness_failure"):
            failures.append(attempt["harness_failure"])
        for case in attempt.get("cases", []):
            case_id = case["case_id"]
            cases[case_id] = cases.get(case_id, True) and case["passed"]
            for phase in ("before", "after"):
                check = case[phase]
                force = check.get("force_check", {})
                numerical = force.get("numerical", {})
                if "reason" in numerical:
                    invalid.add(case_id)
                if numerical.get("feasible") is False:
                    infeasible.add(case_id)
                metrics = dict(numerical, **{key: value for key, value in force.items() if key != "numerical"})
                metrics.update({key: check[key] for key in ("wrench_error", "cone_violation") if key in check})
                for key, _, _ in gates:
                    if key in metrics:
                        observed[key][case_id] = max(observed[key].get(case_id, 0), metrics[key])
    failures = list(dict.fromkeys(failures))
    failed_ids = [key for key, passed in cases.items() if not passed]
    if failures or failed_ids or any(row.get("score") != 1 for row in scored):
        lines = ["Revise the NKI solver using this public checker feedback. Correctness comes before throughput.",
                 "Do not relax gates, hardcode case answers or access oracle outputs. Suggested causes are hypotheses.",
                 f"Public cases observed: {len(cases)}; failed in at least one phase/repeat: {len(failed_ids)}.",
                 "Each metric below is the worst value per case across before/after execution and repeats."]
        if invalid:
            lines.append(f"Invalid shape/nonfinite outputs: {len(invalid)} cases. Return finite FP32 forces with the specified shape.")
        if infeasible:
            lines.append(f"Negative pyramid-edge forces: {len(infeasible)} cases. Preserve nonnegative edges; physical tangential forces may be signed.")
        for key, limit, label in gates:
            values = observed[key]
            if values:
                failed = sum(value > limit for value in values.values())
                lines.append(f"{label}: range {min(values.values()):.6g}..{max(values.values()):.6g}; limit {limit:.6g}; failures {failed}/{len(values)}.")
        ranking = sorted(failed_ids, key=lambda cid: max(
            (observed[key].get(cid, 0) / limit for key, limit, _ in gates), default=0), reverse=True)
        for case_id in ranking[:3]:
            details = "; ".join(f"{label}={observed[key][case_id]:.6g}" for key, limit, label in gates
                                if observed[key].get(case_id, 0) > limit)
            lines.append(f"Representative {case_id}: {details or 'invalid output or other failed gate'}.")
        if observed["residual"] and any(value > 1e-4 for value in observed["residual"].values()):
            lines.append("Investigate convergence, step size and conditioning before low-level speed tuning. More iterations may help, but FP32 stagnation can remain; consider acceleration, preconditioning or another mathematically valid solver.")
        if any(value > 1e-4 for value in observed["force_error"].values()):
            lines.append("A small objective/residual alone is insufficient: edge forces, reconstructed contact forces and generalized accelerations must all meet their own gates.")
        lines.extend(failures[:3])
        if len(failures) > 3:
            lines.append(f"{len(failures) - 3} additional unique execution/harness diagnostics remain in the complete scored log.")
        return "\n".join(lines)
    return ("All public physics and harness gates passed. Preserve these gates and optimize "
            "accepted problems per second on Trainium. Simulator correctness is not hardware "
            "performance evidence; compare verified device throughput across the full same workload.")
