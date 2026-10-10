"""Frozen candidate gates for the engine-backed force task, not impulses."""

import numpy as np
from contact import diagnostics

CONTRACT = dict(version=1, output="normal_contact_forces", precision="FP32 candidate, FP64 checks",
                nonnegative_tolerance=1e-6, projected_residual_limit=1e-4,
                objective_error_limit=1e-5, force_error_limit=1e-4,
                acceleration_error_limit=1e-4, padding_tolerance=1e-6,
                all_cases_required=True, references_visible_to_candidate=False)


def check_forces(fixture, candidate, reference):
    candidate = np.asarray(candidate)
    base = diagnostics(fixture["A"], fixture["b"], candidate, reference)
    if candidate.shape != fixture["b"].shape or not np.isfinite(candidate).all():
        return dict(passed=False, numerical=base)
    acceleration = fixture["qacc_smooth"] + fixture["response"] @ candidate.astype(np.float64)
    expected = fixture["qacc_smooth"] + fixture["response"] @ reference
    force_error = float(np.max(np.abs(candidate - reference)) / max(1, np.max(np.abs(reference))))
    acceleration_error = float(np.max(np.abs(acceleration - expected)) / max(1, np.max(np.abs(expected))))
    return dict(passed=bool(base["passed"] and force_error <= CONTRACT["force_error_limit"] and
                           acceleration_error <= CONTRACT["acceleration_error_limit"]),
                numerical=base, force_error=force_error, acceleration_error=acceleration_error)


def public_gate(report, manifest, candidate_hash):
    """Check trusted controller reports before permitting private evaluation."""
    expected = {r["case_id"] for r in manifest["fixtures"]}
    rows = report.get("cases", [])
    ids = [r.get("case_id") for r in rows]
    return bool(expected and report.get("candidate_sha256") == candidate_hash and
                report.get("suite_id") == manifest["suite_id"] and
                report.get("contract_sha256") == manifest["contract_sha256"] and
                len(ids) == len(expected) and set(ids) == expected and
                all(r.get("passed") is True for r in rows))
