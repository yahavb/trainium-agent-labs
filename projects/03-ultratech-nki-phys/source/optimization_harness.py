"""Explain applicability and select measured winners within reviewed kernel forms."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from plane_loop import canonical, reviewed_forms


def plan(source):
    forms = reviewed_forms()
    matched = next((name for name, code in forms.items()
                    if canonical(source) == canonical(code)), None)
    # Exact reviewed forms establish the buffer layout and operation semantics.
    # A matching instruction name alone is not sufficient for safe rewriting.
    transitions = {
        "baseline": ["copyfree", "scale-fused", "hoist-bias", "engine-vector", "interleave", "tile32"],
        "copyfree": ["scale-fused"],
        "scale-fused": [],
    }
    supported = matched in transitions
    copy_available = matched == "baseline"
    scaling_available = matched in ("baseline", "copyfree")
    return {
        "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "input_form": matched,
        "supported": supported,
        "techniques": [
            {"name": "copy-removal", "applicable": supported and copy_available,
             "reason": ("Reviewed PSUM product is copied into gradient before the sum; "
                        "the reviewed sum can consume product directly." if copy_available else
                        "No removable product-to-gradient copy in this supported form." if supported else
                        "Input is outside the supported reviewed forms; human review required.")},
            {"name": "scaling-update-fusion", "applicable": supported and scaling_available,
             "reason": ("Separate multiply-by-rate and subtraction can use the reviewed "
                        "negative-rate multiply/add form." if scaling_available else
                        "Scaling/update is already fused in this supported form." if supported else
                        "Input is outside the supported reviewed forms; human review required.")},
        ],
        "candidate_forms": transitions.get(matched, []),
        "policy": "Benchmark all applicable reviewed candidates against this input; keep input if none wins.",
        "scope": "Finite human-reviewed catalog, not arbitrary NKI analysis or a general sandbox.",
        "composition": "scale-fused includes copy removal where a copy exists; it is not an isolated scaling ablation.",
    }


def select(report, measurements):
    """Consume trusted checker/timing summaries, not candidate self-reported scores."""
    winner = "original"
    best = 1.0
    decisions = []
    for row in measurements:
        ratio = row.get("throughput_ratio")
        valid_ratio = (isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
                       and math.isfinite(ratio) and ratio > 0)
        reason = None
        if not report["supported"] or row.get("form") not in report["candidate_forms"]:
            reason = "Not an applicable reviewed candidate"
        elif row.get("input_sha256") != report["input_sha256"]:
            reason = "Measured against a different input kernel"
        elif row.get("physics_score") != 1.0:
            reason = "Did not pass every correctness gate"
        elif row.get("stable_baseline") is not True or not valid_ratio:
            reason = "Missing or invalid stable paired benchmark"
        elif not row.get("evidence_path"):
            reason = "Missing trusted checker/benchmark evidence location"
        if reason is None and ratio > best:
            best, winner = ratio, row["form"]
        decisions.append({"form": row.get("form"), "eligible": reason is None,
                          "reason": reason or "Correctness passed; valid measured comparison"})
    return {"winner": winner, "throughput_ratio": best, "decisions": decisions,
            "note": "Best observed eligible result only; not statistical significance or global optimality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = plan(args.kernel.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "applicability.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in report["candidate_forms"]:
        (args.out / (name + ".py")).write_text(reviewed_forms()[name])
    print(f"Input: {report['input_form']}; applicable candidates: {report['candidate_forms']}")
    print("Plan only: correctness and device throughput are not measured by this command.")


if __name__ == "__main__":
    main()
