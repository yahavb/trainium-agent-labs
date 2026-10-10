#!/usr/bin/env python3
"""Compare a Trainium prediction with the frozen CPU reference.

The checker gives no performance result until correctness passes. Both files
must be NumPy ``.npz`` archives. They must contain the prediction array and all
coordinate arrays listed in the fixture manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


CONFIG_ERROR = 2
CORRECTNESS_FAILURE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "manifest.json",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Report numerical error without applying tolerances or exposing performance.",
    )
    parser.add_argument("--performance-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required. Run the checker in the Samudra environment."
        ) from error
    return np


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def resolve_reference(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    configured = manifest.get("reference_artifact")
    if not isinstance(configured, str) or not configured:
        raise ValueError("Manifest reference_artifact must be a non-empty path")
    project_root = manifest_path.resolve().parent.parent
    return (project_root / configured).resolve()


def validate_manifest(
    manifest: dict[str, Any], precision: str, diagnostic_only: bool = False
) -> tuple[float | None, float | None, str, list[str]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("Manifest schema_version must be 1")
    status = manifest.get("status")
    if diagnostic_only:
        if status not in ("draft", "frozen"):
            raise ValueError("Manifest status must be 'draft' or 'frozen'")
    elif status != "frozen":
        raise ValueError(
            "CPU reference is not frozen. Set numeric tolerances and hashes, "
            "then set manifest status to 'frozen'."
        )

    tolerance_map = manifest.get("precision_tolerances")
    if not isinstance(tolerance_map, dict):
        if not diagnostic_only:
            raise ValueError("Manifest precision_tolerances must be an object")
        tolerance_map = {}
    tolerance = tolerance_map.get(precision)
    if diagnostic_only:
        atol = rtol = None
    elif not isinstance(tolerance, dict):
        raise ValueError(f"Manifest has no tolerance entry for precision {precision!r}")
    else:
        atol = tolerance.get("atol")
        rtol = tolerance.get("rtol")

    if diagnostic_only:
        pass
    elif not isinstance(atol, (int, float)) or not isinstance(rtol, (int, float)):
        raise ValueError(
            f"Manifest tolerances for precision {precision!r} must be numeric"
        )
    if not diagnostic_only and (
        atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol)
    ):
        raise ValueError(
            f"Manifest tolerances for precision {precision!r} must be finite and non-negative"
        )

    prediction_key = manifest.get("prediction_key")
    coordinate_keys = manifest.get("coordinate_keys")
    if not isinstance(prediction_key, str) or not prediction_key:
        raise ValueError("Manifest prediction_key must be a non-empty string")
    if not isinstance(coordinate_keys, list) or not all(
        isinstance(key, str) and key for key in coordinate_keys
    ):
        raise ValueError("Manifest coordinate_keys must be a list of strings")
    return (
        None if atol is None else float(atol),
        None if rtol is None else float(rtol),
        prediction_key,
        coordinate_keys,
    )


def compare(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    np = load_numpy()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    if args.diagnostic_only and args.performance_json is not None:
        raise ValueError("--performance-json cannot be used with --diagnostic-only")
    atol, rtol, prediction_key, coordinate_keys = validate_manifest(
        manifest, args.precision, args.diagnostic_only
    )
    reference_path = (
        args.reference.resolve()
        if args.reference is not None
        else resolve_reference(manifest_path, manifest)
    )
    candidate_path = args.candidate.resolve()

    if not reference_path.is_file():
        raise ValueError(f"CPU reference not found: {reference_path}")
    if not candidate_path.is_file():
        raise ValueError(f"Candidate not found: {candidate_path}")

    expected_hash = manifest.get("reference_sha256")
    actual_hash = sha256(reference_path)
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError(
            "CPU reference SHA-256 does not match the manifest. "
            f"Expected {expected_hash!r}, found {actual_hash}."
        )

    with np.load(reference_path, allow_pickle=False) as reference_archive:
        reference = {key: reference_archive[key] for key in reference_archive.files}
    with np.load(candidate_path, allow_pickle=False) as candidate_archive:
        candidate = {key: candidate_archive[key] for key in candidate_archive.files}

    required_keys = [prediction_key, *coordinate_keys]
    missing_reference = [key for key in required_keys if key not in reference]
    missing_candidate = [key for key in required_keys if key not in candidate]
    if missing_reference or missing_candidate:
        result = {
            "status": "failed",
            "correctness_score": None if args.diagnostic_only else 0.0,
            "reason": "missing required arrays",
            "missing_reference": missing_reference,
            "missing_candidate": missing_candidate,
            "performance": None,
        }
        return CORRECTNESS_FAILURE, result

    coordinate_failures = []
    for key in coordinate_keys:
        if not np.array_equal(reference[key], candidate[key]):
            coordinate_failures.append(key)

    expected = reference[prediction_key]
    actual = candidate[prediction_key]
    shape_matches = tuple(expected.shape) == tuple(actual.shape)
    finite = bool(np.isfinite(actual).all()) if shape_matches else False

    result: dict[str, Any] = {
        "status": "failed",
        "reference": str(reference_path),
        "reference_sha256": actual_hash,
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "prediction_key": prediction_key,
        "reference_shape": list(expected.shape),
        "candidate_shape": list(actual.shape),
        "shape_matches": shape_matches,
        "coordinates_match": not coordinate_failures,
        "coordinate_failures": coordinate_failures,
        "candidate_finite": finite,
        "precision": args.precision,
        "correctness_score": None if args.diagnostic_only else 0.0,
        "performance": None,
    }

    if not shape_matches:
        result["reason"] = "prediction shape mismatch"
        return CORRECTNESS_FAILURE, result
    if coordinate_failures:
        result["reason"] = "coordinate mismatch"
        return CORRECTNESS_FAILURE, result
    if not finite:
        result["reason"] = "candidate contains NaN or infinite values"
        return CORRECTNESS_FAILURE, result

    expected64 = expected.astype(np.float64, copy=False)
    actual64 = actual.astype(np.float64, copy=False)
    difference = actual64 - expected64
    absolute_error = np.abs(difference)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(expected64))))
    normalized_rmse = rmse / reference_rms if reference_rms else rmse
    if args.diagnostic_only:
        result.update(
            {
                "status": "diagnostic",
                "reason": "diagnostic-only; no tolerance verdict was applied",
                "rmse": rmse,
                "normalized_rmse": normalized_rmse,
                "max_abs_error": float(np.max(absolute_error)),
                "values_compared": int(expected.size),
            }
        )
        return 0, result

    result["atol"] = atol
    result["rtol"] = rtol
    allowed_error = atol + rtol * np.abs(expected64)
    within = absolute_error <= allowed_error
    fraction_within = float(np.mean(within))
    passed = bool(np.all(within))

    result.update(
        {
            "status": "passed" if passed else "failed",
            "reason": "all values are within tolerance"
            if passed
            else "one or more values exceed tolerance",
            "correctness_score": fraction_within,
            "rmse": rmse,
            "normalized_rmse": normalized_rmse,
            "max_abs_error": float(np.max(absolute_error)),
            "values_compared": int(expected.size),
            "values_within_tolerance": int(np.count_nonzero(within)),
        }
    )

    if passed and args.performance_json is not None:
        performance = load_json(args.performance_json.resolve())
        if performance.get("status") != "passed":
            raise ValueError("Performance JSON status must be 'passed'")
        result["performance"] = performance

    return (0 if passed else CORRECTNESS_FAILURE), result


def emit(result: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        exit_code, result = compare(args)
    except (OSError, RuntimeError, ValueError) as error:
        result = {
            "status": "configuration_error",
            "correctness_score": 0.0,
            "reason": str(error),
            "performance": None,
        }
        emit(result, args.json_out)
        return CONFIG_ERROR
    emit(result, args.json_out)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
