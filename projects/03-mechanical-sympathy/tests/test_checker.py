from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKER = PROJECT_ROOT / "checker.py"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference.npz"
        self.candidate = self.root / "candidate.npz"
        self.manifest = self.root / "manifest.json"
        self.result = self.root / "result.json"
        self.prediction = np.asarray([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        self.time = np.asarray(["2014-10-20"], dtype="U")
        self.variables = np.asarray(["t+1:thetao_0"], dtype="U")
        self.write_npz(self.reference, self.prediction)
        self.write_manifest(
            status="frozen",
            tolerances={
                "fp32": {"atol": 1e-5, "rtol": 1e-5},
                "bf16-autocast": {"atol": 0.0, "rtol": 0.02},
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_npz(self, path: Path, prediction: np.ndarray) -> None:
        np.savez_compressed(
            path,
            prediction=prediction,
            time=self.time,
            variables=self.variables,
        )

    def write_manifest(
        self, status: str, tolerances: dict[str, dict[str, float | None]]
    ) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": status,
                    "reference_artifact": str(self.reference),
                    "reference_sha256": file_sha256(self.reference),
                    "prediction_key": "prediction",
                    "coordinate_keys": ["time", "variables"],
                    "precision_tolerances": tolerances,
                }
            ),
            encoding="utf-8",
        )

    def check(
        self, precision: str = "fp32", *extra_args: str
    ) -> tuple[int, dict]:
        completed = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--manifest",
                str(self.manifest),
                "--candidate",
                str(self.candidate),
                "--precision",
                precision,
                *extra_args,
                "--json-out",
                str(self.result),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode, json.loads(self.result.read_text(encoding="utf-8"))

    def test_accepts_values_within_tolerance(self) -> None:
        self.write_npz(self.candidate, self.prediction + np.float32(1e-6))
        code, result = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["correctness_score"], 1.0)

    def test_rejects_shape_mismatch(self) -> None:
        self.write_npz(self.candidate, self.prediction[..., :1])
        code, result = self.check()
        self.assertEqual(code, 1)
        self.assertEqual(result["reason"], "prediction shape mismatch")
        self.assertIsNone(result["performance"])

    def test_rejects_non_finite_values(self) -> None:
        candidate = self.prediction.copy()
        candidate.flat[0] = np.nan
        self.write_npz(self.candidate, candidate)
        code, result = self.check()
        self.assertEqual(code, 1)
        self.assertIn("NaN", result["reason"])

    def test_rejects_numeric_error(self) -> None:
        self.write_npz(self.candidate, self.prediction + np.float32(0.1))
        code, result = self.check()
        self.assertEqual(code, 1)
        self.assertLess(result["correctness_score"], 1.0)
        self.assertIsNone(result["performance"])

    def test_refuses_draft_reference(self) -> None:
        self.write_npz(self.candidate, self.prediction)
        self.write_manifest(
            status="draft",
            tolerances={
                "fp32": {"atol": None, "rtol": None},
                "bf16-autocast": {"atol": None, "rtol": None},
            },
        )
        code, result = self.check()
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "configuration_error")

    def test_selects_precision_specific_tolerance(self) -> None:
        self.write_npz(self.candidate, self.prediction + np.float32(0.01))
        bf16_code, bf16 = self.check("bf16-autocast")
        fp32_code, fp32 = self.check("fp32")
        self.assertEqual(bf16_code, 0)
        self.assertEqual(bf16["status"], "passed")
        self.assertEqual(bf16["atol"], 0.0)
        self.assertEqual(bf16["rtol"], 0.02)
        self.assertEqual(fp32_code, 1)
        self.assertEqual(fp32["status"], "failed")

    def test_rejects_missing_precision_tolerance(self) -> None:
        self.write_npz(self.candidate, self.prediction)
        self.write_manifest(
            status="frozen", tolerances={"fp32": {"atol": 0.0, "rtol": 0.0}}
        )
        code, result = self.check("bf16-autocast")
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "configuration_error")
        self.assertIn("no tolerance entry", result["reason"])

    def test_diagnostic_reports_errors_for_draft_without_pass_or_performance(self) -> None:
        self.write_npz(self.candidate, self.prediction + np.float32(0.01))
        self.write_manifest(
            status="draft",
            tolerances={
                "fp32": {"atol": None, "rtol": None},
                "bf16-autocast": {"atol": None, "rtol": None},
            },
        )
        code, result = self.check("bf16-autocast", "--diagnostic-only")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "diagnostic")
        self.assertIsNone(result["correctness_score"])
        self.assertIsNone(result["performance"])
        self.assertGreater(result["normalized_rmse"], 0.0)
        self.assertGreater(result["max_abs_error"], 0.0)

    def test_rejects_coordinate_mismatch(self) -> None:
        np.savez_compressed(
            self.candidate,
            prediction=self.prediction,
            time=np.asarray(["2014-10-21"], dtype="U"),
            variables=self.variables,
        )
        code, result = self.check()
        self.assertEqual(code, 1)
        self.assertEqual(result["reason"], "coordinate mismatch")
        self.assertEqual(result["coordinate_failures"], ["time"])


if __name__ == "__main__":
    unittest.main()
