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
        self.write_manifest(status="frozen", atol=1e-5, rtol=1e-5)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_npz(self, path: Path, prediction: np.ndarray) -> None:
        np.savez_compressed(
            path,
            prediction=prediction,
            time=self.time,
            variables=self.variables,
        )

    def write_manifest(self, status: str, atol: float | None, rtol: float | None) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": status,
                    "reference_artifact": str(self.reference),
                    "reference_sha256": file_sha256(self.reference),
                    "prediction_key": "prediction",
                    "coordinate_keys": ["time", "variables"],
                    "tolerance": {"atol": atol, "rtol": rtol},
                }
            ),
            encoding="utf-8",
        )

    def check(self) -> tuple[int, dict]:
        completed = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--manifest",
                str(self.manifest),
                "--candidate",
                str(self.candidate),
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
        self.write_manifest(status="draft", atol=None, rtol=None)
        code, result = self.check()
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "configuration_error")


if __name__ == "__main__":
    unittest.main()
