from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "runners" / "trainium_runner.py"
CHECKER = PROJECT_ROOT / "checker.py"


class TrainiumRunnerContractTest(unittest.TestCase):
    def test_adapter_output_passes_checker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "reference.npz"
            candidate = root / "candidate.npz"
            metrics = root / "metrics.json"
            result = root / "result.json"
            manifest = root / "manifest.json"
            adapter = root / "adapter.py"

            prediction = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
            np.savez_compressed(
                fixture,
                prognostic=prediction.copy(),
                boundary=np.ones((1, 1, 2, 4), dtype=np.float32),
                prediction=prediction,
                time=np.asarray(["2014-10-20"], dtype="U"),
                variables=np.asarray(["t+1:thetao_0"], dtype="U"),
            )
            reference_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "frozen",
                        "reference_artifact": str(fixture),
                        "reference_sha256": reference_hash,
                        "prediction_key": "prediction",
                        "coordinate_keys": ["time", "variables"],
                        "tolerance": {"atol": 0.0, "rtol": 0.0},
                    }
                ),
                encoding="utf-8",
            )
            adapter.write_text(
                textwrap.dedent(
                    """
                    class Prepared:
                        def __init__(self, inputs):
                            self.inputs = inputs

                        def run(self):
                            return self.inputs["prognostic"]

                        def synchronize(self):
                            return None

                    def prepare(inputs, context):
                        assert context["workload"] == "single-step"
                        return Prepared(inputs)
                    """
                ),
                encoding="utf-8",
            )

            runner = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--adapter",
                    str(adapter),
                    "--fixture",
                    str(fixture),
                    "--manifest",
                    str(manifest),
                    "--candidate-output",
                    str(candidate),
                    "--metrics-json",
                    str(metrics),
                    "--precision",
                    "float32",
                    "--warmup",
                    "1",
                    "--repeats",
                    "3",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(runner.returncode, 0, runner.stderr)

            checked = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--manifest",
                    str(manifest),
                    "--candidate",
                    str(candidate),
                    "--performance-json",
                    str(metrics),
                    "--json-out",
                    str(result),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["correctness_score"], 1.0)
            self.assertEqual(payload["performance"]["timed_runs"], 3)


if __name__ == "__main__":
    unittest.main()
