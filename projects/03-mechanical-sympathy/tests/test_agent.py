from __future__ import annotations

import csv
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
AGENT = PROJECT_ROOT / "agent.py"


class AgentAttemptTest(unittest.TestCase):
    def test_records_timing_only_after_correctness_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "reference.npz"
            manifest = root / "manifest.json"
            attempts = root / "attempts.csv"
            good_adapter = root / "good_adapter.py"
            bad_adapter = root / "bad_adapter.py"
            prediction = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
            np.savez_compressed(
                fixture,
                prognostic=prediction,
                boundary=np.ones_like(prediction),
                prediction=prediction,
                time=np.asarray(["2014-10-20"], dtype="U"),
                variables=np.asarray(["t+1:thetao_0"], dtype="U"),
            )
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "frozen",
                        "reference_artifact": str(fixture),
                        "reference_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        "prediction_key": "prediction",
                        "coordinate_keys": ["time", "variables"],
                        "precision_tolerances": {
                            "fp32": {"atol": 0.0, "rtol": 0.0},
                            "bf16-autocast": {"atol": 0.0, "rtol": 0.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapter_template = """
                class Prepared:
                    def __init__(self, inputs):
                        self.inputs = inputs

                    def run(self):
                        return self.inputs["prognostic"]{offset}

                    def synchronize(self):
                        return None

                def prepare(inputs, context):
                    return Prepared(inputs)
            """
            good_adapter.write_text(
                textwrap.dedent(adapter_template.format(offset="")),
                encoding="utf-8",
            )
            bad_adapter.write_text(
                textwrap.dedent(adapter_template.format(offset=" + 1.0")),
                encoding="utf-8",
            )

            rows = []
            for experiment, adapter in (
                ("correct", good_adapter),
                ("incorrect", bad_adapter),
            ):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(AGENT),
                        "--experiment",
                        experiment,
                        "--adapter",
                        str(adapter),
                        "--fixture",
                        str(fixture),
                        "--manifest",
                        str(manifest),
                        "--attempts",
                        str(attempts),
                        "--precision",
                        "fp32",
                        "--warmup",
                        "0",
                        "--repeats",
                        "1",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                rows.append(completed.returncode)

            self.assertEqual(rows, [0, 1])
            with attempts.open(newline="", encoding="utf-8") as stream:
                attempts_by_experiment = {
                    row["experiment"]: row for row in csv.DictReader(stream)
                }
            self.assertEqual(attempts_by_experiment["correct"]["status"], "passed")
            self.assertNotEqual(attempts_by_experiment["correct"]["median_ms"], "")
            self.assertEqual(attempts_by_experiment["incorrect"]["status"], "failed")
            self.assertEqual(attempts_by_experiment["incorrect"]["median_ms"], "")


if __name__ == "__main__":
    unittest.main()
