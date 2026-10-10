"""Dataset replay, leakage checks and artifact tamper detection."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from build_engine_sets import build, verify_set


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "Install requirements-engine.txt for engine tests")
class EngineSetTests(unittest.TestCase):
    def test_replay_schema_disjointness_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            out = Path(temporary) / "suite"
            build(out)
            verify_set(out)
            public = json.loads((out / "public/manifest.json").read_text())
            private = json.loads((out / "evaluator-only/private-manifest.json").read_text())
            self.assertTrue({r["problem_sha256"] for r in public["fixtures"]}.isdisjoint(
                {r["problem_sha256"] for r in private["fixtures"]}))
            self.assertFalse(list((out / "public").glob("*-reference.npz")))
            for row in public["fixtures"]:
                with np.load(out / "public" / row["input_file"], allow_pickle=False) as inputs:
                    self.assertEqual(set(inputs.files), {"A", "b"})
            contract = out / "public/contract.json"
            contract.write_text(contract.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "Commitment mismatch"):
                verify_set(out)
            with self.assertRaises(FileExistsError):
                build(out)


if __name__ == "__main__":
    unittest.main()
