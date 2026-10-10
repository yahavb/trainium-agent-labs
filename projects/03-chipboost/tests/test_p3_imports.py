"""CPU regression: experimental_agent.py borrows diagnose and stage12 from P3's tree.

The referee-timing branch does not carry P3's modules, so the pilot must find them through
CHIPBOOST_P3 (P3's projects/03-chipboost) and fail with a message naming it otherwise.
Run: python projects/03-chipboost/tests/test_p3_imports.py
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


AGENTS = ("experimental_agent",)


def import_agent(env_p3, module):
    env = dict(os.environ)
    env.pop("CHIPBOOST_P3", None)
    if env_p3 is not None:
        env["CHIPBOOST_P3"] = env_p3
    return subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT, env=env,
                          capture_output=True, text=True)


@unittest.skipIf((ROOT / "diagnose.py").exists(), "P3's modules are in this tree")
class P3ImportTests(unittest.TestCase):
    def test_imports_from_p3_tree(self):
        with tempfile.TemporaryDirectory() as p3:
            os.mkdir(os.path.join(p3, "redteam"))
            Path(p3, "diagnose.py").write_text("def diagnose(*a, **k):\n    return None\n")
            Path(p3, "redteam", "stage12.py").write_text("LEVEL = 4\nSHAPES = []\n")
            for module in AGENTS:
                done = import_agent(p3, module)
                self.assertEqual(done.returncode, 0, module + ": " + done.stderr)

    def test_missing_p3_names_the_variable(self):
        for module in AGENTS:
            done = import_agent(None, module)
            self.assertNotEqual(done.returncode, 0, module)
            self.assertIn("CHIPBOOST_P3", done.stderr, module)


if __name__ == "__main__":
    unittest.main()
