import unittest
from pathlib import Path

from grip_loop import SEED_SHA, restricted_revision
import hashlib
import argparse
import json
import tempfile
import subprocess
from unittest.mock import patch
from grip_loop import Controller


class LoopGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = (Path(__file__).parent / "data/qwen-grip-6074b673bb6348ce932b631a7a079f4c/candidate.py").read_text()

    def test_seed_pinned(self):
        self.assertEqual(hashlib.sha256(self.seed.encode()).hexdigest(), SEED_SHA)

    def test_beta_and_comments_only(self):
        self.assertEqual(restricted_revision(self.seed.replace("beta = 0.9", "beta = 0.95") + "\n# note\n", self.seed), .95)

    def test_reject_host_code(self):
        with self.assertRaises(ValueError):
            restricted_revision("import os\n" + self.seed, self.seed)

    def test_reject_algorithm_change(self):
        with self.assertRaises(ValueError):
            restricted_revision(self.seed.replace("moving=y", "moving=x"), self.seed)

    def test_reject_invalid_beta(self):
        for value in ("1.0", "-0.1", "True", "float('nan')", "__import__('os').system('id')"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                restricted_revision(self.seed.replace("beta = 0.9", "beta = " + value), self.seed)

    def test_controller_grades_revises_and_stops(self):
        seed_source = self.seed

        class FakeController(Controller):
            def upload(self, local, remote, log):
                pass

            def remote_command(self, argv, log):
                pass

            def command(self, argv, log):
                if "grade" in argv:
                    result = Path(argv[argv.index("--results") + 1])
                    grading = result / "grading-test"
                    grading.mkdir()
                    passed = 8 if "attempt-000" in str(result) else 16
                    (grading / "results.json").write_text(json.dumps(dict(
                        passing_case_evaluations=passed, expected_case_evaluations=16,
                        all_cases_pass=passed == 16)))
                    (grading / "next-prompt.txt").write_text("Measured failures: improve convergence.")
                    if passed != 16:
                        raise subprocess.CalledProcessError(1, argv)

            def download(self, remote, local, log):
                local.mkdir()
                if "generation-" in remote:
                    (local / "candidate.py").write_text(seed_source.replace("beta = 0.9", "beta = 0.95"))

        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / "seed.py"
            seed.write_text(seed_source)
            args = argparse.Namespace(out=Path(directory) / "loop", seat="seat-test", minutes=1,
                                      seed=seed, suite=Path(directory), steps=256, attempts=5,
                                      base_url="http://localhost:8000/v1")
            controller = FakeController(args)
            controller.run()
            self.assertEqual([r["passed"] for r in controller.rows], [8, 16])
            self.assertEqual(len((args.out / "attempts.jsonl").read_text().splitlines()), 2)
            self.assertEqual(json.loads((args.out / "best.json").read_text())["beta"], .95)

    def test_grader_exit_one_without_new_report_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            old = results / "grading-old"
            old.mkdir()
            (old / "results.json").write_text("{}")
            args = argparse.Namespace(out=results, minutes=1, suite=results)
            controller = Controller(args)
            with patch.object(controller, "command", side_effect=subprocess.CalledProcessError(1, ["grade"])):
                with self.assertRaisesRegex(ValueError, "fresh trusted report"):
                    controller.evaluate(results, results / "log")


if __name__ == "__main__":
    unittest.main()
