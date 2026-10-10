"""Startup accounting: non-baseline continuation must get an independent referee check."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location("pilot", Path(__file__).resolve().parents[1] / "run_improvement_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class StartupTests(unittest.TestCase):
    def test_baseline_reuses_acceptance_without_worker(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kernel.py"
            path.write_bytes(b"x=1\r\n")
            worker, acceptance = Mock(), {"verdict": "no_gain"}
            record, reused = pilot.evaluate_startup(worker, path, "x=1\n", acceptance)
            self.assertTrue(reused)
            self.assertEqual(record, acceptance)
            self.assertIsNot(record, acceptance)
            worker.check.assert_not_called()

    def test_continuation_checked_never_imported(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kernel.py"
            path.write_text("raise RuntimeError('must not execute candidate')")
            record = {"verdict": "faster", "speedup": 1.517}
            worker = Mock()
            worker.check.return_value = record
            result, reused = pilot.evaluate_startup(worker, path, "baseline", {"verdict": "no_gain"})
            self.assertIs(result, record)
            self.assertFalse(reused)
            worker.check.assert_called_once_with(path)

    def test_infrastructure_failure_not_replaced_by_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kernel.py"
            path.write_text("candidate")
            worker = Mock()
            worker.check.return_value = None
            self.assertEqual(pilot.evaluate_startup(worker, path, "baseline", {"verdict": "no_gain"}), (None, False))


if __name__ == "__main__":
    unittest.main()
