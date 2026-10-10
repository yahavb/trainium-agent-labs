import json
from pathlib import Path
import tempfile
import unittest

from grip_feedback import explain_check, next_prompt
from grip_throughput import aggregate_timing, execute, grade


SUITE = Path(__file__).with_name("data") / "gripping-development-v1"


class FeedbackTests(unittest.TestCase):
    def test_compact_feedback_merges_repeats_and_preserves_worst_phase(self):
        before = dict(passed=False, force_check=dict(numerical=dict(residual=.2), force_error=.1))
        after = dict(passed=False, force_check=dict(numerical=dict(residual=.4), force_error=.3))
        case = dict(case_id="public-case", passed=False, before=before, after=after)
        prompt = next_prompt([dict(score=0, cases=[case])] * 5)
        self.assertIn("projected residual: range 0.4..0.4", prompt)
        self.assertEqual(prompt.count("Representative public-case:"), 1)
        self.assertLess(len(prompt), 2500)

    def test_force_error_explained_without_oracle_values(self):
        check = dict(passed=False, force_check=dict(numerical=dict(passed=True), force_error=.002))
        guidance = explain_check(check)
        self.assertIn("Edge-force error 0.002 exceeds 0.0001", guidance)
        self.assertIn("Residual-only stopping", guidance)

    def test_bad_shape_and_feasibility_guidance(self):
        self.assertIn("finite FP32", explain_check(dict(passed=False, force_check=dict(
            numerical=dict(reason="wrong shape or nonfinite impulses")))))
        self.assertIn("nonnegativity", explain_check(dict(passed=False, force_check=dict(
            numerical=dict(feasible=False)))))

    def test_success_does_not_claim_performance(self):
        self.assertIn("not hardware", next_prompt([dict(score=1)]))
        self.assertIn("Execution failure", next_prompt([dict(score=0, error="compile error")]))


class ThroughputTests(unittest.TestCase):
    def test_unreviewed_candidate_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            candidate = Path(temp) / "candidate.py"
            candidate.write_text("raise RuntimeError('must not execute')\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                execute(Path(temp), Path(temp) / "out", "simulate", 2, 32, 1, 1, 2, candidate, "wrong")
            self.assertFalse((Path(temp) / "out").exists())
            with self.assertRaisesRegex(ValueError, "CPU backend"):
                execute(Path(temp), Path(temp) / "out", "cpu", 2, 32, 1, 1, 2, candidate, "wrong")

    def test_timing_uses_total_work_divided_by_total_time(self):
        rows = [dict(device_stats=dict(iterations=2, warmup_iterations=1,
                    durations_ms=[1., 3.], mean_ms=2.)),
                dict(device_stats=dict(iterations=2, warmup_iterations=1,
                    durations_ms=[2., 4.], mean_ms=3.))]
        result = aggregate_timing(rows, 8, 2, 1, "device")
        self.assertEqual(result["accepted_worlds_per_second"], 3200)
        rows[0]["device_stats"]["durations_ms"][0] = 0
        with self.assertRaises(ValueError):
            aggregate_timing(rows, 8, 2, 1, "device")

    @unittest.skipUnless(SUITE.exists(), "Requires local certified gripping development suite")
    def test_public_only_execution_grade_and_missing_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            self.assertEqual(execute(SUITE / "public", out, "cpu", 4, 32, 1, 1, 2), 0)
            summary = grade(SUITE, out)
            self.assertEqual(summary["expected_case_evaluations"], 16)
            self.assertFalse(summary["all_cases_pass"])
            self.assertFalse(summary["throughput_eligible"])
            self.assertNotIn("device_throughput", summary)
            prompt = next(out.glob("grading-*/next-prompt.txt")).read_text()
            self.assertIn("checker feedback", prompt)
            report = json.loads((out / "execution.json").read_text())
            report["attempts"].pop()
            (out / "execution.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                grade(SUITE, out)


if __name__ == "__main__":
    unittest.main()
