"""Checks for reproducible measurement and the model/calculator exchange."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import agent
import level1_heatrod
import run_experiments as experiments


class ExperimentTests(unittest.TestCase):
    def test_configs_change_only_feedback(self):
        baseline = experiments.load_config(PROJECT / "configs/baseline.json")
        structured = experiments.load_config(PROJECT / "configs/structured.json")
        changed = {k for k in baseline if baseline[k] != structured[k]}
        self.assertEqual(changed, {"feedback_style"})

    def test_bad_config_fails_before_starting_a_run(self):
        cases = ({"samples": 0}, {"rounds": True}, {"no_tools": "false"},
                 {"tool_steps": -1}, {"feedback_style": "unknown"}, {"sample": 4})
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.json"
            for case in cases:
                with self.subTest(case=case):
                    config.write_text(json.dumps(case))
                    with self.assertRaises(ValueError):
                        experiments.load_config(config)

    def test_tool_exchange_records_actual_prompts_results_and_usage(self):
        replies = [
            dict(choices=[dict(message=dict(content="COMPUTE: 2 + 3"), finish_reason="stop")],
                 usage=dict(prompt_tokens=100, completion_tokens=10)),
            dict(choices=[dict(message=dict(content="u(x, t) = 5"), finish_reason="length")],
                 usage=dict(prompt_tokens=130, completion_tokens=20)),
        ]
        responses = [SimpleNamespace(status_code=200, json=lambda r=r: r) for r in replies]
        args = SimpleNamespace(offline=False, no_tools=False, tool_steps=1,
                               model="test-model", base="http://model.invalid/v1",
                               max_tokens=1200, think=False)
        with patch("httpx.post", side_effect=responses) as post:
            answer, used, trace = agent.one_attempt(args, None, "question", 0)
        self.assertEqual(answer, "u(x, t) = 5")
        self.assertEqual(used, 1)
        self.assertEqual(len(trace), 2)
        self.assertIn("= 5", trace[0]["tool_results"])
        self.assertIn(trace[0]["tool_results"], trace[1]["prompt"])
        self.assertEqual(trace[1]["finish_reason"], "length")
        self.assertEqual(trace[0]["usage"]["completion_tokens"], 10)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["messages"][0]["content"],
                         trace[1]["prompt"])

    def test_feedback_preserves_original_baseline_and_uses_checker_status(self):
        best = dict(expr="candidate", feedback="The coefficient is too small.",
                    parts=dict(equation=True, left_bc=True, right_bc=True, start_shape=False))
        self.assertEqual(agent.repair_prompt("question", best, "baseline"),
                         "question\n\nA previous attempt was:\n  u(x, t) = candidate\n"
                         "A checker found this problem with it: The coefficient is too small.\nFix it.")
        structured = agent.repair_prompt("question", best, "structured")
        self.assertIn("heat equation: PASS", structured)
        self.assertIn("initial temperature: FAIL", structured)
        self.assertIn(best["feedback"], structured)

    def test_solve_logs_checker_feedback_and_round_timing(self):
        problem = level1_heatrod.make(3)
        args = SimpleNamespace(rounds=2, samples=1, feedback_style="structured", offline=False)
        answer = "u(x, t) = " + str(level1_heatrod.series_answer(problem, 3))
        log = io.StringIO()
        with patch.object(agent, "ask_round", return_value=[(answer, 0, [])]), redirect_stdout(io.StringIO()):
            reward, rounds = agent.solve(problem, args, log)
        self.assertEqual((reward, rounds), (1.0, 1))
        record = json.loads(log.getvalue())
        self.assertEqual(record["feedback"], "Solved.")
        self.assertTrue(all(record["parts"].values()))
        self.assertGreaterEqual(record["checker_seconds"], 0)
        self.assertIn("generation_and_tools_seconds", record)

    def test_incomplete_batches_do_not_report_a_solve_rate(self):
        result = dict(status="complete", solved=True, rounds=1, wall_seconds=1.0)
        self.assertIsNone(experiments.summary_of([result], 5, False)["solve_rate"])
        self.assertIsNone(experiments.summary_of([result], 1, True)["solve_rate"])
        self.assertEqual(experiments.summary_of([result], 1, False)["solve_rate"], 1.0)

    def test_interrupt_stops_child_and_marks_batch_incomplete(self):
        process = MagicMock()
        process.stdout.__iter__.side_effect = KeyboardInterrupt
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict("os.environ", {"HEATROD_BASE_URL": "http://model.invalid/v1"}), \
                    patch.object(experiments.subprocess, "Popen", return_value=process), \
                    redirect_stdout(io.StringIO()):
                code = experiments.main(["--repeats", "1", "--output-root", folder])
            self.assertEqual(code, 130)
            process.terminate.assert_called_once()
            batch = next(Path(folder).iterdir())
            summary = json.loads((batch / "summary.json").read_text())
            self.assertFalse(summary["batch_complete"])
            self.assertEqual(summary["results"][0]["status"], "interrupted")
            self.assertIsNone(summary["solve_rate"])

    def test_offline_runner_snapshots_source_and_keeps_scores_out_of_real_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            for style in ("baseline", "structured"):
                process = subprocess.run([
                    sys.executable, str(PROJECT / "run_experiments.py"), "--offline",
                    "--config", str(PROJECT / f"configs/{style}.json"),
                    "--repeats", "1", "--output-root", folder,
                ], capture_output=True, text=True, timeout=90)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            batches = list(Path(folder).iterdir())
            self.assertEqual(len(batches), 2)
            for batch in batches:
                summary = json.loads((batch / "summary.json").read_text())
                self.assertEqual(summary["kind"], "offline_preview")
                self.assertTrue(summary["batch_complete"])
                self.assertIsNone(summary["solve_rate"])
                self.assertIsNone(summary["median_wall_seconds"])
                self.assertEqual((batch / "source/agent.py").read_text(),
                                 (PROJECT / "agent.py").read_text())
                attempts = batch / "run-01/attempts.jsonl"
                records = [json.loads(line) for line in attempts.read_text().splitlines()]
                self.assertTrue(records)
                self.assertTrue(all(r["offline"] for r in records))
                self.assertTrue((batch / "run-01/console.log").is_file())


if __name__ == "__main__":
    unittest.main()
