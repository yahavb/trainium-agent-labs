import ast
import copy
import json
import hashlib
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from math_harness import main
from math_program import evaluate, lower
from math_tasks import check, fixture
from qwen_math import request
from math_agent_loop import MathAgent


SPRING_GRAPH = {"operations": [
    {"id": "v0", "op": "mul_neg", "inputs": ["displacement", "stiffness"]},
    {"id": "v1", "op": "mul_neg", "inputs": ["velocity", "damping"]},
    {"id": "v2", "op": "add", "inputs": ["v0", "v1"]}], "result": "v2"}
SUM_GRAPH = {"operations": [{"id": "v0", "op": "sum", "inputs": ["forces"]}], "result": "v0"}


class MathAgentTests(unittest.TestCase):
    def test_proposed_graphs_pass_all_cpu_equation_checks(self):
        for task, graph in (("spring", SPRING_GRAPH), ("net-force", SUM_GRAPH)):
            ast.parse(lower(task, graph))
            for seed in range(16):
                inputs = fixture(task, seed)
                originals = tuple(a.copy() for a in inputs)
                self.assertEqual(check(task, evaluate(task, graph, inputs), inputs, originals)["score"], 1)

    def test_invalid_graphs_fail_before_execution(self):
        variants = []
        for change in ({"op": "eval"}, {"id": "__import__"}, {"inputs": ["missing"]},
                       {"engine": "scalar"}, {"extra": "side effect"}):
            graph = copy.deepcopy(SUM_GRAPH)
            graph["operations"][0].update(change)
            variants.append(graph)
        variants += [{"operations": [], "result": "forces"},
                     {"operations": SUM_GRAPH["operations"] * 17, "result": "v0"},
                     {"operations": SUM_GRAPH["operations"], "result": "forces"}]
        for graph in variants:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                lower("net-force", graph)

    def test_bad_math_is_rejected_even_when_program_is_legal(self):
        graph = {"operations": [{"id": "v0", "op": "copy", "inputs": ["displacement"]}], "result": "v0"}
        inputs = fixture("spring", 3)
        self.assertEqual(check("spring", evaluate("spring", graph, inputs), inputs, inputs)["score"], 0)

    def test_scalar_layout_is_validated(self):
        graph = copy.deepcopy(SPRING_GRAPH)
        graph["operations"][0]["inputs"][1] = "velocity"
        with self.assertRaises(ValueError):
            lower("spring", graph)

    def test_prompt_contains_baseline_and_measured_feedback_not_exact_target(self):
        payload = request("spring", "Previous proposal throughput_ratio=0.95", {"program": SPRING_GRAPH})
        user = payload["messages"][1]["content"]
        self.assertIn("def calculate", user)
        self.assertIn("throughput_ratio=0.95", user)
        self.assertIn("No technique is assigned", payload["messages"][0]["content"])
        self.assertNotIn("Return this exact human-reviewed template", user)

    def test_runner_checks_actual_proposal_not_fixed_template(self):
        bad = {"operations": [{"id": "v0", "op": "copy", "inputs": ["displacement"]}], "result": "v0"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proposal = root / "proposal.json"
            proposal.write_text(json.dumps({"hypothesis": "incorrect identity", "program": bad}))
            with patch("sys.argv", ["math_harness", "--task", "spring", "--backend", "cpu",
                                    "--program", str(proposal), "--out", str(root / "results")]):
                self.assertEqual(main(), 1)
            report = json.loads((root / "results/spring/results.json").read_text())
            self.assertIsNone(report["winner"])
            self.assertEqual(report["attempts"][1]["status"], "correctness_failed")
            self.assertEqual(report["attempts"][1]["correctness_score"], 0)
            self.assertIsNone(report["attempts"][1]["throughput_ratio"])

    def test_feedback_reaches_next_attempt_and_slower_proposal_does_not_replace_best(self):
        class FakeAgent(MathAgent):
            def command(self, argv, log): pass
            def upload(self, local, remote, log): pass
            def start_model(self, log): pass
            def stop_model(self, log): pass
            def remote_command(self, argv, log): pass
            def download(self, remote, local, log):
                local.mkdir()
                program = copy.deepcopy(SPRING_GRAPH)
                if "001" in remote:
                    program["operations"][0]["engine"] = "vector"
                source = lower("spring", program)
                self.proposal_hash = hashlib.sha256(source.encode()).hexdigest()
                (local / "proposal.json").write_text(json.dumps({"hypothesis": "mock hypothesis", "program": program}))
                (local / "generation-attempt.json").write_text(json.dumps(dict(
                    status="generated_not_evaluated", candidate_sha256=hashlib.sha256(source.encode()).hexdigest())))
            def evaluate_math(self, task, proposal, backend, remote, local, log):
                row = dict(correctness_score=1.0, status="correct_not_benchmarked", throughput_ratio=None,
                           source_sha256=self.proposal_hash)
                if backend == "device":
                    row.update(status="benchmarked", throughput_ratio=.99 if "001" in remote else 1.02)
                return {}, row

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "loop"
            args = Namespace(out=out, seat="mock-seat", minutes=1, tasks="spring", attempts=2,
                             runs=5, iters=20, cores="0,1", base_url="http://localhost:8000/v1")
            FakeAgent(args).run()
            best = json.loads((out / "best.json").read_text())["spring"]
            self.assertEqual(best["attempt"], 0)
            self.assertEqual(best["throughput_ratio"], 1.02)
            self.assertIn("1.02", (out / "attempt-001/feedback.txt").read_text())
            self.assertEqual(len((out / "attempts.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
