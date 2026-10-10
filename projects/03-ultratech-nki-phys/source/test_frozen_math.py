from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from frozen_math_eval import freeze, grade, heldout_inputs, input_signature, manifest, remote_evaluation, sha, verified
from math_program import lower
from math_tasks import TASKS, fixture


class FrozenMathTests(unittest.TestCase):
    def create_run(self, root):
        run = root / "development"
        run.mkdir()
        programs = {
            "spring": {"operations": [
                {"id": "v0", "op": "mul", "inputs": ["displacement", "stiffness"]},
                {"id": "v1", "op": "mul", "inputs": ["velocity", "damping"]},
                {"id": "v2", "op": "add", "inputs": ["v0", "v1"]},
                {"id": "v3", "op": "neg", "inputs": ["v2"]}], "result": "v3"},
            "net-force": {"operations": [{"id": "v0", "op": "sum", "inputs": ["forces"]}], "result": "v0"},
        }
        winners, rows = {}, []
        for index, (task, program) in enumerate(programs.items()):
            source = lower(task, program)
            directory = run / f"attempt-{index:03d}/generation"
            directory.mkdir(parents=True)
            (directory / "proposal.json").write_text(json.dumps({"program": program}))
            (run / f"best-{task}.py").write_text(source)
            rows.append(dict(task=task, attempt=index, status="benchmarked", correctness_score=1.0,
                             candidate_sha256=hashlib.sha256(source.encode()).hexdigest()))
            winners[task] = dict(winner="agent-proposal", attempt=index)
        (run / "best.json").write_text(json.dumps(winners))
        (run / "attempts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return run

    def test_inputs_are_distinct_from_development_and_each_other(self):
        for task in TASKS:
            public = {input_signature(fixture(task, seed)) for seed in range(16)}
            cases = heldout_inputs(task)
            signatures = {signature for _, _, signature in cases}
            self.assertEqual(len(cases), 32)
            self.assertEqual(len(signatures), 32)
            self.assertFalse(signatures & public)
            self.assertTrue(all(seed >= 10000 for seed, _, _ in cases))

    def test_freeze_pins_sources_inputs_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = freeze(self.create_run(root), root / "bundle")
            info = manifest(bundle)
            self.assertEqual(sum(len(t["cases"]) for t in info["tasks"].values()), 64)
            for task in info["tasks"].values():
                verified(bundle, task["kernel"], task["kernel_sha256"])
                for case in task["cases"]:
                    verified(bundle, case["file"], case["sha256"])
            (bundle / "manifest.json").write_text("{}")
            with self.assertRaises(ValueError): manifest(bundle)

    def test_stale_winner_is_rejected_before_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.create_run(root)
            (run / "best-spring.py").write_text("changed")
            with self.assertRaises(ValueError): freeze(run, root / "bundle")
            self.assertFalse((root / "bundle").exists())

    def test_missing_outputs_are_unmeasured_not_correctness_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = freeze(self.create_run(root), root / "bundle")
            outputs = root / "outputs"
            outputs.mkdir()
            (outputs / "context.json").write_text(json.dumps(dict(backend="device", manifest_sha256=sha(bundle / "manifest.json"))))
            (outputs / "execution.json").write_text("[]")
            self.assertFalse(grade(bundle, outputs, root / "grade"))
            report = json.loads((root / "grade/results.json").read_text())
            self.assertTrue(all(row["score"] is None for row in report["cases"]))

    def test_repeat_assessment_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = freeze(self.create_run(root), root / "bundle")
            (bundle / "assessment-started.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "already has a final assessment"):
                remote_evaluation(Namespace(bundle=bundle))


if __name__ == "__main__":
    unittest.main()
