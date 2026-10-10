"""CPU controller regression; no model or hardware measurements are synthesized as results."""
import ast
import hashlib
import io
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
import json

SOURCE = Path(__file__).resolve().parents[1] / "experimental_agent.py"
tree = ast.parse(SOURCE.read_text())
names = {"code_identity", "recovery_prompt", "run_once", "strip_module_docstring", "better"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = dict(ast=ast, hashlib=hashlib, os=os, time=time, json=json, OP="matmul",
          EXPERIMENT="referee_recovery_v3", TASK="TASK", KEEP="KEEP", SHAPE_CONTRACT="SHAPES",
          REJECTED=("wrong", "rules", "heldout_fail"),
          RANK={"wrong": 1, "no_gain": 5, "faster": 6}, HERE=str(SOURCE.parent),
          agent02=types.SimpleNamespace(API_CARD="API CARD", extract_code=lambda x: x))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)


class ExperimentalTests(unittest.TestCase):
    def test_identity_ignores_formatting_comments(self):
        self.assertEqual(ns["code_identity"]("x=1 # comment"), ns["code_identity"]("x = 1"))
        self.assertNotEqual(ns["code_identity"]("x=1"), ns["code_identity"]("x=2"))

    def test_prompt_bounded_and_retains_api(self):
        history = [dict(attempt=i, identity=str(i), verdict="wrong", duplicate=True,
                        instruction="safe instruction") for i in range(7)]
        prompt = ns["recovery_prompt"]("x=1", "optimize", history, True)
        self.assertIn("restored", prompt)
        self.assertIn("API CARD", prompt)
        self.assertIn("repeated program", prompt)
        self.assertNotIn("attempt 2:", prompt)
        self.assertIn("attempt 3:", prompt)

    def test_two_failures_reset_and_duplicates_count(self):
        prompts, graded = [], []
        def grade(src, *args):
            graded.append(src)
            return dict(verdict="no_gain" if len(graded) == 1 else "wrong",
                        instruction_given="reuse rhs" if len(graded) == 1 else "fix DMA"), None
        def ask(a, prompt, n):
            prompts.append(prompt)
            return [("broken=1", 100)] * n
        ns.update(grade=grade, says=lambda r, ref: r["instruction_given"], ask_parallel=ask,
                  schema=types.SimpleNamespace(ATTEMPT_FIELDS={k: None for k in
                      ("verdict", "instruction_given", "code_hash", "timestamp")},
                      validate=lambda rec: []))
        a = types.SimpleNamespace(offline=False, dry=False, arm="referee", rounds=None,
                                  budget=3, samples=1, seat=100)
        with tempfile.TemporaryDirectory() as td:
            ns["HERE"] = td
            path = Path(td) / "start.py"
            path.write_text("correct=1")
            log = io.StringIO()
            result = ns["run_once"](a, ("test", object()), str(path), 0, log, td)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(len(graded), 4)  # startup plus three counted candidates
        self.assertIn("correct=1", prompts[2])
        self.assertIn("restored", prompts[2])
        self.assertIn("reuse rhs", prompts[2])
        self.assertIn("fix DMA", prompts[2])
        self.assertEqual(len(log.getvalue().splitlines()), 3)
        self.assertIn("referee_recovery_v3", result["run_id"])


if __name__ == "__main__":
    unittest.main()
