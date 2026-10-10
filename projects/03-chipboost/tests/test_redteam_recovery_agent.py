"""Mocked loop verifies recovery retains correct candidate/feedback and exact attempt accounting."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "redteam_recovery_agent.py").read_text())
names = {"first_prompt", "repair_prompt", "run_once", "strip_module_docstring", "better", "says"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PRESERVE" for t in n.targets)]
ns = dict(ast=ast, hashlib=hashlib, os=os, time=time, json=json, OP="matmul", EXPERIMENT="redteam_recovery_qwen", TASK="TASK", KEEP="KEEP", MAKE_FASTER="faster",
          REJECTED=("wrong", "rules", "heldout_fail"), RANK={"wrong":1, "no_gain":5, "faster":6},
          agent02=types.SimpleNamespace(API_CARD="API DOCUMENTATION", extract_code=lambda x:x))
exec(compile(ast.Module(body=nodes, type_ignores=[]), "controller_test", "exec"), ns)

class RecoveryTests(unittest.TestCase):
    def test_reset_preserves_best_feedback_and_counts_every_candidate(self):
        prompts, calls = [], []
        outcomes = ["faster", "wrong", "wrong", "faster", "wrong", "wrong", "no_gain"]
        def grade(src, *args):
            i = len(calls)
            calls.append(src)
            verdict = outcomes[i]
            return dict(verdict=verdict, instruction_given=f"safe hint {i}", referee_message="UNTRUSTED EXCEPTION",
                        speedup=1.517 if i==0 else 1.8 if i==3 else None, source="chip"), None
        def ask(a,prompt,n):
            prompts.append(prompt)
            return [(f"candidate={len(prompts)}",100)]*n
        ns.update(grade=grade, ask_parallel=ask, schema=types.SimpleNamespace(
            ATTEMPT_FIELDS={k:None for k in ("verdict","instruction_given","referee_message","speedup","source","code_hash","timestamp")}, validate=lambda r: []))
        a=types.SimpleNamespace(offline=False,dry=False,arm="referee",tag="test",rounds=None,budget=6,samples=1,seat=100)
        with tempfile.TemporaryDirectory() as td:
            ns["HERE"]=td
            path=Path(td)/"start.py"
            path.write_text("baseline=1")
            log=io.StringIO()
            out=ns["run_once"](a,("speedcheck",object()),str(path),0,log,td)
        self.assertEqual(out["attempts"],6)
        self.assertEqual(len(calls),7)
        self.assertEqual(len(log.getvalue().splitlines()),6)
        self.assertIn("baseline=1",prompts[2])
        self.assertIn("safe hint 0",prompts[2])
        self.assertIn("candidate=3",prompts[5])
        self.assertIn("safe hint 3",prompts[5])
        for prompt in prompts:
            self.assertIn("API DOCUMENTATION",prompt)
            self.assertIn("full K // TILE_K contraction loop",prompt)
            self.assertNotIn("UNTRUSTED EXCEPTION",prompt)
            self.assertNotIn("already been tried and did not work",prompt)

if __name__ == "__main__":
    unittest.main()
