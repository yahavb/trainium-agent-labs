import importlib.util
import unittest

import numpy as np
import sympy as sp

from controllerlab.physics import STARTER, checker, coeff_method, decay_report, problem_of, prompt_of, reference, x, t
from controllerlab.problems import make


class PhysicsTests(unittest.TestCase):
    def test_reference_answers_all_fifteen_types(self):
        for level in range(5):
            for sub in (1, 2, 3):
                with self.subTest(level=level, sub=sub):
                    record = make(level, sub, references=True)
                    answer = reference(record)
                    result = checker.check(problem_of(record), sp.sstr(answer))
                    self.assertEqual(result["reward"], 1.0, result["feedback"])

    def test_representative_failures(self):
        record = make(0, 1, references=True)
        p = problem_of(record)
        exact = reference(record)
        self.assertEqual(checker.check(p, sp.sstr(p["f"]))["reward"], 0.6)
        self.assertLess(checker.check(p, sp.sstr(exact.subs(x, x+p["L"]/2)))["reward"], 1)
        wrong = checker.check(p, sp.sstr(2*exact))
        self.assertEqual(wrong["reward"], 0.8)
        self.assertIn("too large", wrong["feedback"])
        for bad in ("No answer.", "u(x,t) = a*sin(pi*x)", "u(x,t) = float('nan')"):
            self.assertEqual(checker.check(p, bad)["reward"], 0)
        record = make(3, 3, references=True)
        exact = reference(record)
        p = problem_of(record)
        constant = sp.integrate(p["f"], (x, 0, p["L"]))/p["L"]
        self.assertEqual(checker.check(p, sp.sstr(exact-constant))["reward"], 0.8)
        record = make(1, 3)
        self.assertEqual(checker.check(problem_of(record), sp.sstr(reference(record, 1)))["reward"], 0.8)

    def test_original_scores_unchanged(self):
        spec = importlib.util.spec_from_file_location("untouched_checker", STARTER/"pdecheck.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        original.np = checker.np
        for level in (0, 1):
            for sub in (1, 2, 3):
                record = make(level, sub, references=True)
                p = problem_of(record)
                for answer in (sp.sstr(reference(record, 4)), "0", sp.sstr(p["f"]), "u(x,t)=unknown"):
                    with self.subTest(level=level, sub=sub, answer=answer[:60]):
                        a, b = original.check(p, answer), checker.check(p, answer)
                        self.assertEqual(a["reward"], b["reward"])
                        self.assertEqual(a["parts"], b["parts"])
                        self.assertEqual(a["start_error"], b["start_error"])

    def test_hints_constant_norm_and_no_reference_leak(self):
        record = make(3, 1, references=True)
        p = problem_of(record)
        self.assertIn("constant mode has norm L", coeff_method(p))
        feedback = checker.check(p, "0")["feedback"]
        self.assertIn("missing", feedback)
        self.assertNotIn("should be", feedback)
        mixed = make(3, 3, references=True)
        self.assertNotIn(mixed["reference"], prompt_of(mixed))
        record = make(2, 1)
        self.assertIn("u_x(0,t)=0", prompt_of(record))
        self.assertIn("u(", prompt_of(record))
        record = make(3, 3, references=True)
        p = problem_of(record)
        answer = reference(record)
        grid = np.linspace(0, float(p["L"]), 801)
        self.assertEqual(decay_report(p, answer, grid, float(p["L"]), p["k"]), "")
        wrong = answer*sp.exp(-t)
        self.assertIn("constant mode does not decay", decay_report(p, wrong, grid, float(p["L"]), p["k"]))
