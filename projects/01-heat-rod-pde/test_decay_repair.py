import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sympy as sp
import decay_repair
import improved_agent
import level0_heatrod
import level1_heatrod
import pdecheck


class DecayRepairTests(unittest.TestCase):
    def test_derives_rates_for_multiple_conductivities_and_lengths(self):
        x, t = pdecheck.x, pdecheck.t
        for length in (1, 2, 3, 5):
            for k in (sp.Rational(1, 2), 1, 2, 3):
                spatial = -3*sp.sin(3*sp.pi*x/length) + sp.Rational(2, 5)*sp.cos(sp.pi*x/(2*length))
                candidate = sp.exp(-t)*spatial
                result = decay_repair.repair(sp.sstr(candidate), k)
                self.assertTrue(result['applied'], result)
                fixed = pdecheck.parse(pdecheck.extract(result['answer']))
                self.assertEqual(sp.simplify(fixed.subs(t, 0)-spatial), 0)
                self.assertEqual(sp.simplify(sp.diff(fixed,t)-k*sp.diff(fixed,x,2)), 0)

    def test_correct_modes_coefficients_and_start_error_are_preserved(self):
        problem = level1_heatrod.make(3)
        answer = ('(32/pi**3)*exp(-2*pi**2*t)*sin(pi*x/2) + '
                  '(32/(27*pi**3))*exp(-8*pi**2*t)*sin(3*pi*x/2) + '
                  '(32/(125*pi**3))*exp(-18*pi**2*t)*sin(5*pi*x/2)')
        before = pdecheck.check(problem, answer)
        fixed = decay_repair.repair(answer, problem['k'])
        after = pdecheck.check(problem, fixed['answer'])
        self.assertEqual(before['reward'], .6)
        self.assertEqual(after['reward'], 1.0)
        self.assertEqual(before['start_error'], after['start_error'])

    def test_declines_unsupported_or_already_correct_expressions(self):
        for answer in ('exp(-t**2)*sin(pi*x)', 'x*exp(-t)*sin(pi*x)',
                       'exp(-x*t)*sin(pi*x)', 'sin(x*x)', '__import__("os")',
                       'exp(-pi**2*t)*sin(pi*x)'):
            self.assertFalse(decay_repair.repair(answer, 1)['applied'], answer)

    def test_agent_accepts_original_checker_tool_output_and_logs_both_answers(self):
        problem = level0_heatrod.make(1)
        a = SimpleNamespace(rounds=1, workers=1, samples=1, seed=0, decay_repair=True)
        raw = 'u(x, t) = ' + sp.sstr(problem['f'])
        log = io.StringIO()
        response = dict(answer=raw, tool_calls=0, trace=[], error=None)
        with patch.object(improved_agent, 'attempt', return_value=response):
            result = improved_agent.solve(problem, a, log, 'test')
        self.assertEqual(result['status'], 'solved')
        row = json.loads(log.getvalue())
        self.assertEqual(row['answer'], raw)
        self.assertNotEqual(row['executed_answer'], raw)
        self.assertEqual(row['trace'][-1]['input_grade']['reward'], .6)
        self.assertEqual(row['grade']['grading_policy'], 'original_checker')

    def test_agent_does_not_repair_wrong_initial_shape(self):
        problem = level0_heatrod.make(1)
        a = SimpleNamespace(rounds=1, workers=1, samples=1, seed=0, decay_repair=True)
        response = dict(answer='u(x, t) = 2*sin(3*pi*x)', tool_calls=0, trace=[], error=None)
        with patch.object(improved_agent, 'attempt', return_value=response), \
             patch.object(decay_repair, 'repair') as repair:
            result = improved_agent.solve(problem, a, io.StringIO(), 'test')
        repair.assert_not_called()
        self.assertEqual(result['status'], 'unsolved')


if __name__ == '__main__':
    unittest.main()
