import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import httpx
import sympy as sp
import improved_agent as improved
import level0_heatrod as level0
import level1_heatrod as level1
import pdecheck
import json
from answer_syntax import checked_expression


class CheckerParityTests(unittest.TestCase):
    def test_original_acceptance_and_parser_are_not_overridden(self):
        problem = level0.make(1)
        answers = [sp.sstr(problem['exact']) +
                   f" + exp(-1000000*t)*sin(1600*pi*x/{problem['L']})",
                   'exp(-9*pi**2*t) sin(3*pi*x)']
        settings = SimpleNamespace(rounds=2, samples=1, workers=1, seed=0)
        for answer in answers:
            with self.subTest(answer=answer):
                expected = pdecheck.check(problem, answer)
                self.assertEqual(expected['reward'], 1.0)
                response = dict(answer=answer, tool_calls=0, trace=[], error=None)
                log = io.StringIO()
                with patch.object(improved, 'attempt', return_value=response) as attempt:
                    result = improved.solve(problem, settings, log, 'parity')
                self.assertEqual(result['status'], 'solved')
                self.assertEqual(result['rounds'], 1)
                attempt.assert_called_once()
                grade = json.loads(log.getvalue())['grade']
                for key, value in expected.items():
                    self.assertEqual(grade[key], value)

    def test_syntax_limits_apply_only_to_the_optional_tool(self):
        for answer in ('x.__class__', '__import__("os")', '[x for x in (1,2)]',
                       'sin(x).evalf()', 'x if t else 0'):
            with self.assertRaises((ValueError, SyntaxError)):
                checked_expression(answer)


class AgentTests(unittest.TestCase):
    def settings(self):
        return SimpleNamespace(model='test', base='http://localhost/v1', max_tokens=100,
                               retries=1, timeout=1, no_tools=False, tool_steps=2,
                               rounds=2, samples=1, workers=1, seed=0)

    def test_http_500_retry(self):
        replies = [httpx.Response(500), httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}}]})]
        trace = []
        with patch.object(httpx, 'post', side_effect=replies) as post, patch.object(improved.time, 'sleep'):
            self.assertEqual(improved.ask(self.settings(), [], trace, 0), 'answer')
        # Neuron's sampler does not support request-specific torch.Generator seeds.
        self.assertNotIn('seed', post.call_args.kwargs['json'])
        self.assertEqual(trace[0]['type'], 'http_error')

    def test_service_failure_is_not_math_failure(self):
        with patch.object(httpx, 'post', side_effect=httpx.ConnectError('unreachable')), patch.object(improved.time, 'sleep'):
            result = improved.solve(level0.make(1), self.settings(), io.StringIO(), 'test')
        self.assertEqual(result['status'], 'infrastructure_error')
        self.assertIsNone(result['reward'])

    def test_tool_trace_retains_integral(self):
        with patch.object(improved, 'ask', side_effect=['COMPUTE: Integral(x, (x, 0, 2))', 'u(x, t) = 0']):
            result = improved.attempt(self.settings(), 'problem', 0)
        self.assertEqual(result['tool_calls'], 1)
        self.assertIn('2', result['trace'][0]['results'])

    def test_best_answer_is_not_discarded(self):
        a = self.settings()
        a.rounds = 3
        captured = []
        problem = level0.make(1)
        answers = ['u(x, t) = ' + sp.sstr(problem['exact'] * 2), 'u(x, t) = nonsense',
                   'u(x, t) = ' + sp.sstr(problem['exact'])]
        def fake_attempt(settings, prompt, seed):
            captured.append(prompt)
            return {'answer': answers.pop(0), 'tool_calls': 0, 'trace': [], 'error': None}
        with patch.object(improved, 'attempt', side_effect=fake_attempt):
            result = improved.solve(problem, a, io.StringIO(), 'test')
        self.assertEqual(result['status'], 'solved')
        self.assertIn('Best verified attempt so far: u(x, t) = ' + sp.sstr(problem['exact'] * 2), captured[2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
