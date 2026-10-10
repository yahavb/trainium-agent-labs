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
import validation


class PhysicsTests(unittest.TestCase):
    def test_exact_answers_across_seeds_without_oracle(self):
        for mod in (level0, level1):
            for seed in range(3):
                for sub in mod.SUBS:
                    problem = mod.make(sub, seed)
                    answer = problem['exact']
                    if answer is None:
                        continue
                    problem['exact'] = None
                    self.assertTrue(validation.verify(problem, sp.sstr(answer))['accepted'])

    def test_series_truncation_threshold(self):
        problem = level1.make(3)
        for terms in (1, 2, 3, 5):
            self.assertEqual(validation.verify(problem, sp.sstr(level1.series_answer(problem, terms)))['accepted'], terms >= 3)

    def test_wrong_decay_and_boundary(self):
        problem = level1.make(1)
        self.assertFalse(validation.verify(problem, sp.sstr(problem['f']))['parts']['equation'])
        self.assertFalse(validation.verify(problem, 'exp(-pi**2*t)*sin(pi*x)')['parts']['right_bc'])

    def test_high_frequency_grid_alias_is_rejected(self):
        problem = level0.make(1)
        answer = sp.sstr(problem['exact']) + f" + exp(-1000000*t)*sin(1600*pi*x/{problem['L']})"
        self.assertEqual(pdecheck.check(problem, answer)['reward'], 1.0)
        self.assertFalse(validation.verify(problem, answer)['accepted'])
        grade = validation.grade(problem, answer)
        self.assertEqual(grade['original_reward'], 1.0)
        self.assertTrue(all(grade['original_parts'].values()))
        self.assertEqual(grade['validation_status'], 'failed')
        self.assertLess(grade['reward'], 1.0)

    def test_original_score_and_validation_status_are_distinct(self):
        problem = level0.make(1)
        good = validation.grade(problem, sp.sstr(problem['exact']))
        self.assertEqual(good['original_reward'], 1.0)
        self.assertEqual(good['validation_status'], 'passed')
        wrong = validation.grade(problem, sp.sstr(problem['f']))
        self.assertEqual(wrong['original_reward'], wrong['reward'])
        self.assertEqual(wrong['validation_status'], 'not_run')
        invalid = validation.grade(problem, 'x.__class__')
        self.assertIsNone(invalid['original_reward'])
        self.assertEqual(invalid['validation_status'], 'syntax_rejected')

    def test_untrusted_expression_rejected(self):
        for answer in ('x.__class__', '__import__("os")', '[x for x in (1,2)]', 'sin(x).evalf()', 'x if t else 0'):
            self.assertEqual(validation.grade(level0.make(1), answer)['reward'], 0.0)


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
