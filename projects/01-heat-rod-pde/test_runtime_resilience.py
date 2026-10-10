"""Failure-path regressions. Controlled responses here are not model benchmarks."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sympy as sp

import compare_agents
import benchmark
import improved_agent
import level0_heatrod
import level1_heatrod
import pdecheck
import tool_calc
from attempt_records import read_attempts
from checker_runtime import grade_candidate


def response(answer, **kwargs):
    return dict(answer=answer, tool_calls=0, trace=[], error=None, **kwargs)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.problem = level0_heatrod.make(1)
        self.correct = sp.sstr(self.problem['exact'])
        self.settings = SimpleNamespace(rounds=2, samples=1, workers=1, seed=0)

    def test_checker_exception_is_logged_and_next_round_can_solve(self):
        for bad in ('u(x,t) = [1,2]', 'u(x,t) = (1,2)'):
            with self.subTest(bad=bad):
                log = io.StringIO()
                with patch.object(improved_agent, 'attempt',
                                  side_effect=[response(bad), response(self.correct)]) as attempt:
                    result = improved_agent.solve(self.problem, self.settings, log, 'recovery')
                self.assertEqual(result['status'], 'solved')
                self.assertEqual(attempt.call_count, 2)
                self.assertIn('one scalar expression', attempt.call_args.args[1])
                rows = [json.loads(line) for line in log.getvalue().splitlines()]
                self.assertEqual(rows[0]['answer'], bad)
                self.assertIsNone(rows[0]['grade']['reward'])
                self.assertIsNone(rows[0]['grade']['original_reward'])
                self.assertEqual(rows[0]['grade']['evaluation_error']['type'], 'AttributeError')
                self.assertEqual(result['evaluation_errors'], 1)

    def test_all_checker_exceptions_end_with_unknown_score_within_budget(self):
        log = io.StringIO()
        with patch.object(improved_agent, 'attempt', side_effect=lambda *a: response('[1,2]')) as attempt:
            result = improved_agent.solve(self.problem, self.settings, log, 'unknown')
        self.assertEqual(result['status'], 'evaluation_error')
        self.assertIsNone(result['reward'])
        self.assertIsNone(result['original_reward'])
        self.assertEqual(attempt.call_count, self.settings.rounds)
        self.assertEqual(len(log.getvalue().splitlines()), 2)

    def test_grade_failure_does_not_discard_previous_scored_candidate(self):
        raw = sp.sstr(self.problem['exact'] * 2)
        with patch.object(improved_agent, 'attempt',
                          side_effect=[response(raw), response('[1,2]')]):
            result = improved_agent.solve(self.problem, self.settings, io.StringIO(), 'best')
        self.assertEqual(result['status'], 'unsolved')
        self.assertEqual(result['reward'], pdecheck.check(self.problem, raw)['reward'])
        self.assertEqual(result['answer'], raw)

    def test_original_checker_results_remain_unchanged(self):
        for module, sub in ((level0_heatrod, 1), (level0_heatrod, 2),
                            (level1_heatrod, 1), (level1_heatrod, 3)):
            problem = module.make(sub, 0)
            for answer in ('0', 'sin(pi*x)', 'u(x,t) = nonsense',
                           'exp(-9*pi**2*t) sin(3*pi*x) + Rational(0,1)'):
                with self.subTest(level=problem['name'], answer=answer):
                    official = pdecheck.check(problem, answer)
                    wrapped = grade_candidate(problem, answer)
                    for key in ('reward', 'parts', 'expr', 'feedback', 'start_error'):
                        self.assertEqual(wrapped[key], official[key])
                    self.assertIsNone(wrapped['evaluation_error'])

    def test_checker_interrupt_is_not_swallowed(self):
        with patch.object(pdecheck, 'check', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                grade_candidate(self.problem, self.correct)

    def test_failed_request_stops_after_logging_all_inflight_samples(self):
        self.settings.samples = 2
        failed = dict(answer='', tool_calls=0, trace=[dict(type='http_error', status=500)],
                      error='HTTP 500 after 1 requests')
        log = io.StringIO()
        with patch.object(improved_agent, 'attempt',
                          side_effect=[response(self.correct), failed]) as attempt:
            result = improved_agent.solve(self.problem, self.settings, log, 'service')
        self.assertEqual(attempt.call_count, 2)
        self.assertEqual(result['status'], 'infrastructure_error')
        self.assertEqual(result['reward'], 1.0)  # Preserve the successful in-flight answer.
        self.assertEqual(result['requests'], 1)
        self.assertEqual(len(log.getvalue().splitlines()), 2)

    def test_agent_cli_saves_failure_and_does_not_start_next_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['improved_agent.py', '--all', '--output', tmp]
            with patch.object(sys, 'argv', argv), patch.object(improved_agent, 'solve',
                    return_value=dict(status='infrastructure_error', reward=None)) as solve:
                code = improved_agent.main()
            self.assertEqual(code, 1)
            solve.assert_called_once()
            doc = json.loads(next(Path(tmp).glob('*/summary.json')).read_text())
            self.assertEqual(doc['results'][0]['status'], 'infrastructure_error')

    def test_calculator_container_errors_are_returned_with_trace_and_can_recover(self):
        self.settings.no_tools = False
        self.settings.tool_steps = 1
        for expression in ('(1,2)', '[Integral(x,(x,0,1)),Integral(x,(x,0,2))]', 'True'):
            with self.subTest(expression=expression), patch.object(
                    improved_agent, 'ask', side_effect=['COMPUTE: ' + expression, self.correct]):
                result = improved_agent.attempt(self.settings, 'problem', 0)
                self.assertIsNone(result['error'])
                self.assertEqual(result['answer'], self.correct)
                self.assertEqual(result['trace'][0]['requests'], [expression])
                self.assertIn('one scalar expression', result['trace'][0]['results'])
                self.assertEqual(result['tool_calls'], 1)
        self.assertTrue(tool_calc.compute('Integral(x, (x, 0, 2))').startswith('2'))
        self.assertTrue(tool_calc.compute('Rational(1,2)').startswith('1/2'))


class ComparisonResilienceTests(unittest.TestCase):
    def setUp(self):
        self.problem = level0_heatrod.make(1)
        self.row = dict(answer=sp.sstr(self.problem['exact']), round=0, trace=[])

    def summarize_bytes(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'attempts.jsonl'
            path.write_bytes(data)
            return compare_agents.summarize_attempts([path], self.problem)

    def test_partial_log_retains_complete_candidate_and_reports_exact_line(self):
        data = (json.dumps(self.row) + '\n{"answer":').encode()
        result = self.summarize_bytes(data)
        self.assertTrue(result['executed_original_solved'])
        self.assertEqual(result['candidates'], 1)
        self.assertFalse(result['artifacts_complete'])
        self.assertEqual(result['log_errors'][0]['line'], 2)

    def test_bad_unicode_and_bad_schema_do_not_erase_valid_records(self):
        data = (json.dumps(self.row) + '\n').encode() + b'\xff\n[]\n'
        result = self.summarize_bytes(data)
        self.assertEqual(result['candidates'], 1)
        self.assertEqual([error['line'] for error in result['log_errors']], [2, 3])
        self.assertFalse(result['artifacts_complete'])

    def test_corrupt_middle_line_is_reported_while_later_records_are_recovered(self):
        data = (json.dumps(self.row) + '\nbroken\n' + json.dumps(self.row) + '\n').encode()
        result = self.summarize_bytes(data)
        self.assertEqual(result['candidates'], 2)
        self.assertEqual(len(result['log_errors']), 1)

    def test_checker_failure_in_saved_candidate_does_not_crash_summary(self):
        rows = [dict(answer='[1,2]', round=0), self.row]
        result = self.summarize_bytes(('\n'.join(map(json.dumps, rows)) + '\n').encode())
        self.assertEqual(result['evaluation_failures'], 1)
        self.assertEqual(result['evaluation_errors'][0]['layer'], 'model_and_executed')
        self.assertTrue(result['executed_original_solved'])

    def test_failed_requests_count_and_unknown_tokens_stay_unknown(self):
        row = self.row | dict(trace=[dict(type='http_error', status=500),
                                    dict(type='transport_error'),
                                    dict(type='model', usage=dict(completion_tokens=27)),
                                    dict(type='calculator')])
        result = self.summarize_bytes(json.dumps(row).encode())
        self.assertEqual(result['model_requests'], 3)
        self.assertEqual(result['attempted_requests'], 3)
        self.assertEqual(result['successful_responses'], 1)
        self.assertEqual(result['failed_requests'], 2)
        self.assertIsNone(result['completion_tokens'])
        self.assertEqual(result['known_completion_tokens'], 27)
        self.assertEqual(result['unknown_usage_requests'], 2)

    def test_legacy_untyped_trace_and_absent_usage(self):
        row = self.row | dict(trace=[dict(usage=dict(completion_tokens=7)), dict(answer='x')])
        result = self.summarize_bytes(json.dumps(row).encode())
        self.assertEqual(result['attempted_requests'], 2)
        self.assertEqual(result['successful_responses'], 2)
        self.assertEqual(result['known_completion_tokens'], 7)
        self.assertIsNone(result['completion_tokens'])

    def test_missing_log_is_an_explicit_artifact_error(self):
        rows, errors = read_attempts([Path('/nonexistent-heatrod-review/attempts.jsonl')])
        self.assertEqual(rows, [])
        self.assertEqual(len(errors), 1)

    def run_batch(self, behavior, variants=('improved', 'decay')):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root/'cases.json'
            cases.write_text('[[0,1,0]]')
            commands = []
            row = self.row

            class Process:
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    case = Path(command[command.index('--output')+1])
                    folder = case/'test-run'
                    folder.mkdir()
                    status = 'infrastructure_error' if behavior.startswith('service') else 'solved'
                    summary = folder/'summary.json'
                    summary.write_text('{"results":' if behavior == 'bad_summary' else
                                       json.dumps(dict(results=[dict(status=status)])))
                    attempts = folder/'attempts.jsonl'
                    attempts.write_text(json.dumps(row)+'\n' +
                                        ('{"answer":' if behavior in ('partial', 'timeout') else ''))
                    self.killed = False

                def wait(self, timeout=None):
                    if behavior == 'interrupt' and not self.killed:
                        raise KeyboardInterrupt
                    if behavior == 'timeout' and not self.killed:
                        raise subprocess.TimeoutExpired('controlled-test', timeout)
                    return 1 if behavior == 'service_nonzero' else 0

                def kill(self):
                    self.killed = True

                terminate = kill

            with patch.dict(os.environ, {'HEATROD_BASE_URL': 'http://test.invalid/v1'}), \
                 patch.object(compare_agents.subprocess, 'Popen', Process), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = compare_agents.main(['--cases', str(cases), '--variants', *variants,
                                            '--output-root', str(root/'runs')])
            doc = json.loads(next((root/'runs').glob('*/comparison.json')).read_text())
        return code, commands, doc

    def test_service_failure_stops_batch_for_both_old_and_new_agent_exit_codes(self):
        for behavior in ('service_zero', 'service_nonzero'):
            with self.subTest(behavior=behavior):
                code, commands, doc = self.run_batch(behavior)
                self.assertEqual(code, 1)
                self.assertEqual(len(commands), 1)
                self.assertFalse(doc['batch_complete'])
                self.assertEqual(doc['expected_runs'], 2)
                self.assertEqual(doc['results'][0]['status'], 'infrastructure_error')

    def test_artifact_corruption_stops_batch_after_preserving_results(self):
        for behavior in ('partial', 'bad_summary'):
            with self.subTest(behavior=behavior):
                code, commands, doc = self.run_batch(behavior)
                self.assertEqual(code, 1)
                self.assertEqual(len(commands), 1)
                self.assertEqual(doc['results'][0]['status'], 'artifact_error')
                self.assertTrue(doc['results'][0]['executed_original_solved'])
                self.assertFalse(doc['batch_complete'])

    def test_timeout_with_partial_log_retains_timeout_and_marks_incomplete(self):
        code, commands, doc = self.run_batch('timeout', variants=('improved',))
        self.assertEqual(code, 1)
        self.assertEqual(doc['results'][0]['status'], 'budget_timeout')
        self.assertEqual(doc['results'][0]['candidates'], 1)
        self.assertFalse(doc['batch_complete'])

    def test_interruption_is_preserved_and_stops_new_cases(self):
        code, commands, doc = self.run_batch('interrupt')
        self.assertEqual(code, 130)
        self.assertEqual(len(commands), 1)
        self.assertEqual(doc['results'][0]['status'], 'interrupted')
        self.assertFalse(doc['batch_complete'])

    def test_healthy_batch_still_completes(self):
        code, commands, doc = self.run_batch('healthy')
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 2)
        self.assertTrue(doc['batch_complete'])
        self.assertTrue(all(r['executed_original_solved'] for r in doc['results']))


class LegacyBenchmarkTests(unittest.TestCase):
    def run_benchmark(self, behavior):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                path = Path(command[command.index('--log')+1])
                row = dict(answer='0', round=0, trace=[])
                path.write_text(json.dumps(row)+'\n'+('{"answer":' if behavior == 'partial' else ''))
                return SimpleNamespace(returncode=1 if behavior == 'service' else 0)
            with patch.object(benchmark.subprocess, 'run', side_effect=run), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.main(['--output-root', tmp])
            doc = json.loads(next(Path(tmp).glob('*/comparison.json')).read_text())
        return code, calls, doc

    def test_legacy_entrypoint_also_stops_on_service_failure(self):
        code, calls, doc = self.run_benchmark('service')
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(doc['results'][0]['status'], 'infrastructure_error')
        self.assertFalse(doc['batch_complete'])

    def test_legacy_entrypoint_preserves_partial_evidence(self):
        code, calls, doc = self.run_benchmark('partial')
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(doc['results'][0]['status'], 'artifact_error')
        self.assertEqual(doc['results'][0]['candidates'], 1)
        self.assertFalse(doc['batch_complete'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
