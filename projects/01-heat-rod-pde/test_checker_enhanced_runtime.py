"""Controlled worker failures, distinct from live-model performance tests."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import checker_enhanced_runtime as runtime


PROBLEM = dict(L=1, k=1, f='sin(pi*x)', left='dirichlet', right='dirichlet',
               tol=.005, seed=0, name='worker-failure-regression')
ANSWER = 'sin(pi*x)*exp(-pi**2*t)'
PARTS = dict(equation=True, left_bc=True, right_bc=True, start_shape=True)


def original_record():
    return dict(reward=1.0, original_reward=1.0, parts=dict(PARTS), expr=ANSWER,
                start_error=0.0, feedback='Original passed.', scored=True)


def core_record(status='valid'):
    parts = dict(PARTS)
    if status != 'valid':
        parts['equation'] = False
    return dict(accepted=status == 'valid', status=status, parts=parts,
                errors={}, start_error=0.0, diagnostics=[], feedback='Core checked.',
                validation_complete=True, diagnostics_complete=False)


def emitted(kind, result):
    return json.dumps(dict(kind=kind, result=result))+'\n'


class EnhancedRuntimeFailureTests(unittest.TestCase):
    def assert_serializable(self, result):
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_timeout_before_core_is_inconclusive(self):
        for mode in ('verify', 'grade'):
            with self.subTest(mode=mode):
                output = emitted('original', original_record()) if mode == 'grade' else ''
                timeout = subprocess.TimeoutExpired('controlled-worker', 8, output=output)
                with patch.object(runtime.subprocess, 'run', side_effect=timeout):
                    result = runtime.run_check(mode, PROBLEM, ANSWER)
                validation = result if mode == 'verify' else result['validation']
                self.assertFalse(validation['accepted'])
                self.assertEqual(validation['status'], 'inconclusive')
                self.assertIn('worker_issue', validation)
                if mode == 'grade':
                    self.assertFalse(result['solved'])
                    self.assertEqual(result['original_reward'], 1.0)
                self.assert_serializable(result)

    def test_worker_exit_before_core_is_inconclusive(self):
        for mode in ('verify', 'grade'):
            with self.subTest(mode=mode):
                output = emitted('original', original_record()) if mode == 'grade' else ''
                process = subprocess.CompletedProcess([], 7, stdout=output, stderr='failure')
                with patch.object(runtime.subprocess, 'run', return_value=process):
                    result = runtime.run_check(mode, PROBLEM, ANSWER)
                validation = result if mode == 'verify' else result['validation']
                self.assertFalse(validation['accepted'])
                self.assertEqual(validation['status'], 'inconclusive')
                self.assertEqual(validation['worker_issue']['diagnostics'][0]['evidence']['returncode'], 7)
                self.assert_serializable(result)

    def test_timeout_after_core_preserves_each_outcome(self):
        for mode in ('verify', 'grade'):
            for status in ('valid', 'invalid', 'inconclusive'):
                with self.subTest(mode=mode, status=status):
                    output = emitted('validation_core', core_record(status))
                    if mode == 'grade':
                        output = emitted('original', original_record())+output
                    timeout = subprocess.TimeoutExpired('controlled-worker', 8, output=output.encode())
                    with patch.object(runtime.subprocess, 'run', side_effect=timeout):
                        result = runtime.run_check(mode, PROBLEM, ANSWER)
                    validation = result if mode == 'verify' else result['validation']
                    self.assertEqual(validation['status'], status)
                    self.assertEqual(validation['accepted'], status == 'valid')
                    self.assertTrue(validation['validation_complete'])
                    self.assertFalse(validation['diagnostics_complete'])
                    self.assertIn('worker_issue', validation)
                    if mode == 'grade':
                        self.assertEqual(result['solved'], status == 'valid')
                        self.assertEqual(result['original_reward'], 1.0)
                    self.assert_serializable(result)

    def test_incomplete_core_cannot_be_preserved_as_pass(self):
        core = core_record()
        core['validation_complete'] = False
        timeout = subprocess.TimeoutExpired('controlled-worker', 8,
                                            output=emitted('validation_core', core))
        with patch.object(runtime.subprocess, 'run', side_effect=timeout):
            result = runtime.run_check('verify', PROBLEM, ANSWER)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'inconclusive')
        self.assert_serializable(result)

    def test_original_score_survives_late_timeout(self):
        original = original_record()
        timeout = subprocess.TimeoutExpired('controlled-worker', 8,
                                            output=emitted('original', original))
        with patch.object(runtime.subprocess, 'run', side_effect=timeout):
            result = runtime.run_check('original', PROBLEM, ANSWER)
        self.assertEqual(result, original)
        self.assert_serializable(result)

    def test_completed_validation_has_no_worker_issue(self):
        core = core_record()
        core['diagnostics_complete'] = True
        process = subprocess.CompletedProcess([], 0,
            stdout=emitted('validation', core), stderr='')
        with patch.object(runtime.subprocess, 'run', return_value=process):
            result = runtime.run_check('verify', PROBLEM, ANSWER)
        self.assertTrue(result['accepted'])
        self.assertNotIn('worker_issue', result)
        self.assert_serializable(result)

    def test_real_subprocess_timeout_before_core(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory)/'worker.py'
            worker.write_text('import time\ntime.sleep(10)\n')
            for mode in ('verify', 'grade'):
                with self.subTest(mode=mode), patch.object(runtime, 'WORKER', worker), \
                     patch.object(runtime, 'WALL_SECONDS', .2):
                    result = runtime.run_check(mode, PROBLEM, ANSWER)
                    validation = result if mode == 'verify' else result['validation']
                    self.assertFalse(validation['accepted'])
                    self.assertEqual(validation['status'], 'inconclusive')
                    self.assert_serializable(result)

    def test_real_subprocess_exit_before_core(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory)/'worker.py'
            worker.write_text('raise SystemExit(7)\n')
            with patch.object(runtime, 'WORKER', worker):
                result = runtime.run_check('verify', PROBLEM, ANSWER)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['status'], 'inconclusive')
            self.assert_serializable(result)


if __name__ == '__main__':
    unittest.main()
