import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidate_store import CandidateStore, candidate_hash
from neuron_solver import solve, specification, evaluate_candidate
from repair_engine import diagnose, repair_prompt
from evaluate import aggregate
import solver_checker


def result(reward=.3, solved=False, passed=()):
    return dict(reward=reward, solved=solved, passed_cases=list(passed), simulator_calls=1,
                cases=[dict(label='large', passed=solved, feedback='partition dimension 256 exceeds maximum 128')])

class SolverTests(unittest.TestCase):
    def test_hash_ignores_formatting(self):
        self.assertEqual(candidate_hash('x=1 # comment'), candidate_hash('x = 1\n'))
        self.assertNotEqual(candidate_hash('x=1'), candidate_hash('x=2'))

    def test_diagnostics(self):
        messages = {'dimensions':'must have at least two dimensions', 'partition':'partition dimension 256 exceeds maximum 128', 'dma_shape':'dma_copy requires same number of elements src=64 dst=128', 'bounds':'Out-of-bound access', 'api':"module has no attribute 'dot'", 'matmul_layout':'stationary operand layout wrong', 'memory':'dst must be in psum', 'reshape':'cannot reshape array', 'output_shape':'WRONG SHAPE', 'missing_writes':'uninitialized output', 'traffic':'too much HBM traffic', 'numerical':'mismatch at index (2, 3)'}
        for category, feedback in messages.items():
            with self.subTest(category=category):
                ds=diagnose('t = nl.ndarray((256, 32), buffer=nl.sbuf)', {'cases':[dict(label='x',passed=False,feedback=feedback)]})
                self.assertEqual(ds[0]['category'], category)
                self.assertIn('hypothesis', ds[0]['certainty'])
                self.assertTrue(ds[0]['validation'])

    def test_prompt_reference_boundary(self):
        import inspect
        with patch.object(inspect, 'getsource', side_effect=AssertionError('reference source read')):
            for lv in range(1,9):
                text=specification(lv)
                self.assertNotIn('ref_',text)
                self.assertIn('Official shapes',text)
        prompt=repair_prompt('math spec', 'x=1', result(), [{'hash':'abc'}], True)
        self.assertIn('smallest relevant',prompt)
        self.assertIn('fallback',prompt)
        self.assertIn('partition',prompt)

    def test_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=CandidateStore(tmp)
            store.record('x=1', result(.6))
            store.record('x=2', result(.3))
            self.assertEqual((Path(tmp)/'best.py').read_text(),'x=1')
            store.record('x=3',result(1.,True))
            self.assertEqual((Path(tmp)/'solved.py').read_text(),'x=3')
            self.assertEqual(len((Path(tmp)/'attempts.jsonl').read_text().splitlines()),3)

    def test_loop_duplicates_fallback_stop(self):
        prompts=[]
        answers=iter(['x=1','x = 1','x=2'])
        def generate(prompt,count):
            prompts.append(prompt)
            return ["```python\n" + next(answers) + "\n```"]
        calls=[]
        def checker(source,level):
            calls.append(source)
            return result(1.,True) if source=='x=2' else result()
        a=SimpleNamespace(rounds=8,samples=1,context=100000,max_tokens=2500)
        with tempfile.TemporaryDirectory() as tmp:
            summary=solve(a,4,tmp,generate,checker)
            self.assertTrue(summary['solved'])
            self.assertEqual(summary['rounds'],3)
            self.assertEqual(len(calls),2)
            self.assertEqual(summary['duplicates'],1)
            self.assertIn('fallback',prompts[2])
            self.assertTrue((Path(tmp)/'result.json').exists())

    def test_regression_does_not_replace_best(self):
        replies=iter(['x=1','x=2','x=3'])
        results=iter([result(.6,passed=['small']),result(.7,passed=['large']),result(.2)])
        a=SimpleNamespace(rounds=3,samples=1,context=100000,max_tokens=2500)
        with tempfile.TemporaryDirectory() as tmp:
            summary=solve(a,4,tmp,lambda p,n:["```python\n" + next(replies) + "\n```"],lambda s,l:next(results))
            self.assertEqual(summary['best_reward'],.6)
            logs=[json.loads(x) for x in (Path(tmp)/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual(logs[1]['regression'],['small'])
            self.assertEqual((Path(tmp)/'best.py').read_text(), 'x=1')

    def test_timeout(self):
        import subprocess
        with patch('neuron_solver.subprocess.run',side_effect=subprocess.TimeoutExpired('checker',1)):
            r=evaluate_candidate('x=1',4,1)
            self.assertFalse(r['solved'])
            self.assertIn('timeout',r['cases'][0]['feedback'])

    def test_real_checker_rejects_invalid_source(self):
        r=evaluate_candidate('def broken(',1)
        self.assertFalse(r['solved'])
        self.assertIn('SyntaxError',r['cases'][0]['feedback'])

    def test_checker_all_shapes_and_traffic_gate(self):
        import numpy as np
        with patch.object(solver_checker.bench,'describe_mismatch',return_value=None), patch.object(solver_checker.bench,'check_rules',return_value=[]), patch.object(solver_checker.bench,'load_kernel',return_value=lambda:None), patch.object(solver_checker.bench,'simulate_and_count',side_effect=lambda k,args:(np.zeros((args[0].shape[1], args[1].shape[1]), dtype=np.float32),dict(bytes=10**12,warnings=[]))):
            r=solver_checker.grade('x=1',5)
            self.assertFalse(r['solved'])
            self.assertEqual(len(r['cases']),4)
            self.assertEqual(r['simulator_calls'],4)
            self.assertIn('TRAFFIC',r['cases'][0]['feedback'])

    def test_grader_import_boundary(self):
        r=solver_checker.grade('import nkibench\nx=1',1)
        self.assertFalse(r['solved'])
        self.assertIn('grading helpers',r['cases'][0]['feedback'])

    def test_probes_preserve_attention_signature(self):
        import numpy as np
        signatures=[]
        def simulation(kernel,args):
            signatures.append([x.shape for x in args])
            return solver_checker.bench.ref_attention(*args), dict(bytes=0,warnings=[])
        with patch.object(solver_checker.bench,'check_rules',return_value=[]), patch.object(solver_checker.bench,'load_kernel',return_value=None), patch.object(solver_checker.bench,'simulate_and_count',side_effect=simulation):
            r=solver_checker.grade('x=1',8,probes=True)
        self.assertTrue(r['solved'])
        self.assertEqual(len(r['probe_results']),3)
        self.assertTrue(all(len(s)==3 for s in signatures))
        self.assertEqual(r['simulator_calls'],6)

    def test_identical_trials_flag(self):
        t=dict(method='solver',level=1,solved=True,mean_reward=1.,best_reward=1.,rounds_until_success=1,model_calls=1,simulator_calls=4,error_categories={},wall_seconds=2.,response_hashes=['same'])
        report=aggregate([t,t])['solver/level-1']
        self.assertTrue(report['identical_trial_outputs'])
        self.assertEqual(report['solve_rate'],1.)

if __name__=='__main__':
    unittest.main()
