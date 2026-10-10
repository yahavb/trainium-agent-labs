"""Algorithm regressions with controlled responses; these are not live benchmarks."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import sympy as sp
import algorithm_agent as aa
import algorithm_tools as at
import algorithm_benchmark as ab
import level0_heatrod as level0
import level1_heatrod as level1
import pdecheck
import tool_calc


def settings(features=''):
    return SimpleNamespace(features=features,deadline=100,timeout=10,samples=1,workers=1,rounds=2,
                           tool_steps=1,max_tokens=512,model='test',base='http://unused/v1',seed=0)


class AlgorithmTests(unittest.TestCase):
    def test_cache_single_owner_and_scope(self):
        def evaluator(source):
            time.sleep(.02)
            return dict(status='ok',result='2')
        with patch.object(at,'evaluate_with_timeout',side_effect=lambda s,t:evaluator(s)) as compute:
            cache=at.CalculatorCache()
            with ThreadPoolExecutor(2) as pool:
                results=list(pool.map(cache.compute,['Integral(x,(x,0,2))',' (Integral( x, (x,0,2))) ']))
            self.assertEqual(compute.call_count,1)
            self.assertEqual(sum(r['cached'] for r in results),1)
            at.CalculatorCache().compute('Integral(x,(x,0,2))')
            self.assertEqual(compute.call_count,2)

    def test_numeric_precision_is_never_merged(self):
        self.assertNotEqual(at.canonical_expression('0.1000000000000000000000001'),
                            at.canonical_expression('0.1000000000000000000000002'))
        self.assertNotEqual(at.canonical_expression('0.1'),at.canonical_expression('Rational(1,10)'))

    def test_calculator_retains_original_semantics(self):
        for source in ('Integral(x,(x,0,2))','0.1+0.2','(1,2)'):
            self.assertEqual(at.evaluate_with_timeout(source,10)['result'],tool_calc.compute(source))

    def test_calculator_deadline_does_not_execute(self):
        evaluator=lambda source: self.fail('Expired calculation executed')
        cache=at.CalculatorCache(evaluator=evaluator)
        self.assertEqual(cache.compute('x',time.monotonic()-1)['status'],'budget_timeout')

    def test_diagnostics_never_delete_index_or_guess_sum_bounds(self):
        for answer in ('u(x,t)=n*sin(n*pi*x)','u(x,t)=Sum(sin(n*pi*x),(n,1,3))',
                       'u(x,t)=Integral(x,(x,0,1))','u(x,t)=exp('):
            result=at.assess_final(answer,'length')
            self.assertFalse(result['valid'])
            self.assertEqual(result['answer'],answer)

    def test_supported_original_syntax_stays_supported(self):
        answer='exp(-9*pi**2*t) sin(3*pi*x) + Rational(0,1)'
        self.assertTrue(at.assess_final(answer)['valid'])

    def test_public_problem_does_not_read_hidden_helpers(self):
        class Guard(dict):
            def __getitem__(self,key):
                if key in ('exact','basis','lam','series_answer'):
                    raise AssertionError('Hidden helper read')
                return super().__getitem__(key)
        problem=aa.public_problem(Guard(level1.make(1)))
        self.assertIn('already a finite collection',aa.workflow(problem,0))

    def test_projection_feedback_requires_model_requested_normalization(self):
        p=aa.public_problem(level1.make(3,2))
        correct='8/pi**3'
        records=[dict(status='ok',expression='2*Integral((x-x**2)*sin(pi*x),(x,0,1))',result=correct)]
        self.assertIn(correct,at.coefficient_feedback(p,'4/pi**3*sin(pi*x)',records))
        records[0]['expression']='Integral((x-x**2)*sin(pi*x),(x,0,1))'
        self.assertEqual(at.coefficient_feedback(p,'4/pi**3*sin(pi*x)',records),'')

    def test_repair_consumes_only_reserved_request(self):
        a=settings('final')
        ctx=aa.Context(a)
        problem=aa.public_problem(level0.make(1))
        exact=sp.sstr(level0.make(1)['exact'])
        with patch.object(ctx,'ask',side_effect=[('u(x,t)=n*sin(n*x)', 'length'),(exact,'stop')]) as ask:
            result=aa.run_attempt(problem,a,ctx,'solve',0)
        self.assertEqual(ask.call_count,2)
        self.assertEqual(result['answer'],exact)
        self.assertEqual(sum(t['type']=='targeted_repair' for t in result['trace']),1)

    def test_final_diagnostic_never_overrides_official_acceptance(self):
        p=level0.make(1);a=settings('final')
        answer=sp.sstr(p['exact'])
        with patch.object(aa.Context,'ask',return_value=(answer,'stop')) as ask, \
             patch.object(at,'assess_final',return_value=dict(valid=False,issues=[dict(message='advisory')])):
            result=aa.solve(p,a,io.StringIO(),'parity')
        self.assertEqual(result['status'],'solved')
        ask.assert_called_once()

    def test_request_cap_is_global_and_token_cap_respected(self):
        a=settings('adaptive');a.samples=a.rounds=1;a.tool_steps=0
        ctx=aa.Context(a)
        reply=httpx.Response(200,json=dict(choices=[dict(message=dict(content='0'),finish_reason='stop')],
                                         usage=dict(prompt_tokens=10,completion_tokens=1)),
                             request=httpx.Request('POST',a.base))
        with patch.object(httpx,'post',return_value=reply) as post:
            ctx.ask([],[],'final')
            with self.assertRaises(aa.BudgetError):
                ctx.ask([],[],'final')
        self.assertLessEqual(post.call_args.kwargs['json']['max_tokens'],512)
        self.assertNotIn('seed',post.call_args.kwargs['json'])

    def test_malformed_answer_is_logged_without_fabricated_score(self):
        a=settings()
        response=dict(answer='[1,2]',trace=[],structure=None,tool_calls=0,tool_executions=0,cache_hits=0,error=None)
        log=io.StringIO()
        with patch.object(aa,'run_attempt',side_effect=lambda *args:dict(response)):
            result=aa.solve(level0.make(1),a,log,'bad')
        self.assertEqual(result['status'],'evaluation_error')
        self.assertIsNone(json.loads(log.getvalue().splitlines()[0])['grade']['reward'])

    def test_summary_preserves_timeout_and_unknown_token_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'telemetry.jsonl').write_text(json.dumps(dict(type='request_start',request_id='a',request=dict(max_tokens=512)))+'\n')
            row=dict(answer=sp.sstr(level0.make(1)['exact']),round=0)
            (p/'attempts.jsonl').write_text(json.dumps(row)+'\n')
            result=ab.summarize_case(p,(0,1,0),'baseline','budget_timeout')
        self.assertEqual(result['status'],'budget_timeout')
        self.assertTrue(result['candidate_original_solved'])
        self.assertEqual(result['requests'],1)
        self.assertEqual(result['unfinished_requests'],1)
        self.assertIsNone(result['completion_tokens'])
        self.assertEqual(result['additional_validation_status'],'not_run')

    def test_summary_flags_corrupt_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'attempts.jsonl').write_text(json.dumps(dict(answer='0',round=0))+'\n{"answer":')
            result=ab.summarize_case(p,(0,1,0),'baseline','unsolved')
        self.assertEqual(result['status'],'artifact_error')
        self.assertEqual(result['attempts_logged'],1)
        self.assertEqual(len(result['log_errors']),1)


if __name__=='__main__':
    unittest.main()
