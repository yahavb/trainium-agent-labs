"""Offline tests; SDK integration tests skip explicitly outside the Neuron host."""

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from curriculum import select_feedback
from portable import verify_sources,load_agent
from targeted_feedback import enrich_selected,evaluate,guidance

LIMITS={'pmax':128,'gemm_stationary_fmax':128,'gemm_moving_fmax':512,'psum_bank_fmax':512}
CASES=[{'K':128,'M':128,'N':512},{'K':256,'M':256,'N':1024},
       {'K':512,'M':128,'N':512},{'K':256,'M':512,'N':1024}]
BENCH=SimpleNamespace(LEVELS={4:{'shapes':CASES}},label=lambda c,l:' '.join(f'{k}={v}' for k,v in c.items()))


class FeedbackTests(unittest.TestCase):
    def select(self,indices,policy='curriculum',level=4):
        failures=[(BENCH.label(CASES[i],4),f'error {i}') for i in indices]
        passed=4-len(indices);label,message=failures[0]
        original=(.625,{'correct':False},f'{passed} of 4 shapes passed. On {label}: {message}')
        selected,details=select_feedback(original,level,failures,passed,BENCH,policy)
        self.assertEqual(original[:2],selected[:2]);self.assertIs(original[1],selected[1])
        return original,selected,details

    def test_easiest_K_only(self):
        _,_,details=self.select([1,2,3]);self.assertEqual(details['selected_shape'],CASES[2])

    def test_tie_order(self):
        old,new,_=self.select([1,3]);self.assertEqual(old,new)

    def test_zero_exceeded_dimensions_first(self):
        old,new,_=self.select([0,1,2,3]);self.assertEqual(old,new)

    def test_official_policy(self):
        old,new,_=self.select([1,2,3],'official');self.assertEqual(old,new)

    def test_other_levels(self):
        for level in (1,2,3):
            old,new,_=self.select([1,2,3],level=level);self.assertEqual(old,new)

    def apply(self,message,kind='stationary',enabled=True,events=True,level=4):
        parts={'correct':False};old=(.75,parts,'2 of 4 shapes passed. On shape: raised AssertionError: '+message)
        detail={'failures':[('shape','raised AssertionError: '+message)]}
        proof=[{'kind':kind,'message':message}] if events else []
        new,info=enrich_selected(old,detail,level,LIMITS,proof,enabled)
        self.assertEqual(old[:2],new[:2]);self.assertIs(old[1],new[1])
        return old,new,info

    def test_stationary_guidance(self):
        old,new,info=self.apply('Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128')
        self.assertTrue(info['guidance_applied']);self.assertEqual(new[2],old[2]+guidance(LIMITS))

    def test_moving_guidance(self):
        _,_,info=self.apply('Matmul moving free dimension 1024 exceeds max 512 for nc_version=nc_version.gen3','moving')
        self.assertTrue(info['guidance_applied'])

    def test_disabled_guidance(self):
        old,new,info=self.apply('Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128',enabled=False)
        self.assertEqual(old,new);self.assertFalse(info['guidance_applied'])

    def test_sdk_provenance_required(self):
        old,new,_=self.apply('Matmul stationary free dimension 256 exceeds gemm_stationary_fmax=128',events=False)
        self.assertEqual(old,new)

    def test_unrelated_error(self):
        old,new,_=self.apply('Matmul contraction dimension 512 exceeds pmax=128');self.assertEqual(old,new)

    def test_gen4_not_targeted(self):
        old,new,_=self.apply('Matmul moving free dimension 8192 exceeds max 4096 for nc_version=nc_version.gen4','moving')
        self.assertEqual(old,new)

    def test_disabled_evaluate_delegates_to_B(self):
        original=(.625,{},'unchanged');details={}
        with patch('targeted_feedback.base.evaluate',return_value=(original,details)) as checker:
            new,_=evaluate(None,'code',4,False)
        self.assertEqual(new,original);checker.assert_called_once_with(None,'code',4,'curriculum')

    def test_original_source_manifest(self):
        self.assertEqual(verify_sources(),{'checked_files':7,'unchanged':True})

    def test_guidance_has_no_complete_function(self):
        text=guidance(LIMITS);self.assertNotIn('def nki_matmul',text);self.assertIn('out[m0:m0+mt, n0:n0+nt]',text)

    def test_winning_source_matches_evidence(self):
        import json
        result=json.loads((HERE/'results/summary.json').read_text())
        source=(HERE/'examples/winning_kernel.py').read_text().strip()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),result['arms']['C']['best_code_sha256'])


@unittest.skipUnless(importlib.util.find_spec('nki') and importlib.util.find_spec('numpy'),'Requires the installed Neuron SDK and NumPy')
class SDKIntegrationTests(unittest.TestCase):
    def test_original_request_semantics_with_mocked_HTTP(self):
        import argparse,io,json,httpx
        from agent import TargetedInstrumentation
        canonical=load_agent()
        args=argparse.Namespace(max_tokens=2500,context=8192,model='Qwen/Qwen3-8B',
                                think=False,base='http://offline.invalid/v1')
        log=io.StringIO();instrument=TargetedInstrumentation(canonical,True,log,io.StringIO())
        instrument.local.context={'run':1,'level':4,'round':0,'sample':1}
        payload={'choices':[{'message':{'content':'```python\npass\n```'},'finish_reason':'stop'}],
                 'usage':{'prompt_tokens':4,'completion_tokens':3,'total_tokens':7}}
        def fake_post(url,**kwargs):return httpx.Response(200,json=payload)
        with patch.object(httpx,'post',side_effect=lambda url,**kw:instrument.post(fake_post,url,**kw)):
            canonical.ask(args,'Offline')
        record=json.loads(log.getvalue())
        self.assertEqual(record['request'],{'model':args.model,'messages':[{'role':'user','content':'Offline'}],
            'max_tokens':2500,'temperature':.6,'top_p':.95,'chat_template_kwargs':{'enable_thinking':False}})
        self.assertEqual(record['usage'],payload['usage'])

    def test_winning_kernel_off_on_canonical_contract(self):
        agent=load_agent();source=(HERE/'examples/winning_kernel.py').read_text()
        off,off_details=evaluate(agent,source,4,False)
        on,on_details=evaluate(agent,source,4,True)
        self.assertEqual(off,on);self.assertEqual(off[0],1.0);self.assertTrue(off[1]['correct'])
        self.assertEqual(off_details['shapes'],on_details['shapes']);self.assertFalse(on_details['guidance_applied'])


if __name__=='__main__':unittest.main()
