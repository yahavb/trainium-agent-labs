"""Installed SDK introspection and mocked repair; no live grade or model."""
import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import agent
from failure_selection import classify_failure
from nki_knowledge import (CATALOG, SDK_BUILD, compatible, ground_prompt,
                           installed_compatibility, local_token_counter,
                           referenced_operations, retrieve)

DMA = 'dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384'
SOURCE = 'import nki.isa as nisa\nimport nki.language as nl\ndef kernel(a):\n t=nl.ndarray((32,512),nl.float32,buffer=nl.sbuf)\n nisa.dma_copy(dst=t,src=a)\n return t'


class KnowledgeTests(unittest.TestCase):
    def test_all_cards_verified_against_installed_sdk(self):
        compatibility = installed_compatibility()
        self.assertEqual(compatibility['sdk_version'], SDK_BUILD)
        for card in CATALOG:
            with self.subTest(card=card.id):
                self.assertTrue(compatible(card, compatibility))
                self.assertIn('awsdocs-neuron', card.source_url)
                self.assertIn('/v2.32.0/', card.source_url)
                self.assertIn('not device tested', card.verification_status)

    def test_compatibility_fails_closed(self):
        card = CATALOG[0]
        good = installed_compatibility()
        for bad in [dict(good,sdk_version='0.5.0'), dict(good,sdk_version='0.6.0+unverified'), dict(good,hardware='trn3'),
                    dict(good,signatures={}), dict(good,signatures={card.api:('dst','src')})]:
            self.assertFalse(compatible(card,bad))
            result = retrieve(classify_failure(DMA),SOURCE,compatibility=bad)
            self.assertNotIn('dma', result['card_ids'])
            if bad['hardware'] != 'trn2' or not bad['sdk_version'].startswith('0.6.0'):
                self.assertEqual(result['card_ids'], [])

    def test_actual_failure_routes(self):
        cases = [
            (DMA, SOURCE, 'dma'),
            (DMA.replace('src=4, dst=16384','src=384, dst=512'),SOURCE,'dma'),
            ("nc_matmul() got an unexpected keyword argument 'transpose_moving'", 'import nki.isa as ni\ndef f():\n ni.nc_matmul(dst=c,stationary=a,moving=b,transpose_moving=True)', 'matmul'),
            ("dst must be in ['psum'], got sbuf",'import nki.isa as ni\nimport nki.language as nl\ndef f():\n c=nl.ndarray((32,32),nl.float32,buffer=nl.sbuf)\n ni.nc_matmul(dst=c,stationary=a,moving=b)','matmul'),
            ("module 'nki.isa' has no attribute 'multiply'",'import nki.isa as ni\ndef f():\n ni.multiply(dst=x,data1=a,data2=b)','binary'),
            ("module 'nki.isa' has no attribute 'scalar_mul'",'import nki.isa as ni\ndef f():\n ni.scalar_mul(dst=x,data=a,scalar=0.5)','scalar'),
            ('SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim)',SOURCE,'rank'),
        ]
        for feedback, source, first in cases:
            with self.subTest(feedback=feedback):
                result = retrieve(classify_failure(feedback),source)
                self.assertEqual(result['card_ids'][0],first)
                self.assertLessEqual(len(result['cards']),2)
                self.assertLessEqual(result['context_token_count'],500)
                self.assertTrue(result['source_calls'])

    def test_alias_and_from_import_resolution(self):
        apis,calls = referenced_operations('from nki.isa import nc_matmul as mm\ndef f():\n mm(dst=c,stationary=a,moving=b)')
        self.assertIn('nki.isa.nc_matmul',apis)
        self.assertEqual(calls[0]['line'],3)
        self.assertEqual(referenced_operations('def broken('),((),()))

    def test_unknown_and_missing_docs_fallback(self):
        for feedback in ('strange unknown failure',"module 'nki.isa' has no attribute 'warp_magic'"):
            self.assertEqual(retrieve(classify_failure(feedback),SOURCE)['card_ids'],[])

    def test_budget_enforcement_and_no_context_loss(self):
        result = retrieve(classify_failure(DMA),SOURCE,token_budget=1)
        self.assertEqual(result['text'],'')
        for budget in (50,100,300,500):
            result=retrieve(classify_failure(DMA),SOURCE,token_budget=budget)
            self.assertLessEqual(result['context_token_count'],budget)
        prompt = agent.repair_prompt(1,SOURCE,DMA)
        grounded,metadata=ground_prompt(prompt,classify_failure(DMA),SOURCE)
        self.assertTrue(grounded.startswith(prompt))
        self.assertIn('Failure category: DMA_SHAPE_MISMATCH',grounded)
        self.assertIn('equal total element counts',grounded)
        self.assertTrue(metadata['applied'])
        tight,metadata=ground_prompt(prompt,classify_failure(DMA),SOURCE,context=100,answer_budget=2500)
        self.assertEqual(tight,prompt)
        self.assertFalse(metadata['applied'])

    def test_fallback_budget_not_reported_as_endpoint_usage(self):
        with patch.dict('sys.modules',{'tokenizers':None}):
            local_token_counter.cache_clear()
            counter,method=local_token_counter('not-cached')
            self.assertEqual(method,'utf8_byte_ceiling')
            self.assertEqual(counter('hello'),5)
        local_token_counter.cache_clear()

    def test_grounded_solve_integration_and_standard_unchanged(self):
        for candidate_policy in ('standard','diverse'):
            options=SimpleNamespace(terse=0,rounds=2,samples=4,offline=False,give_up_after=4,
                                    candidate_policy=candidate_policy,selection_policy='diagnostic',repair_policy='grounded')
            records=io.StringIO()
            with patch.object(agent,'ask',return_value='```python\n'+SOURCE+'\n```') as ask_mock, \
                 patch.object(agent,'grade',return_value=(.3,{},DMA)), \
                 contextlib.redirect_stdout(io.StringIO()):
                agent.solve(options,1,records)
            rows=[json.loads(line) for line in records.getvalue().splitlines()]
            self.assertEqual(ask_mock.call_count,8)
            self.assertIsNone(rows[0]['grounding'])
            self.assertIn('dma',rows[4]['grounding']['card_ids'])
            self.assertIn(SOURCE,ask_mock.call_args_list[4].args[1])
            self.assertIn(DMA,ask_mock.call_args_list[4].args[1])
            self.assertEqual(rows[4]['repair_policy'],'grounded')


if __name__ == '__main__': unittest.main()
