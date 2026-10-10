import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import agent
from failure_selection import classify_failure,code_fingerprint
from synthetic_nki.generate import templates,build
from synthetic_nki.verify import validate
from synthetic_nki.mutations import inject
from synthetic_nki.retrieval import load_train,retrieve_examples,example_prompt,DATA
from repair_history import RepairHistory

class SyntheticTests(unittest.TestCase):
 def test_deterministic_and_structural_uniqueness(self):
  first=templates();self.assertEqual(first,templates())
  self.assertEqual(len({code_fingerprint(s['source']) for s in first}),12)
  self.assertNotEqual(first[0]['seed'],templates(42)[0]['seed'])
 def test_all_clean_mutations_and_repairs(self):
  for spec in templates():
   with self.subTest(kind=spec['kind']):
    self.assertTrue(validate(spec['source'],spec)['passed'])
    source,category,cause=inject(spec['source'],spec);result=validate(source,spec)
    self.assertFalse(result['passed']);self.assertEqual(result['failure_category'],category)
    self.assertTrue(result['error']);self.assertTrue(validate(spec['source'],spec)['passed'])
 def test_mutation_that_does_not_fail_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   with patch('synthetic_nki.generate.inject',side_effect=lambda source,spec:(source,'UNKNOWN','fake')):
    summary=build(Path(temp)/'data')
   self.assertEqual(summary['verified_repair_pairs'],0);self.assertEqual(len(summary['rejected']),12)
 def test_dataset_serialization_and_heldout_isolation(self):
  train=load_train();held=[json.loads(line) for line in DATA.with_name('heldout.jsonl').read_text().splitlines()]
  self.assertEqual(len(train),22);self.assertEqual(len(held),2)
  self.assertFalse(set(r['operation_family'] for r in train)&set(r['operation_family'] for r in held))
  self.assertFalse(load_train(DATA.with_name('heldout.jsonl')))
  self.assertFalse(set(r['structural_hash'] for r in train)&set(r['structural_hash'] for r in held))
 def test_duplicate_and_incompatible_records_rejected(self):
  record=load_train()[0]
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'train.jsonl';bad=dict(record,sdk_version='unknown')
   path.write_text('\n'.join(json.dumps(r) for r in [record,record,bad]))
   self.assertEqual(len(load_train(path)),1)
 def test_retrieval_cause_and_budget(self):
  feedback='cannot reshape array of size 6 into shape (1,2)';source='import nki.isa as ni\ndef f():\n ni.nc_matmul(dst=d,stationary=a,moving=b)'
  result=retrieve_examples('INVALID_TENSOR_DIMENSIONS',feedback,source)
  self.assertEqual(result['root_cause'],'matmul_destination');self.assertTrue(result['example_ids'])
  self.assertNotIn('def kernel',result['text']);self.assertLessEqual(result['context_token_count'],350)
  self.assertFalse(retrieve_examples('INVALID_TENSOR_DIMENSIONS',feedback,source,token_budget=1)['text'])
  self.assertFalse(retrieve_examples('UNKNOWN','unknown',source)['text'])
 def test_preserve_source_feedback_and_context(self):
  spec=templates()[0];broken,_,_=inject(spec['source'],spec);feedback='dma_copy requires src and dst to have the same number of elements, got src=6, dst=8'
  prompt=agent.repair_prompt(1,broken,feedback)
  result,metadata=example_prompt(prompt,classify_failure(feedback),broken)
  self.assertTrue(result.startswith(prompt));self.assertTrue(metadata['example_ids'])
  limited,meta=example_prompt(prompt,classify_failure(feedback),broken,context=100)
  self.assertEqual(limited,prompt)
 def test_history_escalates_and_preserves_best(self):
  history=RepairHistory();history.observe('def f(): pass','dma_copy shape mismatch',.625,1)
  history.observe('def f(): return 1','dma_copy shape mismatch',.3,0)
  self.assertIn('coordinated changes',history.guidance());self.assertEqual(history.best['reward'],.625)
  self.assertTrue(history.entries[-1]['source_changed'])
 def test_composable_policies_and_exact_call_budget(self):
  for feedback_policy,example_policy in [('legacy','off'),('targeted','off'),('legacy','synthetic'),('targeted','synthetic')]:
   options=SimpleNamespace(terse=0,rounds=2,samples=4,offline=False,give_up_after=4,candidate_policy='diverse',selection_policy='diagnostic',repair_policy='grounded',feedback_policy=feedback_policy,example_policy=example_policy,adaptive_repair=feedback_policy=='targeted' and example_policy=='synthetic')
   spec=templates()[0];broken,_,_=inject(spec['source'],spec);reply='```python\n'+broken+'\n```';log=io.StringIO()
   with patch.object(agent,'ask',return_value=reply) as ask,patch.object(agent,'grade',return_value=(.3,{},'dma_copy requires src and dst to have the same number of elements, got src=6, dst=8')),contextlib.redirect_stdout(io.StringIO()):agent.solve(options,1,log)
   self.assertEqual(ask.call_count,8)
   rows=[json.loads(line) for line in log.getvalue().splitlines()]
   if example_policy=='synthetic':self.assertTrue(rows[4]['synthetic_context']['example_ids'])
   if feedback_policy=='targeted':self.assertIsNotNone(rows[4]['shape_plan'])
if __name__=='__main__':unittest.main()
