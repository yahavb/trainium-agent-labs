import inspect
import json
from pathlib import Path
import tempfile
import unittest
import nki.language as nl
from shape_repair import plan_repair
from synthetic_nki.diagnostics import evaluate,identifies
from synthetic_nki.generate_pairs import build_pairs
from synthetic_nki.generate import templates
from synthetic_nki.verify import validate
from synthetic_nki.retrieval import DATA,load_train
import run_controlled
from types import SimpleNamespace

class DiagnosticTests(unittest.TestCase):
 def test_real_load_store_are_present(self):
  self.assertIn('src',inspect.signature(nl.load).parameters)
  self.assertIn('dst',inspect.signature(nl.store).parameters)
 def test_dma_before_matmul(self):
  source='import nki.language as nl\nimport nki.isa as ni\ndef f(a,b):\n t=nl.ndarray((2,4),nl.float32,buffer=nl.sbuf)\n d=nl.ndarray((1,2),nl.float32,buffer=nl.psum)\n ni.dma_copy(dst=t,src=a)\n ni.nc_matmul(dst=d,stationary=t,moving=b)\n return d'
  plan=plan_repair(source,'dma_copy requires src and dst to have the same number of elements, got src=6, dst=8',input_shapes={'a':(2,3),'b':(2,3)})
  self.assertIn('line 6',plan['root_causes'][0]);self.assertIn('src a shape (2, 3)',plan['root_causes'][0])
  self.assertNotIn('simulator reshapes',plan['guidance'])
 def test_explicit_rank_before_reduction(self):
  source='import nki.language as nl\ndef f(a):\n t=nl.ndarray((6,),nl.float32,buffer=nl.sbuf)\n r=nl.sum(t,axis=1)\n return r'
  plan=plan_repair(source,'SBUF and PSUM tensors must have at least 2 dimensions')
  self.assertIn('explicit allocation',plan['root_causes'][0]);self.assertNotIn('keepdims',plan['root_causes'][0])
 def test_argument_diagnosis_names_actual_keyword(self):
  source='import nki.isa as ni\ndef f():\n ni.tensor_scalar(dst=t,data=t,op0=op,operand=.25)'
  plan=plan_repair(source,"tensor_scalar() got an unexpected keyword argument 'operand'")
  self.assertIn('operand0',plan['guidance']);self.assertIn('line 3',plan['guidance'])
 def test_evaluation_uses_added_guidance_and_actual_failures(self):
  with tempfile.TemporaryDirectory() as temp:
   result=evaluate(Path('synthetic_nki/data_v2'),Path(temp)/'report.json')
  self.assertEqual(result['total'],24)
  self.assertTrue(all(r['actual_error'] for r in result['records']))
  self.assertFalse(identifies('reduction_rank','Every SBUF needs two dimensions'))
 def test_pair_dedup_and_family_holdout(self):
  rows=[json.loads(line) for split in ('train','heldout') for line in (DATA.parent/(split+'.jsonl')).read_text().splitlines()]
  self.assertEqual(len({r['broken_kernel'] for r in rows}),28)
  self.assertEqual(len({r['correct_kernel'] for r in rows}),16)
  self.assertEqual(len(load_train()),26)
 def test_all_alternate_mutations_observed_and_repairs_verified(self):
  specs={s['source']:s for s in templates()}
  rows=[json.loads(line) for split in ('train','heldout') for line in (DATA.parent/(split+'.jsonl')).read_text().splitlines()]
  for r in rows:
   if r['root_cause']!='unsupported_dma_api':continue
   self.assertFalse(validate(r['broken_kernel'],specs[r['correct_kernel']])['passed'])
   self.assertTrue(validate(r['corrected_kernel'],specs[r['correct_kernel']])['passed'])
 def test_matched_experiment_policies(self):
  options=SimpleNamespace(rounds=4,samples=4,repeat=1)
  for name in ('A_legacy_standard','B_targeted_standard','C_targeted_synthetic'):
   command=run_controlled.build_command(options,(name,'standard','diagnostic','grounded'),3,Path('/private')/name)
   self.assertEqual(command[command.index('--candidate-policy')+1],'standard')
   self.assertEqual('--feedback-policy' in command,name!='A_legacy_standard')
   self.assertEqual('--example-policy' in command,name=='C_targeted_synthetic')
if __name__=='__main__':unittest.main()

class LifetimeTests(unittest.TestCase):
 def test_shared_load_buffer_dependency_in_dma_guidance(self):
  source='import nki.language as nl\nimport nki.isa as ni\ndef f(a,b):\n t=nl.ndarray((2,4),nl.float32,buffer=nl.sbuf)\n d=nl.ndarray((3,4),nl.float32,buffer=nl.psum)\n ni.dma_copy(dst=t,src=a)\n ni.dma_copy(dst=t[:,0:4],src=b)\n ni.nc_matmul(dst=d,stationary=t,moving=t)\n return d'
  plan=plan_repair(source,'dma_copy requires src and dst to have the same number of elements, got src=6, dst=8',input_shapes={'a':(2,3),'b':(2,4)})
  self.assertIn('DMA lines 6 and 7',plan['guidance']);self.assertIn('distinct live SBUF',plan['guidance'])
  self.assertIn('If the computation requires',plan['guidance'])
 def test_separate_inputs_no_shared_buffer_claim(self):
  from shape_repair import input_lifetime_evidence,inspect_shapes
  source='import nki.isa as ni\ndef f(a,b):\n ni.dma_copy(dst=x,src=a)\n ni.dma_copy(dst=y,src=b)\n ni.nc_matmul(dst=z,stationary=x,moving=y)'
  self.assertFalse(input_lifetime_evidence(source,inspect_shapes(source)))
 def test_loop_lifetimes_left_unresolved(self):
  from shape_repair import input_lifetime_evidence,inspect_shapes
  source='import nki.isa as ni\ndef f(a,b):\n for i in range(2):\n  ni.dma_copy(dst=x,src=a)\n  ni.dma_copy(dst=x,src=b)\n  ni.nc_matmul(dst=z,stationary=x,moving=x)'
  self.assertFalse(input_lifetime_evidence(source,inspect_shapes(source)))
