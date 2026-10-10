import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import agent
from shape_repair import inspect_shapes, plan_repair, constrained_prompt, shape_prompt
from experiment_metrics import enabled
import run_controlled
from pathlib import Path

SOURCE='''import nki.language as nl
import nki.isa as ni
def kernel(a):
 rows=2*2
 t=nl.ndarray((rows,8),nl.float32,buffer=nl.sbuf)
 ni.dma_copy(dst=t,src=a[0:2,0:2])
 out=nl.ndarray((4,8),nl.float32,buffer=nl.shared_hbm)
 ni.dma_copy(dst=out,src=t)
 return out
'''
DMA='dma_copy requires src and dst to have the same number of elements, got src=4, dst=32'
class ShapeTests(unittest.TestCase):
 def test_literal_inference_and_slices(self):
  evidence=inspect_shapes(SOURCE,{'a':(4,8)})
  self.assertEqual(evidence['allocations'][0]['derived_shape'],(4,8))
  self.assertIn('4 elements',evidence['issues'][0]);self.assertIn('32',evidence['issues'][0])
  self.assertEqual(evidence['transfers'][0]['line'],6)
 def test_unknown_not_executed(self):
  source=SOURCE.replace('2*2','evil()')
  evidence=inspect_shapes(source)
  self.assertIsNone(evidence['allocations'][0]['derived_shape'])
  self.assertEqual(evidence['issues'],[])
  self.assertIn('unresolved statically',plan_repair(source,DMA)['guidance'])
 def test_conditional_and_loop_values_unknown(self):
  for source in (SOURCE.replace(' rows=2*2',' if flag:\n  rows=4'),SOURCE.replace('rows=2*2','rows=index')):
   self.assertIsNone(inspect_shapes(source)['allocations'][0]['derived_shape'])
 def test_input_shape_unpack(self):
  source=SOURCE.replace('rows=2*2','rows,F=a.shape')
  self.assertEqual(inspect_shapes(source,{'a':(4,8)})['allocations'][0]['derived_shape'],(4,8))
 def test_rebound_input_unknown(self):
  source=SOURCE.replace(' rows=2*2',' a=unknown()\n rows,F=a.shape')
  self.assertIsNone(inspect_shapes(source,{'a':(4,8)})['allocations'][0]['derived_shape'])
 def test_no_budget_preserves_generation_prompt(self):
  prompt=agent.first_prompt(4)
  result,metadata=constrained_prompt(prompt,'matmul',context=100)
  self.assertEqual(result,prompt);self.assertFalse(metadata['applied'])
 def test_rank_error(self):
  self.assertIn('rank 1',inspect_shapes(SOURCE.replace('(rows,8)','(rows,)'))['issues'][0])
 def test_coordinated_dependencies(self):
  plan=plan_repair(SOURCE,DMA)
  self.assertEqual(plan['scope'],'coordinated_dataflow_repair')
  self.assertIn(6,plan['dependent_call_lines']);self.assertIn(8,plan['dependent_call_lines'])
  self.assertIn('cover every element',plan['guidance'])
 def test_api_buffer_and_redesign_scope(self):
  for feedback,scope in [("nc_matmul() got an unexpected keyword argument 'transpose_moving'",'localized_api_correction'),("dst must be in ['psum'], got sbuf",'buffer_placement_correction'),('NUMERICAL MISMATCH: incorrect result','algorithm_redesign')]:
   self.assertEqual(plan_repair(SOURCE,feedback,3)['scope'],scope)
 def test_generation_constraints_no_solution(self):
  prompt=agent.first_prompt(4);result,metadata=constrained_prompt(prompt,'matmul')
  self.assertNotIn('def copy_kernel',result);self.assertIn('stationary[K,M]',result)
  self.assertIn('Write every output element',result);self.assertIn('nki_matmul_tiled_',result)
  self.assertTrue(metadata['applied']);self.assertLessEqual(metadata['context_token_count'],650)
 def test_compatibility_fail_closed(self):
  with patch('shape_repair.installed_compatibility',return_value=dict(sdk_version='0.5',hardware='trn2',signatures={})):
   prompt,metadata=constrained_prompt('original','matmul')
  self.assertEqual(prompt,'original');self.assertFalse(metadata['applied'])
 def test_preserve_code_and_feedback(self):
  prompt=agent.repair_prompt(4,SOURCE,DMA)
  improved,metadata=shape_prompt(prompt,SOURCE,DMA)
  self.assertIn(SOURCE,improved);self.assertIn(DMA,improved)
  self.assertNotIn('Change exactly what the checker names',improved)
  self.assertLessEqual(metadata['context_token_count'],450)
 def test_mock_integration_eight_calls(self):
  options=SimpleNamespace(terse=0,rounds=2,samples=4,offline=False,give_up_after=4,candidate_policy='diverse',selection_policy='diagnostic',repair_policy='shape-aware',generation_policy='constrained')
  log=io.StringIO()
  with patch.object(agent,'ask',return_value='```python\n'+SOURCE+'\n```') as ask,patch.object(agent,'grade',return_value=(.3,{},DMA)),contextlib.redirect_stdout(io.StringIO()):agent.solve(options,4,log)
  rows=[json.loads(line) for line in log.getvalue().splitlines()]
  self.assertEqual(ask.call_count,8);self.assertTrue(rows[0]['generation_constraints']['applied'])
  self.assertEqual(rows[4]['shape_plan']['scope'],'coordinated_dataflow_repair')
  self.assertIn(SOURCE,rows[4]['prompt']);self.assertIn(DMA,rows[4]['prompt'])
 def test_defaults_and_generation_only_instrumentation(self):
  self.assertFalse(enabled(SimpleNamespace()))
  self.assertTrue(enabled(SimpleNamespace(generation_policy='constrained')))
 def test_controlled_arms_identical_budget(self):
  options=SimpleNamespace(rounds=4,samples=4,repeat=1)
  for arm in [('A_full','diverse','diagnostic','grounded'),('B_shape_aware','diverse','diagnostic','shape-aware')]:
   cmd=run_controlled.build_command(options,arm,4,Path('/private')/arm[0])
   self.assertEqual(cmd[cmd.index('--rounds')+1],'4')
   self.assertEqual('--generation-policy' in cmd,arm[0]=='B_shape_aware')
 def test_busy_and_unhealthy_guard(self):
  for busy,healthy in (([{'pid':1}],True),([],False)):
   with patch('sys.argv',['run_controlled.py','--correctness-pilot','--run']),patch.object(run_controlled,'baseline_processes',return_value=[]),patch.object(run_controlled,'evaluation_processes',return_value=busy),patch.object(run_controlled,'endpoint_healthy',return_value=healthy),patch.object(run_controlled.subprocess,'run') as run,contextlib.redirect_stdout(io.StringIO()):self.assertEqual(run_controlled.main(),2)
   run.assert_not_called()
if __name__=='__main__':unittest.main()

class ScalarNormalizationTests(unittest.TestCase):
 def test_tensor_division_guidance_does_not_introduce_matmul(self):
  from shape_repair import plan_repair
  code='import nki.language as nl\ndef f(x,p):\n r=nl.sum(x,axis=1,keepdims=True)\n mean=r/(p*p)\n return mean'
  error="TypeError: unsupported operand type(s) for /: 'NkiTensor' and 'int'"
  r=plan_repair(code,error)
  self.assertIn('line 4',r['guidance']);self.assertIn('tensor_scalar',r['guidance']);self.assertIn('op0=nl.multiply',r['guidance'])
  self.assertIn('does not require matmul',r['guidance']);self.assertEqual(r['original_feedback'],error)
 def test_host_reciprocal_not_misidentified(self):
  from shape_repair import plan_repair
  code='def f(p):\n reciprocal=1/(p*p)\n return reciprocal'
  error="TypeError: unsupported operand type(s) for /: 'NkiTensor' and 'int'"
  self.assertFalse(any('line 2:' in c for c in plan_repair(code,error)['root_causes']))

class NamespaceAndSliceTests(unittest.TestCase):
 def test_wrong_namespace_and_hbm_target(self):
  code='import nki.language as nl\ndef f(x):\n nl.tensor_scalar(dst=x,data=x,op0=nl.multiply,operand0=.5)'
  error="module 'nki.language' has no attribute 'tensor_scalar'"
  result=plan_repair(code,error)
  self.assertIn('nki.isa.tensor_scalar',result['guidance']);self.assertIn('HBM destination',result['guidance']);self.assertIn('line 3',result['guidance'])
 def test_onchip_partition_index_preserved(self):
  code='import nki.language as nl\nimport nki.isa as ni\ndef f(a):\n t=nl.ndarray((3,4,5),a.dtype,buffer=nl.sbuf)\n out=nl.ndarray((1,),a.dtype,buffer=nl.shared_hbm)\n ni.dma_copy(dst=out,src=t[0,1,2])'
  r=inspect_shapes(code,{'a':(3,4,5)})
  self.assertEqual(r['transfers'][0]['operands']['src']['derived_shape'],(1,1))
 def test_hbm_indices_remove_axes(self):
  code='import nki.isa as ni\ndef f(a):\n ni.dma_copy(dst=a[0,1,2],src=a[0,1,2])'
  r=inspect_shapes(code,{'a':(3,4,5)})
  self.assertEqual(r['transfers'][0]['operands']['src']['derived_shape'],(1,))

class OpcodeNamespaceTests(unittest.TestCase):
 def test_attribute_opcode_not_call(self):
  code='import nki.language as nl\nimport nki.isa as ni\ndef f(x):\n ni.tensor_scalar(dst=x,data=x,op0=ni.multiply,operand0=.5)'
  r=plan_repair(code,"module 'nki.isa' has no attribute 'multiply'")
  self.assertIn('nki.language.multiply',r['guidance']);self.assertIn('line 4',r['guidance']);self.assertIn('SBUF result',r['guidance'])

class SliceBoundConservatismTests(unittest.TestCase):
 def test_nki_slices_do_not_use_numpy_clipping(self):
  code='import nki.language as nl\nimport nki.isa as ni\ndef f(a):\n t=nl.ndarray((4,8),a.dtype,buffer=nl.sbuf)\n ni.dma_copy(dst=t,src=a[0:128,:])'
  r=inspect_shapes(code,{'a':(4,8)})
  self.assertIsNone(r['transfers'][0]['operands']['src']['derived_shape']);self.assertEqual(r['issues'],[])

class OperationSpecificNumericalFeedbackTests(unittest.TestCase):
 def test_pool_shape_error_not_given_matmul_advice(self):
  code='import nki.language as nl\ndef f(x,p):\n out=nl.ndarray((3,2),x.dtype,buffer=nl.shared_hbm)'
  r=plan_repair(code,'WRONG SHAPE: returned (), reference is (3,2)')
  self.assertIn('returned tensor',r['guidance']);self.assertNotIn('matmul',r['guidance']);self.assertNotIn('contraction',r['guidance'])
