import os
from pathlib import Path
import tempfile
import unittest
import sympy as sp
from symbolic_shapes import analyze,equality,within_limit,tiling,broadcasting,repair_prompt

K,M,N=sp.symbols('K M N',integer=True,positive=True)
PREFIX='import nki.language as nl\nimport nki.isa as ni\ndef f(a,b):\n'
def kernel(body):return PREFIX+'\n'.join(' '+line for line in body.splitlines())
class SymbolicTests(unittest.TestCase):
 def test_symbolic_equality(self):self.assertEqual(equality(K*(M+N),K*M+K*N),'PROVEN_EQUAL')
 def test_proven_symbolic_inequality(self):self.assertEqual(equality(K+1,K),'PROVEN_MISMATCH')
 def test_unknown_symbolic_difference(self):self.assertEqual(equality(M,N),'UNKNOWN')
 def test_concrete_dma_mismatch(self):
  result=analyze(kernel('t=nl.ndarray((128,128),nl.float32,buffer=nl.sbuf)\nni.dma_copy(dst=t,src=a)'),{'a':(4,)})
  self.assertEqual(result['operations'][0]['status'],'PROVEN_MISMATCH');self.assertEqual(result['operations'][0]['dst_elements'],'16384')
 def test_symbolic_dma_equal(self):
  result=analyze(kernel('P,F=a.shape\nt=nl.ndarray((P,F),a.dtype,buffer=nl.sbuf)\nni.dma_copy(dst=t,src=a)'),{'a':(K,M)})
  self.assertEqual(result['operations'][0]['status'],'PROVEN_EQUAL');self.assertEqual(result['status'],'UNKNOWN')
 def test_equal_elements_do_not_prove_layout(self):
  result=analyze(kernel('t=nl.ndarray((6,),a.dtype,buffer=nl.sbuf)\nni.dma_copy(dst=t,src=a)'),{'a':(2,3)})
  self.assertEqual(result['operations'][0]['status'],'PROVEN_EQUAL');self.assertEqual(result['violations'][0]['kind'],'onchip_rank')
 def matmul(self,dst='(64,512)',left='(128,64)',right='(128,512)',buffer='nl.psum'):
  return kernel(f's=nl.ndarray({left},nl.float32,buffer=nl.sbuf)\nm=nl.ndarray({right},nl.float32,buffer=nl.sbuf)\nd=nl.ndarray({dst},nl.float32,buffer={buffer})\nni.nc_matmul(dst=d,stationary=s,moving=m)')
 def test_matmul_dst_dimensions(self):
  result=analyze(self.matmul('(1,64)'));self.assertTrue(any(v['kind']=='matmul_dimensions' for v in result['violations']))
 def test_same_elements_wrong_axes(self):
  result=analyze(self.matmul('(128,256)'));self.assertIn('matmul_dimensions',[v['kind'] for v in result['violations']])
 def test_matmul_contraction_mismatch(self):
  result=analyze(self.matmul(right='(64,512)'));self.assertIn('PROVEN_MISMATCH',result['operations'][0]['axis_equalities'])
 def test_correct_matmul(self):
  result=analyze(self.matmul());self.assertFalse(result['violations']);self.assertEqual(result['status'],'NO_STATIC_VIOLATION')
 def test_buffer_requirement(self):self.assertIn('matmul_buffer',[v['kind'] for v in analyze(self.matmul(buffer='nl.sbuf'))['violations']])
 def test_partition_limit(self):self.assertEqual(within_limit(sp.Integer(256),128),'PROVEN_VIOLATION');self.assertEqual(within_limit(K,128),'UNKNOWN')
 def test_psum_free_limit(self):self.assertIn('psum_free_limit',[v['kind'] for v in analyze(self.matmul('(64,1024)'))['violations']])
 def test_psum_bank_limit_not_universal_allocation_limit(self):
  result=analyze(kernel('p=nl.ndarray((2,1024),nl.float32,buffer=nl.psum)'))
  self.assertNotIn('psum_free_limit',[v['kind'] for v in result['violations']])
 def test_three_axis_tiling(self):
  self.assertEqual([tiling(sp.Integer(d),sp.Integer(t))['tile_count'] for d,t in [(256,128),(1024,512),(256,128)]],['2','2','2'])
 def test_ragged_tail(self):self.assertEqual(tiling(sp.Integer(257),sp.Integer(128))['final_extent'],'1')
 def test_incorrect_count_coverage(self):self.assertEqual(tiling(sp.Integer(256),sp.Integer(128),sp.Integer(1))['output_coverage'],'PROVEN_MISMATCH')
 def test_reduction_1d(self):
  result=analyze(kernel('t=nl.ndarray((2,3),nl.float32,buffer=nl.sbuf)\nr=nl.sum(t,axis=1)'))
  self.assertIn('reduction_rank',[v['kind'] for v in result['violations']])
 def test_valid_reduction_without_keepdims(self):
  result=analyze(kernel('t=nl.ndarray((2,3,4),nl.float32,buffer=nl.sbuf)\nr=nl.sum(t,axis=2)'));self.assertFalse(result['violations'])
 def test_unsupported_call_is_unknown(self):self.assertEqual(analyze(kernel('t=nl.ndarray(make_shape(),nl.float32,buffer=nl.sbuf)'))['status'],'UNKNOWN')
 def test_no_source_execution(self):
  with tempfile.TemporaryDirectory() as temp:
   target=Path(temp)/'bad';source=kernel(f"x=__import__('pathlib').Path({str(target)!r}).touch()")
   analyze(source);self.assertFalse(target.exists())
 def test_complexity_limits(self):self.assertEqual(analyze('x=1\n'*3000)['status'],'UNKNOWN')
 def test_slices_and_loop_offsets(self):
  result=analyze(kernel('out=nl.ndarray(a.shape,a.dtype,buffer=nl.shared_hbm)\nfor i in range(2):\n t=nl.ndarray((128,3),a.dtype,buffer=nl.sbuf)\n ni.dma_copy(dst=t,src=a[i*128:(i+1)*128,:])'),{'a':(256,3)})
  self.assertEqual(result['loops'][0]['upper_exclusive'],'2');self.assertNotEqual(result['operations'][0]['status'],'PROVEN_MISMATCH')
 def test_ceiling_arithmetic(self):
  result=analyze(kernel('P,F=a.shape\nT=(P+127)//128\nt=nl.ndarray((T,F),a.dtype,buffer=nl.sbuf)'),{'a':(257,3)})
  self.assertEqual(result['allocations'][0]['shape'],['3','3'])
 def test_broadcasting(self):self.assertEqual(broadcasting((sp.Integer(2),sp.Integer(1)),(sp.Integer(2),sp.Integer(3))),'PROVEN_EQUAL')
 def test_prompt_preserves_original_and_budget(self):
  source=self.matmul('(1,64)');prompt='original source and error'
  text,result=repair_prompt(prompt,source,'cannot reshape array of size 32768 into shape (1,64)')
  self.assertTrue(text.startswith(prompt));self.assertIn('64, 512',text);self.assertLessEqual(result['context_token_count'],120)
  text,result=repair_prompt(prompt,source,'cannot reshape array of size 32768 into shape (1,64)',context=1)
  self.assertEqual(text,prompt)
if __name__=='__main__':unittest.main()

class SymbolicIntegrationTests(unittest.TestCase):
 def test_mocked_agent_exact_request_count_and_logging(self):
  import agent,contextlib,io,json
  from unittest.mock import patch
  from types import SimpleNamespace
  options=SimpleNamespace(terse=0,rounds=2,samples=4,offline=False,give_up_after=4,candidate_policy='diverse',selection_policy='diagnostic',repair_policy='standard',feedback_policy='targeted',example_policy='off',shape_analysis='sympy')
  source=kernel('t=nl.ndarray((128,512),nl.float32,buffer=nl.sbuf)\nni.dma_copy(dst=t,src=a)\nreturn t').replace('def f(a,b)','def nki_matmul_basic_(a,b)')
  feedback='0 of 1 shapes passed. On K=128 M=64 N=512: dma_copy requires src and dst to have the same number of elements, got src=8192, dst=65536'
  log=io.StringIO()
  with patch.object(agent,'ask',return_value='```python\n'+source+'\n```') as ask,patch.object(agent,'grade',return_value=(.3,{},feedback)),contextlib.redirect_stdout(io.StringIO()):agent.solve(options,3,log)
  self.assertEqual(ask.call_count,8)
  rows=[json.loads(line) for line in log.getvalue().splitlines()]
  self.assertIsNone(rows[0]['symbolic_analysis']);self.assertTrue(rows[4]['symbolic_analysis']['applied'])
 def test_symbolic_generated_pattern_actually_verified(self):
  from synthetic_nki.symbolic_generate import build
  with tempfile.TemporaryDirectory() as temp:
   r=build(Path(temp)/'data')
  self.assertTrue(r['accepted']);self.assertFalse(r['device_verified'])
 def test_control_treatment_change_only_one_flag(self):
  from types import SimpleNamespace
  import run_controlled
  a=run_controlled.build_command(SimpleNamespace(rounds=2,samples=4,repeat=1),('S_control','diverse','diagnostic','grounded'),3,Path('/private/control'))
  b=run_controlled.build_command(SimpleNamespace(rounds=2,samples=4,repeat=1),('S_sympy','diverse','diagnostic','grounded'),3,Path('/private/treatment'))
  for flag in ('--candidate-policy','--selection-policy','--repair-policy','--feedback-policy','--example-policy','--samples','--rounds'):
   self.assertEqual(a[a.index(flag)+1],b[b.index(flag)+1])
  self.assertNotIn('--shape-analysis',a);self.assertIn('--shape-analysis',b)

class SliceSafetyTests(unittest.TestCase):
 def test_named_view_propagation(self):
  result=analyze(kernel('v=a[:2,:3]\nt=nl.ndarray((2,3),a.dtype,buffer=nl.sbuf)\nni.dma_copy(dst=t,src=v)'),{'a':(4,5)})
  self.assertEqual(result['operations'][0]['src_shape'],['2','3']);self.assertEqual(result['operations'][0]['status'],'PROVEN_EQUAL')
 def test_unexecuted_loop_does_not_prove_runtime_violation(self):
  result=analyze(kernel('for i in range(0):\n t=nl.ndarray((256,3),nl.float32,buffer=nl.sbuf)'))
  self.assertFalse(result['violations']);self.assertFalse(result['loops'][0]['body_executes'])

class NKIBoundaryTests(unittest.TestCase):
 def test_nki_overslice_not_numpy_clipped(self):
  result=analyze(kernel('t=nl.ndarray((2,3),nl.float32,buffer=nl.sbuf)\nni.dma_copy(dst=t,src=a[:2,:4])'),{'a':(2,3)})
  self.assertIn('slice_bounds',[v['kind'] for v in result['violations']]);self.assertIsNone(result['operations'][0]['src_shape'])
 def test_symbolic_full_slice_equality(self):
  result=analyze(kernel('P,F=a.shape\nt=nl.ndarray((P,F),a.dtype,buffer=nl.sbuf)\nni.dma_copy(dst=t,src=a[:P,:F])'),{'a':(K,M)})
  self.assertEqual(result['operations'][0]['status'],'PROVEN_EQUAL')
 def test_ragged_loop_actual_bad_bound(self):
  from synthetic_nki.curriculum import specs
  spec,rule=next((s,r) for s,r in specs() if s['kind']=='ragged_k_offset')
  broken=spec['source'].replace(rule[0],rule[1])
  result=analyze(broken,{'a':(7,2),'b':(7,3)})
  self.assertIn('slice_bounds',[v['kind'] for v in result['violations']])
