import unittest
from shape_repair import inspect_shapes,plan_repair,failure_input_shapes

class TargetedTests(unittest.TestCase):
 def test_reduction_rank_is_distinct_from_allocation(self):
  source='import nki.language as nl\ndef f(a):\n t=nl.ndarray((4,8),nl.float32,buffer=nl.sbuf)\n y=nl.sum(t,axis=1)\n return y'
  plan=plan_repair(source,'SBUF and PSUM tensors must have at least 2 dimensions')
  self.assertIn('line 4',plan['root_causes'][0]);self.assertIn('keepdims=True only if',plan['guidance'])
  self.assertEqual(inspect_shapes(source)['issues'],[])
  self.assertFalse(plan_repair(source.replace('axis=1','axis=1,keepdims=True'),'SBUF and PSUM tensors must have at least 2 dimensions')['root_causes'])
 def test_internal_matmul_reshape(self):
  source='import nki.language as nl\nimport nki.isa as ni\ndef f(a,b):\n s=nl.ndarray((128,64),nl.float32,buffer=nl.sbuf)\n m=nl.ndarray((128,512),nl.float32,buffer=nl.sbuf)\n d=nl.ndarray((1,64),nl.float32,buffer=nl.psum)\n ni.nc_matmul(dst=d,stationary=s,moving=m)\n return d'
  plan=plan_repair(source,'cannot reshape array of size 32768 into shape (1,64)')
  self.assertIn('produce (64,512)',plan['root_causes'][0]);self.assertIn('no explicit reshape',plan['guidance'])
  plan=plan_repair(source.replace(' return d',' d.reshape((1,64))\n return d'),'cannot reshape array of size 32768 into shape (1,64)')
  self.assertIn('explicit reshape calls also exist',plan['guidance'])
 def test_multi_axis_limits_and_accumulation(self):
  source='import nki.language as nl\nimport nki.isa as ni\ndef f(a,b):\n s=nl.ndarray((256,256),nl.float32,buffer=nl.sbuf)\n m=nl.ndarray((256,1024),nl.float32,buffer=nl.sbuf)\n d=nl.ndarray((256,1024),nl.float32,buffer=nl.psum)\n ni.nc_matmul(dst=d,stationary=s,moving=m)\n return d'
  plan=plan_repair(source,'dma_copy dst partition dimension 256 exceeds maximum 128')
  for text in ('K=256 exceeds 128','M=256 exceeds 128','N=1024 exceeds 512','overwrite','accumulate','Cover every output'):self.assertIn(text,plan['guidance'])
 def test_uncertain_operand_shapes_no_fake_result(self):
  source='import nki.isa as ni\ndef f(a,b,d):\n ni.nc_matmul(dst=d,stationary=a,moving=b)\n return d'
  self.assertFalse(plan_repair(source,'cannot reshape array of size 32768 into shape (1,64)')['root_causes'])
 def test_hbm_tensor_copy_precise_root(self):
  source='import nki.isa as ni\ndef f(a):\n ni.tensor_copy(dst=t,src=a)\n return t'
  plan=plan_repair(source,"tensor_copy src must be in ['sbuf', 'psum'], got private_hbm")
  self.assertIn('line 3',plan['root_causes'][0]);self.assertIn('use dma_copy',plan['guidance'])
 def test_generation_avoids_specialized_matmul_options(self):
  from shape_repair import constrained_prompt
  import agent
  text,meta=constrained_prompt(agent.first_prompt(4),'matmul')
  self.assertIn('ordinary matmul needs no perf_mode',text)
 def test_only_observed_case_binds_inputs(self):
  source='def nki_matmul_basic_(lhsT,rhs):\n return lhsT'
  shapes=failure_input_shapes(source,'0 of 1 shapes passed. On K=128 M=64 N=512: failed',3)
  self.assertEqual(shapes,{'lhsT':(128,64),'rhs':(128,512)})
  self.assertEqual(failure_input_shapes(source,'unknown shape',3),{})
if __name__=='__main__':unittest.main()
