import unittest
import tempfile
from pathlib import Path
from kernel_planner import plan,generation_prompt,for_level
from synthetic_nki.curriculum import specs
from synthetic_nki.verify import validate

class PlannerTests(unittest.TestCase):
 def test_output_shapes_derived_from_reference(self):
  r=for_level(1)
  self.assertEqual([c['output_shape'] for c in r['cases']],[(32,16,16),(128,4,4),(8,8,8),(64,4,4)])
  self.assertEqual([c['output_shape'] for c in for_level(2)['cases']],[c['input_shapes'][0] for c in for_level(2)['cases']])
 def test_partition_tiling_counts_from_installed_limits(self):
  p=plan('x',[(256,256),(256,1024)])
  self.assertEqual([o['tile_count'] for o in p['partition_tiling']],['2','2']);self.assertTrue(p['requires_tiling'])
  self.assertEqual(p['tile_limits']['partition'],128)
 def test_traffic_budget_and_signatures_are_generated(self):
  from kernel_planner import guidance_sentences
  text=' '.join(guidance_sentences(for_level(7),'use nisa.nc_matmul'))
  self.assertIn('1.05x',text);self.assertIn('nisa.nc_matmul(dst, stationary, moving)',text)
 def test_prompt_budget_and_no_complete_kernel(self):
  text,meta=generation_prompt('base',1)
  self.assertTrue(text.startswith('base'));self.assertNotIn('def tensor_avgpool',text);self.assertTrue(meta['applied'])
  text,meta=generation_prompt('base',1,context=1);self.assertEqual(text,'base')
 def test_curriculum_actual_clean_mutation_restoration(self):
  for spec,(old,new,category,cause) in specs():
   with self.subTest(kind=spec['kind']):
    self.assertTrue(validate(spec['source'],spec)['passed'])
    broken=spec['source'].replace(old,new,1);bad=validate(broken,spec)
    self.assertFalse(bad['passed']);self.assertEqual(bad['failure_category'],category)
    self.assertTrue(validate(spec['source'],spec)['passed'])
 def test_three_dim_reduction_without_keepdims_is_valid(self):
  s=next(s for s,_ in specs() if s['kind']=='grouped_sum_offset')
  self.assertTrue(validate(s['source'],s)['passed'])
if __name__=='__main__':unittest.main()

class SemanticPlannerTests(unittest.TestCase):
 def test_self_product_recorded_as_evidence_only(self):
  from kernel_planner import analyze_semantics
  r=analyze_semantics('import nki.isa as i\ndef f(x):\n i.nc_matmul(dst=z,stationary=x,moving=x)','any')
  self.assertTrue(r['matmul_calls'][0]['same_operand']);self.assertEqual(r['findings'],[]);self.assertEqual(r['status'],'UNKNOWN')
 def test_constant_weight_matmul_not_blanket_rejected(self):
  from kernel_planner import analyze_semantics
  r=analyze_semantics('import nki.isa as i\ndef f(x):\n i.nc_matmul(dst=z,stationary=weights,moving=x)','average pooling')
  self.assertEqual(r['findings'],[])
 def test_valid_reduction_not_claimed_correct(self):
  from kernel_planner import analyze_semantics
  r=analyze_semantics('import nki.language as l\ndef f(x):\n return l.sum(x,axis=(1,2),keepdims=True)','average pooling')
  self.assertEqual(r['operation_selection'],'reduction_present');self.assertEqual(r['status'],'UNKNOWN')
 def test_source_never_executed(self):
  from kernel_planner import analyze_semantics
  with tempfile.TemporaryDirectory() as d:
   target=Path(d)/'touch';analyze_semantics(f"__import__('pathlib').Path({str(target)!r}).touch()",'pool')
   self.assertFalse(target.exists())
 def test_pool_self_product_independent_math_witness(self):
  import numpy as np
  x=np.array([[1.,2.],[3.,4.]])
  self.assertEqual(np.mean(2*x),2*np.mean(x))
  np.testing.assert_array_equal((2*x).T@(2*x),4*(x.T@x))
  self.assertFalse(np.allclose(x.T@x,np.full((2,2),np.mean(x))))

class SemanticCurriculumTests(unittest.TestCase):
 def test_real_semantic_and_shape_mutations(self):
  from synthetic_nki.semantic_curriculum import spec
  s=spec();self.assertTrue(validate(s['source'],s)['passed'])
  for old,new,category in [('(3,2,3)','(3,2,3,3)','DMA_SHAPE_MISMATCH'),('operand0=1.0/3','operand0=.5','NUMERICAL_MISMATCH'),('y=nl.sum','y=nl.max','NUMERICAL_MISMATCH')]:
   with self.subTest(category=category):
    result=validate(s['source'].replace(old,new,1),s)
    self.assertFalse(result['passed']);self.assertEqual(result['failure_category'],category)
