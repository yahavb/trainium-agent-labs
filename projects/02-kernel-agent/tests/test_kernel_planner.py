import unittest
import tempfile
from pathlib import Path
from kernel_planner import plan,generation_prompt,for_level
from synthetic_nki.curriculum import specs
from synthetic_nki.verify import validate

class PlannerTests(unittest.TestCase):
 def test_pool_semantics(self):
  p=plan('average pooling',[(3,7,8)],pool_size=2)
  self.assertEqual(p['output_shape'],(3,3,4));self.assertEqual(p['divisor'],4)
  self.assertIn('ONLY the two window',p['guidance'])
 def test_transpose_preserves_partition(self):
  p=plan('transpose',[(8,35)],shape2D=(5,7))
  self.assertEqual(p['output_shape'],(8,35));self.assertIn('SAME partition',p['guidance'])
 def test_multi_axis_matmul_counts(self):
  p=plan('matmul',[(256,256),(256,1024)])
  self.assertEqual([p['tile_counts'][axis]['tile_count'] for axis in ('M','N','K')],['2','2','2'])
  self.assertTrue(p['requires_accumulation'])
 def test_invalid_contraction(self):self.assertEqual(plan('matmul',[(2,3),(4,5)])['status'],'PROVEN_INVALID')
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
