import unittest
from synthetic_nki.primitives import specs
from synthetic_nki.verify import validate
from kernel_planner import semantic_gate,analyze_semantics
from failure_selection import classify_failure

class VerifiedPrimitiveTests(unittest.TestCase):
 def test_independent_numpy_primitive_checks(self):
  for s in specs():
   with self.subTest(kind=s['kind']):
    result=validate(s['source'],s)
    self.assertTrue(result['passed'],result);self.assertEqual(len(result['cases']),3)
 def test_k_accumulation_is_numerically_necessary(self):
  s=specs()[3];result=validate(s['source'].replace('accumulate=(step>0)','accumulate=False'),s)
  self.assertFalse(result['passed']);self.assertEqual(result['failure_category'],'NUMERICAL_MISMATCH')
 def test_wrong_scalar_namespace_captured(self):
  s=specs()[0];r=validate(s['source'].replace('nisa.tensor_scalar','nl.tensor_scalar'),s)
  self.assertFalse(r['passed']);self.assertIn("has no attribute 'tensor_scalar'",r['error'])
 def test_access_pattern_bounds_error(self):
  s=specs()[0];r=validate(s['source'].replace('[4,3]','[4,4]'),s)
  self.assertFalse(r['passed']);self.assertEqual(r['failure_category'],'OUT_OF_BOUNDS')
 def test_identity_semantic_proof(self):
  r=semantic_gate('def f(x,p):\n return x',1)
  self.assertEqual(r['status'],'PROVEN_INVALID');self.assertEqual(r['findings'][0]['line'],2)
 def test_unknown_not_rejected(self):
  r=semantic_gate('def f(x,p):\n return helper(x)',1)
  self.assertEqual(r['status'],'UNKNOWN')
 def test_self_product_positional_is_hypothesis(self):
  r=analyze_semantics('import nki.isa as ni\ndef f(x):\n ni.nc_matmul(out,x,x)','pool')
  self.assertTrue(r['matmul_calls'][0]['same_operand']);self.assertEqual(r['status'],'UNKNOWN')
 def test_reduction_scaling_primitive_not_complete_pool(self):
  r=semantic_gate(specs()[0]['source'],1)
  self.assertEqual(r['status'],'UNKNOWN')

class PrimitiveRepairEvidenceTests(unittest.TestCase):
 def test_repeated_overwrite_guidance(self):
  from shape_repair import plan_repair
  s=specs()[3];bad=s['source'].replace('accumulate=(step>0)','accumulate=False');failure=validate(bad,s)
  r=plan_repair(bad,failure['error'],input_shapes={'a':(12,5),'b':(12,7)})
  self.assertIn('repeatedly overwrites',r['guidance']);self.assertIn('accumulate=(k_index>0)',r['guidance'])
 def test_missing_return_not_scored_as_success(self):
  r=semantic_gate('def f(x,p):\n y=make_output(x)',1)
  self.assertEqual(r['status'],'PROVEN_INVALID');self.assertEqual(r['findings'][0]['kind'],'missing_returned_output')
 def test_partition_stride_source_site_and_actual_numbers(self):
  from shape_repair import plan_repair
  source='import nki.language as nl\ndef f(x):\n t=nl.ndarray((2,12),x.dtype,buffer=nl.sbuf)\n view=t.ap([[1,2],[1,12]])\n return view'
  r=plan_repair(source,'ap() pattern has invalid partition stride. Partition step 1 must equal tensor free dimension size 12.')
  self.assertIn('line 4',r['guidance']);self.assertIn('extent 12',r['guidance']);self.assertIn('not 1',r['guidance'])
