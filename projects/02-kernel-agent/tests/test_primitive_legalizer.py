import contextlib,io,json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from primitive_legalizer import legalize
from synthetic_nki.primitives import specs
from synthetic_nki.verify import validate
import agent

class PrimitiveLegalizerTests(unittest.TestCase):
 def test_missing_namespace_repaired_without_math_change(self):
  s=specs()[0];broken=s['source'].replace('nisa.tensor_scalar','nl.tensor_scalar').replace('op0=nl.multiply','op0=nisa.multiply')
  fixed,meta=legalize(broken)
  self.assertTrue(meta['applied']);self.assertTrue(validate(fixed,s)['passed'])
  self.assertIn('instruction_namespace',{c['kind'] for c in meta['changes']})
  self.assertIn('opcode_namespace',{c['kind'] for c in meta['changes']})
 def test_known_hbm_dst_staged(self):
  s=specs()[0]
  broken=s['source'].replace('    z=nl.ndarray(y.shape,a.dtype,buffer=nl.sbuf)\n','    out=nl.ndarray((2,3),a.dtype,buffer=nl.shared_hbm)\n').replace('dst=z,data=y','dst=out,data=y').replace('    out=nl.ndarray((2,3),a.dtype,buffer=nl.shared_hbm)\n    nisa.dma_copy(dst=out,src=z)','')
  self.assertFalse(validate(broken,s)['passed'])
  fixed,meta=legalize(broken);self.assertTrue(validate(fixed,s)['passed']);self.assertIn('hbm_scalar_staging',{c['kind'] for c in meta['changes']})
 def test_clean_bytes_unchanged(self):
  s=specs()[0];fixed,meta=legalize(s['source']);self.assertEqual(fixed,s['source']);self.assertFalse(meta['applied'])
 def test_unknown_buffer_not_rewritten(self):
  source='import nki.language as nl\nimport nki.isa as ni\ndef f(a):\n ni.tensor_scalar(dst=a,data=unknown,op0=nl.multiply,operand0=.5)'
  self.assertEqual(legalize(source)[0],source)
 def test_does_not_add_algorithm(self):
  source='import nki.language as nl\nimport nki.isa as ni\ndef f(a):\n return a'
  self.assertEqual(legalize(source)[0],source)
 def test_alias_shadowing_not_rewritten(self):
  source='import nki.language as nl\nimport nki.isa as ni\ndef f(a,nl):\n nl.tensor_scalar(dst=a,data=a,op0=nl.multiply,operand0=.5)'
  self.assertEqual(legalize(source)[0],source)
 def test_opt_in_source_provenance_and_call_budget(self):
  s=specs()[0]['source'].replace('nisa.tensor_scalar','nl.tensor_scalar');opts=SimpleNamespace(rounds=1,samples=4,offline=False,terse=0,give_up_after=4,primitive_policy='legalize')
  log=io.StringIO()
  with patch.object(agent,'ask',return_value=s) as ask,patch.object(agent,'grade',return_value=(.3,{},'unknown')),contextlib.redirect_stdout(io.StringIO()):agent.solve(opts,1,log)
  rows=[json.loads(x) for x in log.getvalue().splitlines()];self.assertEqual(ask.call_count,4)
  self.assertEqual(rows[0]['generated_source'],s.strip());self.assertTrue(rows[0]['primitive_changes']['applied']);self.assertIn('nisa.tensor_scalar',rows[0]['code'])
