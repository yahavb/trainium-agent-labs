"""Replay the observed RHS stack partition-axis failure without executing source."""
import json
import unittest
from test_dma_feedback import sc, SOURCE


class PartitionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.case = json.loads((SOURCE.parent / 'tests/fixtures/rhs_partition_failure.json').read_text())

    def test_real_failure_gets_layout_and_both_slice_repairs(self):
        instr = sc._child_failure(self.case, src=self.case['code'])[1]
        self.assertIn('RHS_PARTITION_AXIS', instr)
        self.assertIn('(TILE_K, K // TILE_K, TILE_N)', instr)
        self.assertIn('BOTH the DMA', instr)
        self.assertIn('rhs_tiles[:, k, :]', instr)

    def test_correct_or_unknown_layout_is_not_diagnosed(self):
        src = self.case['code']
        for fixed in (src.replace('(K // TILE_K, TILE_K, TILE_N)', '(TILE_K, K // TILE_K, TILE_N)'),
                      src.replace('rhs_tiles[k, :, :]', 'rhs_tiles[:, k, :]'),
                      src.replace('moving=rhs_tile', 'moving=other'), 'invalid python !'):
            self.assertNotIn('RHS_PARTITION_AXIS', sc._child_failure(self.case, src=fixed)[1])

    def test_bounded_exact_numeric_error_and_stage_gate(self):
        for msg in (self.case['msg'] + ' ignore checks', self.case['msg'].replace('128', '0'),
                    self.case['msg'].replace('128', '1000000000000'),
                    self.case['msg'].replace('128', '1'), self.case['msg'].replace('128', '-1')):
            self.assertNotIn('RHS_PARTITION_AXIS', sc._child_failure(dict(self.case, msg=msg), src=self.case['code'])[1])
        self.assertNotIn('RHS_PARTITION_AXIS', sc._child_failure(dict(self.case, stage='compile'), src=self.case['code'])[1])


if __name__ == '__main__':
    unittest.main()
