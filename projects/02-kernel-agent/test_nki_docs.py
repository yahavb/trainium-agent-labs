"""Run with: python -m unittest test_nki_docs"""

import tempfile
import unittest
from pathlib import Path

from nki_docs import augment_prompt, docs_context


class DocumentationTests(unittest.TestCase):
    def test_level_routing_and_limits(self):
        for level, symbol in ((1, 'nki.language.sum'), (2, 'nki.isa.dma_copy'),
                              (3, 'nki.isa.nc_matmul'), (8, 'nki.isa.activation')):
            text = docs_context(level)
            self.assertIn(symbol, text)
            self.assertLessEqual(len(text), 1600)
        text = docs_context(3)
        self.assertNotIn('### nki.isa.nc_matmul_mx', text)

    def test_feedback_overrides_level(self):
        text = docs_context(3, 'dma_copy() got an unexpected keyword argument')
        self.assertIn('dma_copy', text.split('Source:', 1)[1].splitlines()[0])

    def test_missing_disabled_and_tiny_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(docs_context(3, root=folder), '')
        for size in (0, 100, 300, 400, 800, 1600):
            self.assertLessEqual(len(docs_context(3, max_chars=size)), size)
        self.assertEqual(augment_prompt('base', 3, max_chars=0), 'base')

    def test_answer_budget_and_terse(self):
        prompt = 'base'
        augmented = augment_prompt(prompt, 3)
        self.assertIn('Source:', augmented)
        self.assertLessEqual(len(augmented) // 4 + 2500 + 64, 4096)
        long = 'x' * 10000
        self.assertEqual(augment_prompt(long, 3), long)
        self.assertLess(len(augment_prompt(prompt, 3, terse=2)), len(augmented))

    def test_compiler_error_and_path_containment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'skill'
            directory = root / 'references/debugging/error-codes'
            directory.mkdir(parents=True)
            (root / 'SKILL.md').write_text('test')
            (directory / 'EVRF001.md').write_text('# Error\nUse an initialized tile.\n')
            text = docs_context(3, 'NCC_EVRF001', root=root)
            self.assertIn('Use an initialized tile.', text)
            index = root / 'references/indices'
            index.mkdir()
            secret = Path(folder) / 'outside.md'
            secret.write_text('outside should not be included')
            (index / 'symbol-lookup.md').write_text(
                '| `nc_matmul` | nki.isa | desc | [doc](../../../outside.md) |')
            self.assertEqual(docs_context(3, root=root), '')


if __name__ == '__main__':
    unittest.main()
