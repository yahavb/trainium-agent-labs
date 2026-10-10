"""Mocked generation only; no inference or live grade() calls."""
import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import agent
from candidate_diversity import build_variants, diversity_metrics


class DiversityTests(unittest.TestCase):
    def test_standard_and_original_preserved(self):
        for repair in (False, True):
            prompt = agent.repair_prompt(1, 'def kernel(): pass', 'exact error') if repair else agent.first_prompt(1)
            self.assertEqual([v.prompt for v in build_variants(prompt, 4)], [prompt] * 4)
            variants = build_variants(prompt, 4, 'diverse', repair=repair)
            self.assertEqual(variants[0].prompt, prompt)
            self.assertEqual(len({v.prompt for v in variants}), 4)
            self.assertEqual(variants, build_variants(prompt, 4, 'diverse', repair=repair))
            if repair:
                self.assertTrue(all('def kernel(): pass' in v.prompt and 'exact error' in v.prompt for v in variants))

    def test_alternative_is_bounded_and_only_after_repeats(self):
        prompt = agent.repair_prompt(1, 'code', 'failure')
        before = build_variants(prompt, 4, 'diverse', repair=True, repeated=1)
        after = build_variants(prompt, 4, 'diverse', repair=True, repeated=2)
        self.assertIn('keep everything else identical', before[1].prompt)
        self.assertIn('preserving unrelated code', after[1].prompt)
        self.assertEqual(after[0].prompt, prompt)

    def test_duplicates_and_empty_sources(self):
        metrics = diversity_metrics(['def f():\n return 1', '# comment\ndef f():\n    return 1', 'def f():\n return 2', 'def f():\n return 1'])
        self.assertEqual(metrics['unique_exact_sources'], 3)
        self.assertEqual(metrics['unique_ast_sources'], 2)
        self.assertEqual(metrics['duplicate_fraction'], .5)
        self.assertEqual(diversity_metrics([])['duplicate_fraction'], 0.)

    def test_exact_call_count_and_order(self):
        prompts = [v.prompt for v in build_variants('prompt', 4, 'diverse')]
        with patch.object(agent, 'ask', side_effect=lambda a, p: p) as mocked:
            self.assertEqual(agent.ask_parallel(None, prompts, 4), prompts)
        self.assertEqual(mocked.call_count, 4)
        with patch.object(agent, 'ask', return_value='answer') as mocked:
            self.assertEqual(agent.ask_parallel(None, 'same', 4), ['answer'] * 4)
        self.assertEqual(mocked.call_count, 4)

    def test_invalid_parameters(self):
        for samples, policy in [(0, 'diverse'), (4, 'unknown')]:
            with self.assertRaises(ValueError): build_variants('prompt', samples, policy)
        with self.assertRaises(ValueError): agent.ask_parallel(None, ['short'], 4)

    def test_solve_diverse_both_selection_policies(self):
        for policy in ('reward', 'diagnostic'):
            options = SimpleNamespace(terse=0, rounds=2, samples=4, offline=False,
                                      give_up_after=4, candidate_policy='diverse', selection_policy=policy)
            reply = '```python\ndef kernel():\n return 1\n```'
            records = io.StringIO()
            with patch.object(agent, 'ask', return_value=reply) as mocked, \
                 patch.object(agent, 'grade', return_value=(.3, {}, 'dma_copy requires src and dst to have the same number of elements, got src=4, dst=16384')), \
                 contextlib.redirect_stdout(io.StringIO()):
                agent.solve(options, 1, records)
            rows = [json.loads(line) for line in records.getvalue().splitlines()]
            self.assertEqual(mocked.call_count, 8)
            self.assertEqual(len(rows), 8)
            self.assertEqual([r['prompt_strategy'] for r in rows[:4]], ['original','simple','shapes_buffers','boundaries_coverage'])
            self.assertEqual([r['selected'] for r in rows[:4]], [True,False,False,False])
            self.assertEqual(rows[0]['diversity']['duplicate_fraction'], .75)
            self.assertEqual(rows[0]['failure_category'], 'DMA_SHAPE_MISMATCH')


if __name__ == '__main__': unittest.main()
