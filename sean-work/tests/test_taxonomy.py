"""Classification, bounded principle retrieval, and read-only knowledge reports."""
import json
import contextlib
import io
import pathlib
import tempfile
import unittest

import checker
import taxonomy


class TaxonomyTests(unittest.TestCase):
    def test_keywords_match_whole_words_and_phrases(self):
        path = checker.ROOT / 'taxonomy.json'
        found = taxonomy.categories('data comes from the source when already set', path=path)
        self.assertNotIn('memory', found)
        self.assertNotIn('handshake', found)
        self.assertIn('memory', taxonomy.categories('8-bit RAM with read enable', path=path))
        definitions = {'categories': [{'name': 'test', 'spec_keywords': ['read enable']} ]}
        self.assertIn('test', taxonomy.categories('READ ENABLE asserted', data=definitions))
        self.assertNotIn('test', taxonomy.categories('bread enabled', data=definitions))

    def test_only_principles_are_sent_by_default(self):
        entry = dict(signature='SIG', description='some detection', rule='Keep state stable.')
        self.assertEqual(taxonomy.entry_text(entry), 'Principle: Keep state stable.')
        self.assertEqual(taxonomy.entry_text(entry, show_detect_text=True),
                         'Detected signature SIG: some detection.\nPrinciple: Keep state stable.')
        self.assertEqual(taxonomy.entry_text(dict(signature='SIG', description='some detection')), '')

    def test_missing_principles_do_not_take_slots_or_bytes(self):
        definitions = dict(signatures={'EMPTY': {'detect': 'A' * 1000},
                                      'GOOD': {'detect': 'B' * 1000}})
        for verbose in (False, True):
            entries = taxonomy.select(None, ['generic'], ['EMPTY', 'GOOD'], limit=1,
                max_bytes=2000, data=definitions, knowledge={'generic:GOOD': 'Keep state stable.'},
                show_detect_text=verbose)
            self.assertEqual([entry['key'] for entry in entries], ['generic:GOOD'])
            self.assertEqual(entries[0]['signature'], 'GOOD')
        entries = taxonomy.select(None, ['generic'], ['EMPTY', 'GOOD'], limit=1,
            max_bytes=40, data=definitions, knowledge={'generic:GOOD': 'Keep state stable.'})
        self.assertEqual(len(entries), 1)
        self.assertEqual(taxonomy.select(None, ['generic'], ['GOOD'], limit=0,
            data=definitions, knowledge={'generic:GOOD': 'Keep state stable.'}), [])

    def test_empty_category_rule_can_fall_back_to_generic(self):
        entries = taxonomy.select(None, ['lfsr', 'generic'], ['SIG'],
            data={'signatures': {}}, knowledge={'lfsr:SIG': '', 'generic:SIG': 'Hold state.'})
        self.assertEqual(entries[0]['key'], 'generic:SIG')

    def test_report_lists_all_text_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = (pathlib.Path(directory) / name for name in ('before.json', 'after.json'))
            before.write_text(json.dumps({'*:A': 'old', '*:B': 'removed', '*:S': 'same'}))
            after.write_text(json.dumps({'*:A': 'new', '*:C': 'added', '*:S': 'same'}))
            original_bytes, current_bytes = before.read_bytes(), after.read_bytes()
            report = taxonomy.knowledge_report(after, before)
            self.assertEqual(report['added'], {'*:C': 'added'})
            self.assertEqual(report['removed'], {'*:B': 'removed'})
            self.assertEqual(report['changed'], {'*:A': {'before': 'old', 'after': 'new'}})
            self.assertEqual(before.read_bytes(), original_bytes)
            self.assertEqual(after.read_bytes(), current_bytes)
            self.assertEqual(taxonomy.knowledge_report(after, before.parent / 'missing.json')['status'], 'original_missing')

    def test_prompt_option_controls_detection_text(self):
        problem = checker.load_problem(checker._find_spec('lfsr'))
        history = [dict(code='module lfsr; endmodule', score=0., feedback='failed',
                        diagnostics={'signatures': ['X_BEFORE_FIRST_EDGE']})]
        default = checker.prompt_problem(problem, principles=True)
        plain = checker.build_prompt(default, history)[0]['content']
        verbose = checker.prompt_problem(problem, principles=True, show_detect_text=True)
        full = checker.build_prompt(verbose, history)[0]['content']
        self.assertIn('Principle:', plain)
        self.assertNotIn('Detected signature', plain)
        self.assertNotIn('X_BEFORE_FIRST_EDGE', plain)
        self.assertIn('Detected signature X_BEFORE_FIRST_EDGE:', full)
        self.assertEqual(checker.prompt_audit(default)['selected_rules'],
                         checker.prompt_audit(verbose)['selected_rules'])

    def test_cli_detection_option_and_missing_original_report(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                checker.main(['--problem', 'lfsr', '--dut', str(checker.ROOT / 'candidates/bad/lfsr_syncreset.v'),
                              '--principles', '--show-detect-text', '--no-console-log'])
        self.assertEqual(result.exception.code, 1)
        self.assertIn('Detected signature', output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            rc = checker.main(['--knowledge-report', '--knowledge-original', 'missing_original.json', '--no-console-log'])
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads(output.getvalue())['status'], 'original_missing')


if __name__ == '__main__':
    unittest.main()
