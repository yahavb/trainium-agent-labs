"""Prompt ownership, unchanged extraction/grading, and loop log regressions."""
import ast
import contextlib
import copy
import io
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import agent
import checker
import prompt_regression

ROOT = pathlib.Path(checker.__file__).resolve().parent


class PromptOwnerTests(unittest.TestCase):
    def test_saved_prompt_cases_are_identical(self):
        cases = prompt_regression.load_cases()
        self.assertEqual({case['condition'] for case in cases},
            {'no_flags', 'explain', 'explain_principles', 'explain_principles_v2'})
        for case in cases:
            problem = checker.prompt_problem(case['problem'], spec=case['spec'],
                                            layout=case['layout'], show_detect_text=True, **case['options'])
            rebuilt = checker.build_prompt(problem, case['history'])
            self.assertEqual(rebuilt, json.loads((ROOT / 'prompts/regression/before' / (case['id'] + '.json')).read_text()))
            for suffix in ('.json', '.txt'):
                name = case['id'] + suffix
                self.assertEqual((ROOT / 'prompts/regression/before' / name).read_bytes(),
                                 (ROOT / 'prompts/regression/after' / name).read_bytes())

    def test_agent_contains_no_prompt_helpers_or_instructions(self):
        tree = ast.parse((ROOT / 'agent.py').read_text())
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertFalse(names & {'build_prompt', 'public_spec', 'problem_spec', 'extract', '_fit', '_bytes'})
        forbidden = ('Return only', 'Fix the failures', 'Use module name', 'Principle:',
                     'Your previous attempt:', 'Previous attempt:', 'Latest checker result',
                     'A checker tested', 'First, for each failure', 'Change the logic')
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertFalse(any(text in node.value for text in forbidden), node.value)

    def test_extraction_is_a_literal_move(self):
        before = ast.parse((ROOT / 'prompts/regression/before/source_agent.py').read_text())
        after = ast.parse((ROOT / 'checker.py').read_text())
        get = lambda tree: next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'extract')
        self.assertEqual(ast.dump(get(before)), ast.dump(get(after)))

    def test_jsonl_events_are_identical_to_the_old_loop(self):
        source = ROOT / 'prompts/regression/before/source_agent.py'
        old = types.ModuleType('old_agent_log_replay')
        old.__file__ = str(source)
        exec(compile(source.read_text(), str(source), 'exec'), old.__dict__)
        cases = prompt_regression.load_cases()
        cases = [case for case in cases if case['layout'] == 'bounded' and case['round'] == 1
                 and case['problem']['module'] == 'lfsr']
        with tempfile.TemporaryDirectory() as directory:
            for case in cases:
                with self.subTest(condition=case['condition']):
                    initial = case['history'][0]
                    replies = ['```verilog\n' + initial['code'] + '\n```',
                               '```verilog\n' + (ROOT / 'candidates/lfsr.v').read_text() + '\n```']
                    def run(client):
                        results = iter([(initial['score'], initial['feedback'], initial['diagnostics']),
                                        (1., 'correct', {'signatures': []})])
                        def grade(problem, path, diagnostics=None, **options):
                            score, feedback, detail = next(results)
                            diagnostics.update(copy.deepcopy(detail))
                            return score, feedback
                        events = []
                        with contextlib.redirect_stdout(io.StringIO()), \
                             mock.patch.object(client, 'ask', side_effect=replies), \
                             mock.patch.object(checker, 'grade', side_effect=grade), \
                             mock.patch.object(checker, 'official', return_value=False), \
                             mock.patch('time.monotonic', side_effect=[10., 11., 20., 21.]):
                            if client is old:
                                with mock.patch.object(old, 'grade', side_effect=grade):
                                    records = client.run_once('test-model', case['spec'], 2, 0, 'log_replay', True,
                                        case['options']['explain'], .7, case['problem'],
                                        taxonomy_path=case['options']['taxonomy_path'],
                                        knowledge_path=case['options']['knowledge_path'],
                                        output_dir=directory, log_sink=events.append,
                                        principles=case['options']['principles'], detectors=case['options']['detectors'])
                            else:
                                records = client.run_once('test-model', case['spec'], 2, 0, 'log_replay', True,
                                    case['options']['explain'], .7, case['problem'],
                                    taxonomy_path=case['options']['taxonomy_path'],
                                    knowledge_path=case['options']['knowledge_path'],
                                    output_dir=directory, log_sink=events.append,
                                    principles=case['options']['principles'], detectors=case['options']['detectors'])
                        return records, events
                    before, events_before = run(old)
                    after, events_after = run(agent)
                    self.assertEqual(json.dumps(before), json.dumps(after))
                    self.assertEqual(json.dumps(events_before), json.dumps(events_after))

    def test_interface_generation_is_opt_in(self):
        for row in checker.interface_comparison():
            self.assertEqual(row['current'], row['generated'])
        problem = checker.load_problem(ROOT / 'LFSR.md')
        default = checker.public_spec(problem)
        self.assertEqual(default, checker.public_spec(problem, interface_from_tb=True))
        self.assertIn(checker.interface_line(problem), default)

    def test_testbench_runner_needs_no_registry_or_reference(self):
        output = io.StringIO()
        with mock.patch.object(checker, '_official_ok', return_value=True) as official, \
             contextlib.redirect_stdout(output):
            code = checker.main(['--run-tb', '--tb', str(ROOT / 'TestBench/lfsr_tb.v'),
                                 '--dut', str(ROOT / 'verilog/reference/lfsr.v')])
        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), 'correct\n')
        official.assert_called_once()


if __name__ == '__main__':
    unittest.main()
