"""Shared-engine, sequence behavior A, and agent integration regressions."""
import contextlib
import io
import pathlib
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import agent
import checker

ROOT = pathlib.Path(checker.__file__).resolve().parent
SEQ_REFERENCE = ROOT / 'reference/seq/seq_a.v'
SEQ_BAD = ROOT / 'candidates/bad/sequence_generator'


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(checker, 'BUILD', self.directory))
        self.enable_principles = self.enable_detectors = False

    def grade(self, *args, **kwargs):
        return checker.grade(*args, principles=self.enable_principles or self.enable_detectors,
                             detectors=self.enable_detectors, **kwargs)

    def design(self, name, source):
        path = self.directory / (name + '.v')
        path.write_text(source)
        return path

    def test_problem_weights_and_reference_models(self):
        for name, p in checker.PROBLEMS.items():
            with self.subTest(problem=name):
                self.assertAlmostEqual(sum(p['weights'].values()), 1.)
                outputs = {port for port, direction, _ in p['ports'] if direction == 'output'}
                self.assertEqual(set(p['model']['output'](p['model']['reset_state'])), outputs)
                for segment in p['segments']:
                    self.assertIn(segment['label'], p['weights'])
                    if isinstance(segment['inputs'], list):
                        self.assertEqual(len(segment['inputs']), segment['cycles'])

    def test_sequence_reference_and_official(self):
        self.assertEqual(self.grade('sequence_generator', SEQ_REFERENCE), (1., 'correct'))
        self.assertIs(checker.official('sequence_generator', SEQ_REFERENCE), True)
        score, feedback = self.grade('sequence_generator', ROOT / 'reference/seq/seq_b.v')
        self.assertLess(score, 1.)
        self.assertIn('first_step: requirement:', feedback)

    def test_every_legacy_sequence_fault_and_feedback_modes(self):
        # Covers every fixture checked by the removed standalone sequence checker.
        checks = {'no_wrap': 'repetition', 'ignores_enable': 'disable',
                  'reset_position': 'reset_restart', 'bad_item': 'sequence',
                  'restart_on_pause': 'resume'}
        for name, required in checks.items():
            baseline = None
            for v1, v2 in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(design=name, v1=v1, v2=v2):
                    self.enable_principles, self.enable_detectors = v1, v2
                    score, feedback = self.grade('sequence_generator', SEQ_BAD / (name + '.v'))
                    self.assertLess(score, 1.)
                    self.assertIn(required + ': requirement:', feedback)
                    labels = re.findall(r'^(\w+): requirement:', feedback, re.M)
                    self.assertEqual(len(labels), len(set(labels)))
                    plain = '\n'.join(line for line in feedback.splitlines()
                                      if not line.startswith(('Principle:', 'Observed pattern:')))
                    if baseline is None:
                        baseline = (score, plain)
                    self.assertEqual((score, plain), baseline)
                    principles = [line for line in feedback.splitlines() if line.startswith('Principle:')]
                    self.assertLessEqual(len(principles), 2)
                    if v1 or v2:
                        self.assertTrue(principles)
                    if v2 and name != 'ignores_enable':
                        self.assertIn('Observed pattern:', feedback)
                    elif not v2:
                        self.assertNotIn('Observed pattern:', feedback)
                    self.assertNotRegex(feedback, r'(?i)expected|8\x27[hbd]|0x[0-9a-f]|\b[01]{8}\b')

    def test_reset_while_enabled_and_disabled_independently(self):
        source = SEQ_REFERENCE.read_text()
        for enabled in (0, 1):
            with self.subTest(enabled=enabled):
                # Initial reset (enable low) must remain valid. A midrun reset
                # returns AF but uses the wrong next position in this mode only.
                text = source.replace("  reg [2:0] idx;", "  reg [2:0] idx; reg started = 0;")
                text = text.replace("idx <= 3'd1; data <= 8'hAF;",
                    f"idx <= (enable == {enabled} && started) ? 3'd0 : 3'd1; data <= 8'hAF;")
                text = text.replace('else if (enable) begin', 'else if (enable) begin started <= 1;')
                path = self.design('wrong_reset_' + str(enabled), text)
                _, feedback = self.grade('sequence_generator', path)
                self.assertIn('reset_restart: requirement:', feedback)
                self.assertNotIn('reset_value:', feedback)
                self.assertNotIn('first_step:', feedback)

    def test_reset_checks_and_facts_apply_to_sequence(self):
        source = SEQ_REFERENCE.read_text()
        variants = {
            'sync': source.replace(' or negedge reset_n', ''),
            'wrong_reset_value': source.replace("data <= 8'hAF; end", "data <= 8'h00; end"),
        }
        self.enable_principles = True
        for name, text in variants.items():
            with self.subTest(name=name):
                path = self.design(name, text)
                _, feedback = self.grade('sequence_generator', path)
                self.assertIn('reset_immediate: requirement:', feedback)
                self.assertIn('reset_midrun: requirement:', feedback)
                self.assertIn('In your design, data is assigned in 1 place(s):', feedback)
                self.assertIn('Principle:', feedback)

    def test_sequence_plan_covers_lengths_and_reset_positions(self):
        p = checker.PROBLEMS['sequence_generator']
        _, samples = checker._probe(p)
        for label, count in [('sequence', 8), ('repetition', 16), ('disable', 5), ('resume', 16)]:
            self.assertEqual(sum(label in s['checks'] and s['kind'] == 'segment'
                                 for s in samples), count)
        # Both reset_restart episodes happen away from AF, regardless of enable.
        for i, sample in enumerate(samples):
            if sample['kind'] == 'reset' and sample['checks'] == ('reset_restart',) and sample['subcheck'] == 0:
                self.assertNotEqual(samples[i - 1]['want']['data'], checker.SEQUENCE[0])
        initial_output = p['model']['output'](p['model']['reset_state'])['data']
        self.assertEqual(initial_output, checker.SEQUENCE[0])
        first = next(s for s in samples if 'first_step' in s['checks'])
        self.assertEqual(first['want']['data'], checker.SEQUENCE[1])

    def synthetic_problem(self):
        """Third problem proves ports, polarity, reset type and inputs are data."""
        return dict(
            module='counter', ports=(('tick', 'input', 1), ('clear', 'input', 1),
                                    ('step', 'input', 2), ('count', 'output', 3), ('odd', 'output', 1)),
            clock='tick', reset='clear', reset_active=1, async_reset=False,
            model=dict(reset_state=0, next_state=lambda s, i: (s + i['step']) % 8,
                       output=lambda s: {'count': s, 'odd': s & 1}),
            initial_inputs={'step': 0}, reset_hold_cycles=2, restart_cycles=4,
            segments=[dict(label='sequence', cycles=5,
                           inputs=[{'step': n} for n in (1, 2, 0, 3, 1)], first_check='first_step',
                           requirement='each rising edge must add step to count')],
            midrun_inputs={'step': 1},
            requirements=dict(checker.RESET_REQUIREMENTS,
                              reset_midrun='reset held across a clock edge must restore the initial state and restart after release',
                              first_step='the first rising edge must add step'), observations={},
            weights=dict(compiles=.1, reset_value=.15, reset_midrun=.2, first_step=.15, sequence=.4),
            official=str(self.directory / 'no_official_testbench.v'))

    def test_multi_output_sync_active_high_and_settled_sampling(self):
        p = self.synthetic_problem()
        source = '''`timescale 1ns/1ps
module counter(input tick, clear, input [1:0] step, output reg [2:0] count, output reg odd);
always @(posedge tick) begin
  if (clear) begin count <= #0.8 0; odd <= #0.8 0; end
  else begin count <= #0.8 count + step; odd <= #0.8 ((count + step) & 1); end
end
// Catch any probe that changes stimulus at a rising edge or while clock is high.
always @(step or clear) if (tick) $fatal(1, "stimulus changed with clock high");
endmodule
'''
        path = self.design('counter', source)
        with mock.patch.dict(checker.PROBLEMS, {'counter': p}):
            self.assertEqual(self.grade('counter', path), (1., 'correct'))
            harness, samples = checker._probe(p)
            self.assertFalse(any(s['kind'] == 'initial_immediate' for s in samples))
            self.assertFalse(any(s['kind'] == 'reset' and s['subcheck'] == 0 for s in samples))
            self.assertNotIn('reset_immediate', harness)
            self.enable_detectors = True
            wrong = self.design('wrong_odd', source.replace('((count + step) & 1)', '0'))
            score, feedback = self.grade('counter', wrong)
            self.assertLess(score, 1.)
            self.assertIn('odd is wrong in bit(s) [0]', feedback)
            self.assertNotIn('reset_immediate:', feedback)
            self.assertIn('Observed pattern:', feedback)
            self.assertIn('Principle:', feedback)

    def test_early_finish_and_official_failure_do_not_pass(self):
        path = self.design('early_finish', SEQ_REFERENCE.read_text().replace(
            '  reg [2:0] idx;', '  reg [2:0] idx; initial #12 $finish;'))
        score, feedback = self.grade('sequence_generator', path)
        self.assertLess(score, 1.)
        self.assertIn('not observed (the simulation ended early)', feedback)
        with mock.patch.object(checker, '_official_ok', return_value=False):
            score, feedback = self.grade('sequence_generator', SEQ_REFERENCE)
        self.assertEqual(score, .9)
        self.assertIn('official: requirement:', feedback)

    def test_sequence_compile_and_timeout_principles(self):
        self.enable_principles = True
        failed = subprocess.CompletedProcess([], 1, stdout='', stderr='bad syntax')
        compiled = subprocess.CompletedProcess([], 0, stdout='', stderr='')
        for responses, expected in [([failed], 0.), ([compiled, subprocess.TimeoutExpired('vvp', 1)], .1)]:
            with mock.patch.object(checker.subprocess, 'run', side_effect=responses):
                score, feedback = self.grade('sequence_generator', SEQ_REFERENCE)
            self.assertEqual(score, expected)
            self.assertIn('Principle:', feedback)

    def test_agent_cli_problem_and_all_feedback_flags(self):
        for problem in checker.PROBLEMS:
            for flags, v1, v2 in [(['--explain'], False, False),
                                   (['--explain', '--principles'], True, False),
                                   (['--explain', '--principles-v2'], True, True)]:
                with self.subTest(problem=problem, flags=flags):
                    response = io.BytesIO(b'{"data": [{"id": "test-model"}]}')
                    with contextlib.chdir(self.directory), contextlib.redirect_stdout(io.StringIO()), \
                         mock.patch.object(sys, 'argv', ['agent.py', '--repeat', '0', '--problem', problem] + flags), \
                         mock.patch('urllib.request.urlopen', return_value=response):
                        result = runpy.run_path(str(ROOT / 'agent.py'), run_name='__main__')
                    self.assertEqual(result['a'].problem, problem)
                    self.assertTrue(result['a'].explain)
                    self.assertEqual((result["a"].principles, result["a"].detectors), (v1, v2))
                    self.assertIn(problem, result['spec'])

    def test_agent_sequence_rounds_use_shared_grader_and_diagnosis(self):
        self.enable_principles = self.enable_detectors = True
        replies = [f'```verilog\n{(SEQ_BAD / "bad_item.v").read_text()}\n```',
                   f'```verilog\n{SEQ_REFERENCE.read_text()}\n```']
        config = dict(checker.PROBLEMS['sequence_generator'],
                      official=str(ROOT / 'TestBench/sequence_generator_tb.v'))
        with contextlib.chdir(self.directory), contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(agent, 'ask', side_effect=replies) as ask, \
             mock.patch.dict(checker.PROBLEMS, {'sequence_generator': config}):
            log = agent.run_once('test-model', checker.problem_spec('sequence_generator'), 2, 0,
                                 'test', True, True, .7, 'sequence_generator', principles=True, detectors=True)
        self.assertEqual([entry['score'] for entry in log], [.4, 1.])
        self.assertIs(log[-1]['official_pass'], True)
        self.assertEqual(log[-1]['problem'], 'sequence_generator')
        followup = ask.call_args_list[1].args[0][0]['content']
        self.assertIn('say in one sentence', followup)
        self.assertIn('Principle:', followup)
        self.assertIn('Observed pattern:', followup)
        for style in ('mine', 'bench'):
            spec = checker.problem_spec('sequence_generator', style, True)
            self.assertIn('first enabled edge produces BC', spec)
            self.assertIn('asynchronous', spec)


class FileReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(checker, 'BUILD', self.directory))
        self.problem = checker.file_problem(ROOT / 'sequence_generator.md',
            ROOT / 'TestBench/sequence_generator_tb.v', SEQ_REFERENCE)

    def test_both_reference_sources_and_all_bad_designs(self):
        registered = dict(checker.PROBLEMS['sequence_generator'], ref_verilog=str(SEQ_REFERENCE))
        # An unused Python model must not affect file-based grading.
        registered['model'] = {'reset_state': None, 'next_state': lambda *_: self.fail('Python model used'),
                               'output': lambda *_: self.fail('Python model used')}
        for config in (registered, self.problem):
            with self.subTest(config='registered' if config is registered else 'files'):
                self.assertEqual(checker.grade(config, SEQ_REFERENCE), (1., 'correct'))
                for path in sorted(SEQ_BAD.glob('*.v')):
                    score, feedback = checker.grade(config, path, detectors=True, principles=True)
                    self.assertLess(score, 1.)
                    self.assertNotRegex(feedback, r'(?i)expected|8\x27[hbd]|0x[0-9a-f]|\b[01]{8}\b')
        score, feedback = checker.grade(self.problem, SEQ_BAD / 'ignores_enable.v')
        self.assertIn('1 cycle after enable went low', feedback)
        self.assertIn('data is wrong in bit(s)', feedback)

    def test_flags_are_independent_and_do_not_mutate_problem_data(self):
        path = ROOT / 'candidates/bad/lfsr_syncreset.v'
        baseline = checker.grade('lfsr', path)
        _, without_facts = checker.grade('lfsr', path, facts=False)
        self.assertIn('In your design,', baseline[1])
        self.assertNotIn('In your design,', without_facts)
        with mock.patch.object(checker, '_official_ok', return_value=True):
            self.assertEqual(checker.grade('lfsr', path, async_reset=False), (1., 'correct'))
        self.assertEqual(checker.grade('lfsr', path), baseline)
        bad = SEQ_BAD / 'bad_item.v'
        _, detector_feedback = checker.grade('sequence_generator', bad, detectors=True)
        self.assertIn('Observed pattern:', detector_feedback)
        self.assertNotIn('Principle:', detector_feedback)
        _, principle_feedback = checker.grade('sequence_generator', bad, principles=True)
        self.assertIn('Principle:', principle_feedback)
        self.assertNotIn('Observed pattern:', principle_feedback)

    def test_seeded_probe_and_inferred_ports(self):
        self.assertEqual(self.problem['ports'], checker.PROBLEMS['sequence_generator']['ports'])
        self.assertEqual(checker._harness(self.problem), checker._harness(checker.file_problem(
            ROOT / 'sequence_generator.md', ROOT / 'TestBench/sequence_generator_tb.v', SEQ_REFERENCE)))
        other = checker.file_problem(ROOT / 'sequence_generator.md',
                                    ROOT / 'TestBench/sequence_generator_tb.v', SEQ_REFERENCE, seed=12)
        self.assertNotEqual(checker._harness(self.problem), checker._harness(other))
        for sample in checker._probe(self.problem)[1]:
            self.assertEqual(sample['want'], {})

    def test_cli_file_only_needs_no_registry_entry(self):
        argv = ['checker.py', '--spec', str(ROOT / 'sequence_generator.md'),
                '--tb', str(ROOT / 'TestBench/sequence_generator_tb.v'),
                '--ref', str(SEQ_REFERENCE), '--dut', str(SEQ_REFERENCE),
                '--facts', '--detectors', '--principles', '--async-reset']
        with mock.patch.dict(checker.PROBLEMS, {}, clear=True), \
             mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(checker.grade(self.problem, SEQ_REFERENCE), (1., 'correct'))
            with self.assertRaises(SystemExit) as exit:
                runpy.run_path(str(ROOT / 'checker.py'), run_name='__main__')
        self.assertEqual(exit.exception.code, 0)
        self.assertEqual(output.getvalue(), 'score: 1.0\ncorrect\n')

    def test_agent_file_only_passes_all_options(self):
        response = io.BytesIO(b'{"data": [{"id": "test-model"}]}')
        argv = ['agent.py', '--spec', str(ROOT / 'sequence_generator.md'),
                '--tb', str(ROOT / 'TestBench/sequence_generator_tb.v'), '--ref', str(SEQ_REFERENCE),
                '--repeat', '0', '--explain', '--principles-v2', '--async-reset', '--no-facts']
        with contextlib.chdir(self.directory), mock.patch.object(sys, 'argv', argv), \
             mock.patch('urllib.request.urlopen', return_value=response), \
             contextlib.redirect_stdout(io.StringIO()):
            result = runpy.run_path(str(ROOT / 'agent.py'), run_name='__main__')
        self.assertEqual(result['problem']['ref_verilog'], str(SEQ_REFERENCE))
        self.assertEqual(result['grade_options'], dict(facts=False, detectors=True,
                                                      principles=True, async_reset=True))
        replies = [f'```verilog\n{(SEQ_BAD / "bad_item.v").read_text()}\n```',
                   f'```verilog\n{SEQ_REFERENCE.read_text()}\n```']
        with contextlib.chdir(self.directory), contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(agent, 'ask', side_effect=replies), \
             mock.patch.object(checker, 'grade', wraps=checker.grade) as grade:
            log = agent.run_once('test', result['spec'], 2, 0, 'files', True, True, .7,
                                 result['problem'], **result['grade_options'])
        self.assertLess(log[0]['score'], 1.)
        self.assertEqual(log[-1]['score'], 1.)
        passed = grade.call_args.kwargs
        self.assertIn('diagnostics', passed)
        self.assertFalse(passed['principles'])  # The agent retrieves taxonomy for its next prompt.
        for option in ('facts', 'detectors', 'async_reset'):
            self.assertEqual(passed[option], result['grade_options'][option])

    def test_lfsr_stdout_is_byte_identical(self):
        result = subprocess.run([sys.executable, 'checker.py'], cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, (ROOT / 'baseline_lfsr.txt').read_bytes())


if __name__ == '__main__':
    unittest.main()
