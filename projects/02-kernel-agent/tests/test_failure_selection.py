"""Pure classification/selection and mocked agent integration; never run grade()."""

import contextlib
import io
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
from failure_selection import (CATEGORIES, Candidate, classify_failure,
                               code_fingerprint, select_candidate)
from replay_selection import replay_records


DMA = ("0 of 4 shapes passed. On shape=(32, 12) as 3x4: raised AssertionError: "
       "dma_copy requires src and dst to have the same number of elements, "
       "got src=384, dst=512 The tile you allocated holds 512 elements but you copied 384 "
       "into it. nisa.dma_copy does not slice or broadcast: allocate the destination with "
       "EXACTLY the shape of the slice you are moving.")
API = ("0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AttributeError: "
       "module 'nki.isa' has no attribute 'multiply'")
BUFFER = ("raised AssertionError: dst must be in ['psum'], got sbuf "
          "Allocate the `dst` tile with buffer=nl.psum instead of nl.sbuf.")
DIMENSIONS = ("raised AssertionError: SBUF and PSUM tensors must have at least 2 "
              "dimensions (partition-dim and free-dim) Every SBUF and PSUM tile needs "
              "two dimensions: a partition dimension first, then a free dimension.")
SOURCE_A = "def candidate():\n    return 1\n"
SOURCE_B = "def candidate():\n    return 2\n"


class ClassificationTests(unittest.TestCase):
    def test_all_categories_and_original_feedback(self):
        cases = {
            "NO_CODE": "No code came back. Reply with one python code block.",
            "SYNTAX_ERROR": "The code does not parse: invalid decimal literal on line 2.",
            "STATIC_RULE_VIOLATION": "Rule violations, which score zero: line 4: calls `np.mean`",
            "INVALID_API_FUNCTION": API,
            "INVALID_API_ARGUMENT": "raised TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving'",
            "DMA_SHAPE_MISMATCH": DMA,
            "INVALID_BUFFER_PLACEMENT": BUFFER,
            "INVALID_TENSOR_DIMENSIONS": DIMENSIONS,
            "OUT_OF_BOUNDS": "Out-of-bound access for tensor x on dimension 0: index range [0, 127] exceed dimension size of 32",
            "NUMERICAL_MISMATCH": "NUMERICAL MISMATCH: worst error 0.8 of the output's RMS (0.4), tolerance 0.02.",
            "INCOMPLETE_OUTPUT": "OUTPUT IS 99% ZEROS while the reference is not.",
            "MEMORY_TRAFFIC_EXCESS": "CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving 2.00x the byte floor",
            "HARDWARE_CORRECTNESS_HAZARD": "CORRECT ON CPU BUT WRONG ON HARDWARE: incorrect results on hardware",
            "UNKNOWN": "raised RuntimeError: unfamiliar simulator failure",
        }
        self.assertEqual(set(cases), CATEGORIES)
        for category, text in cases.items():
            with self.subTest(category=category):
                diagnostic = classify_failure(text)
                self.assertEqual(diagnostic.failure_category, category)
                self.assertEqual(diagnostic.original_feedback, text)
                self.assertTrue(diagnostic.classification_reason)
                self.assertGreaterEqual(diagnostic.classification_confidence, 0)
                self.assertLessEqual(diagnostic.classification_confidence, 1)
                self.assertEqual(diagnostic, classify_failure(text))

    def test_dimension_normalization(self):
        changed = DMA.replace('(32, 12)', '(128, 64)').replace('3x4', '8x8').replace('384', '8192').replace('512', '65536')
        self.assertEqual(classify_failure(DMA).normalized_signature,
                         classify_failure(changed).normalized_signature)
        for one, two in [
            ('dma_copy dst partition dimension 256 exceeds maximum 128',
             'dma_copy dst partition dimension 512 exceeds maximum 128'),
            ('cannot reshape array of size 32768 into shape (1,64)',
             'cannot reshape array of size 65536 into shape (2,128)'),
            ('WRONG SHAPE: returned (64,), reference is (64,512).',
             'WRONG SHAPE: returned (128,), reference is (128,1024).'),
            ('value array of shape (65536,) could not be broadcast to indexing result of shape (16384,)',
             'value array of shape (32768,) could not be broadcast to indexing result of shape (8192,)'),
            ('Out-of-bound access for tensor x on dimension 0: index range [0, 127] exceed dimension size of 32',
             'Out-of-bound access for tensor x on dimension 1: index range [0, 255] exceed dimension size of 64'),
        ]:
            with self.subTest(one=one):
                self.assertEqual(classify_failure(one).normalized_signature,
                                 classify_failure(two).normalized_signature)

    def test_symbols_are_not_erased(self):
        self.assertNotEqual(classify_failure(API).normalized_signature,
                            classify_failure(API.replace('multiply', 'scalar_mul')).normalized_signature)

    def test_unknowns_and_success_do_not_claim_correctness(self):
        for text in ('', 'Correct on every shape.', 'compiler exploded unexpectedly'):
            with self.subTest(text=text):
                diagnostic = classify_failure(text)
                self.assertEqual(diagnostic.failure_category, 'UNKNOWN')
                self.assertEqual(diagnostic.classification_confidence, 0)
                self.assertEqual(diagnostic.original_feedback, text)
        self.assertNotEqual(classify_failure('unknown A').normalized_signature,
                            classify_failure('unknown B').normalized_signature)

    def test_additional_actual_patterns(self):
        cases = [
            ("'MemoryRegion' object is not callable", 'INVALID_API_FUNCTION'),
            ("There is no module named 'nki.nl'.", 'INVALID_API_FUNCTION'),
            ("'NkiTensor' object has no attribute 'mean'", 'INVALID_API_FUNCTION'),
            ("nc_matmul() got multiple values for argument 'dst'", 'INVALID_API_ARGUMENT'),
            ("tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm", 'INVALID_BUFFER_PLACEMENT'),
            ("Matmul contraction dimension 256 exceeds pmax=128", 'INVALID_TENSOR_DIMENSIONS'),
            ("value array of shape (65536,) could not be broadcast to indexing result of shape (16384,)", 'INVALID_TENSOR_DIMENSIONS'),
            ("NON-FINITE OUTPUT: 8 NaN and 0 Inf, first at (1, 2)", 'NUMERICAL_MISMATCH'),
            ("45% of the output is zero, in the block from (0, 0) to (64, 64)", 'INCOMPLETE_OUTPUT'),
        ]
        for text, category in cases:
            with self.subTest(text=text):
                self.assertEqual(classify_failure(text).failure_category, category)

    def test_primary_static_verdict_beats_appended_example(self):
        text = "Rule violations: no function named `kernel`; example: module 'nki.isa' has no attribute 'foo'"
        self.assertEqual(classify_failure(text).failure_category, 'STATIC_RULE_VIOLATION')


class SelectionTests(unittest.TestCase):
    def test_equal_reward_prefers_precise_target(self):
        decision = select_candidate([Candidate(.3, SOURCE_A, API),
                                     Candidate(.3, SOURCE_B, DMA)], 'diagnostic')
        self.assertEqual(decision.selected_index, 1)
        self.assertIn('unverified', decision.selection_reasons[1])

    def test_localization_breaks_equal_target_tie(self):
        candidates = [Candidate(.3, SOURCE_A, BUFFER),
                      Candidate(.3, SOURCE_B, 'line 12: ' + BUFFER)]
        self.assertEqual(select_candidate(candidates, 'diagnostic').selected_index, 1)

    def test_higher_reward_always_wins(self):
        for reward in (.30000000001, .5, 1.):
            with self.subTest(reward=reward):
                candidates = [Candidate(reward, SOURCE_A, 'unfamiliar error'),
                              Candidate(.3, SOURCE_B, 'line 12: ' + BUFFER)]
                self.assertEqual(select_candidate(candidates, 'diagnostic').selected_index, 0)

    def test_all_categories_at_lower_reward_cannot_override(self):
        for text in (DMA, API, BUFFER, DIMENSIONS, 'No code came back.', 'Rule violations: no function named x'):
            self.assertEqual(select_candidate([Candidate(.5, SOURCE_A, 'unknown'),
                                               Candidate(.3, SOURCE_B, text)], 'diagnostic').selected_index, 0)

    def test_stable_deterministic_ties(self):
        candidates = [Candidate(.3, SOURCE_A, DMA), Candidate(.3, SOURCE_B, BUFFER)]
        first = select_candidate(candidates, 'diagnostic')
        self.assertEqual(first.selected_index, 0)
        for _ in range(10):
            self.assertEqual(select_candidate(candidates, 'diagnostic'), first)

    def test_legacy_reward_policy(self):
        candidates = [Candidate(.3, SOURCE_A, API), Candidate(.3, SOURCE_B, DMA)]
        self.assertEqual(select_candidate(candidates).selected_index, 0)
        self.assertEqual(select_candidate(candidates, 'reward').selected_index, 0)
        candidates.append(Candidate(.5, SOURCE_A, API))
        self.assertEqual(select_candidate(candidates, 'reward').selected_index, 2)

    def test_same_class_can_win_or_lose_by_evidence(self):
        candidates = [Candidate(.3, SOURCE_A, BUFFER), Candidate(.3, SOURCE_B, DMA)]
        self.assertEqual(select_candidate(candidates, 'diagnostic').selected_index, 0)
        candidates[1] = Candidate(.3, SOURCE_B, 'line 7: ' + DMA)
        self.assertEqual(select_candidate(candidates, 'diagnostic').selected_index, 1)

    def test_repeated_code_prefers_structural_novelty(self):
        previous = Candidate(.3, SOURCE_A, DMA)
        candidates = [previous, Candidate(.3, SOURCE_B, DMA)]
        self.assertEqual(select_candidate(candidates, 'diagnostic', [[previous]]).selected_index, 1)
        self.assertEqual(select_candidate(candidates, 'reward', [[previous]]).selected_index, 0)

    def test_repeated_signature_counted_once_per_round(self):
        previous = Candidate(.3, 'def old():\n    return 9', DMA)
        candidates = [Candidate(.3, SOURCE_A, DMA), Candidate(.3, SOURCE_B, BUFFER)]
        decision = select_candidate(candidates, 'diagnostic', [[previous] * 4, [previous]])
        self.assertEqual(decision.evidence[0].repeated_signature_rounds, 2)
        self.assertEqual(decision.evidence[1].repeated_signature_rounds, 0)
        self.assertEqual(decision.selected_index, 1)

    def test_format_and_comments_do_not_create_novelty(self):
        self.assertEqual(code_fingerprint(SOURCE_A),
                         code_fingerprint('# another comment\ndef candidate():\n  return 1'))
        self.assertNotEqual(code_fingerprint(SOURCE_A), code_fingerprint(SOURCE_B))
        self.assertEqual(code_fingerprint('def broken('), code_fingerprint('def broken('))

    def test_metadata_is_additive_and_serializable(self):
        decision = select_candidate([Candidate(.3, SOURCE_A, API), Candidate(.3, SOURCE_B, DMA)], 'diagnostic')
        metadata = decision.log_metadata(1)
        self.assertEqual(metadata['failure_category'], 'DMA_SHAPE_MISMATCH')
        self.assertTrue(metadata['selected'])
        self.assertFalse(decision.log_metadata(0)['selected'])
        self.assertEqual(metadata['selection_policy'], 'diagnostic')
        self.assertEqual(metadata['candidate_index'], 1)
        self.assertIn('normalized_signature', metadata)
        self.assertIn('selection_reason', metadata)
        json.dumps(metadata)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_candidate([])
        with self.assertRaises(ValueError):
            select_candidate([Candidate(.3, SOURCE_A, API)], 'other')
        for reward in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                select_candidate([Candidate(reward, SOURCE_A, API)])


class AgentIntegrationTests(unittest.TestCase):
    def run_solve(self, policy=None, rounds=1, give_up_after=4):
        options = dict(terse=0, rounds=rounds, samples=2, offline=False,
                       give_up_after=give_up_after)
        if policy is not None:
            options['selection_policy'] = policy
        replies = ['```python\n' + SOURCE_A + '```', '```python\n' + SOURCE_B + '```']
        records = io.StringIO()
        parts = dict(parses=True, rules=True, runs=False, correct=False)
        # Both live entry points are mocked; no grade() temporary file or HTTP call.
        def fake_grade(source, level):
            return .3, parts.copy(), API if source == SOURCE_A.strip() else DMA
        with patch.object(agent, 'ask_parallel', return_value=replies) as ask_mock, \
             patch.object(agent, 'grade', side_effect=fake_grade) as grade_mock, \
             contextlib.redirect_stdout(io.StringIO()):
            result = agent.solve(SimpleNamespace(**options), 1, records)
        return result, [json.loads(line) for line in records.getvalue().splitlines()], ask_mock, grade_mock

    def test_default_logs_keep_exact_legacy_schema(self):
        _, records, _, _ = self.run_solve()
        self.assertEqual(set(records[0]), {'level', 'round', 'reward', 'parts',
                                         'prompt_chars', 'reply_chars', 'code', 'feedback'})

    def test_explicit_reward_mode_matches_default(self):
        self.assertEqual(self.run_solve()[1], self.run_solve('reward')[1])

    def test_diagnostic_logs_correct_selection_without_extra_generation(self):
        _, records, ask_mock, grade_mock = self.run_solve('diagnostic')
        self.assertEqual(ask_mock.call_count, 1)
        self.assertEqual(grade_mock.call_count, 2)
        self.assertEqual([r['selected'] for r in records], [False, True])
        self.assertEqual(records[1]['reward'], .3)
        self.assertEqual(records[1]['feedback'], DMA)
        self.assertEqual(records[1]['failure_category'], 'DMA_SHAPE_MISMATCH')
        self.assertIn('selection_reason', records[0])

    def test_repair_prompt_uses_selected_candidate_and_stays_unchanged(self):
        _, records, ask_mock, _ = self.run_solve('diagnostic', rounds=2)
        expected = agent.repair_prompt(1, SOURCE_B.strip(), DMA)
        self.assertEqual(ask_mock.call_args_list[1].args[1], expected)
        self.assertEqual(records[2]['selection_evidence']['repeated_signature_rounds'], 1)

    def test_original_repeat_stopping_is_retained(self):
        result, records, ask_mock, _ = self.run_solve('diagnostic', rounds=8, give_up_after=2)
        self.assertEqual(result, (.3, 2))
        self.assertEqual(ask_mock.call_count, 2)
        self.assertEqual(len(records), 4)

    def test_cli_default_and_choices_without_inference(self):
        for argv, expected in [([], 'reward'), (['--selection-policy', 'diagnostic'], 'diagnostic')]:
            with patch.object(sys, 'argv', ['agent.py', '--offline'] + argv), \
                 patch('builtins.open', return_value=io.StringIO()), \
                 patch.object(agent, 'solve', return_value=(.3, 1)) as solve_mock, \
                 contextlib.redirect_stdout(io.StringIO()):
                agent.main()
            self.assertEqual(solve_mock.call_args.args[0].selection_policy, expected)
        with patch.object(sys, 'argv', ['agent.py', '--selection-policy', 'invalid']), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            agent.main()

    def test_default_solve_matches_original_revision(self):
        # Compare against the real baseline implementation with all generation and
        # grading mocked, including its stale-latest empty-response behavior.
        root = Path(__file__).resolve().parents[3]
        original = subprocess.check_output(
            ['git', 'show', '8f1ca41:projects/02-kernel-agent/agent.py'], cwd=root, text=True)
        baseline = ModuleType('baseline_agent_for_regression')
        exec(compile(original, '<baseline-agent>', 'exec'), baseline.__dict__)
        options = SimpleNamespace(terse=0, rounds=4, samples=2, offline=False, give_up_after=4)
        answer_a = '```python\n' + SOURCE_A + '```'
        answer_b = '```python\n' + SOURCE_B + '```'
        replies = [[answer_a, answer_b], ['', ''], [answer_b, answer_a], [answer_a, answer_b]]
        def fake_grade(source, level):
            parts = dict(parses=bool(source), rules=bool(source), runs=False, correct=False)
            return ((.3, parts, API if source == SOURCE_A.strip() else DMA) if source
                    else (0., parts, 'No code came back.'))
        def run(module):
            log = io.StringIO()
            output = io.StringIO()
            with patch.object(module, 'ask_parallel', side_effect=replies) as ask_mock, \
                 patch.object(module, 'grade', side_effect=fake_grade), \
                 patch.object(module.time, 'perf_counter', return_value=100.), \
                 contextlib.redirect_stdout(output):
                result = module.solve(options, 1, log)
            return result, log.getvalue(), output.getvalue(), ask_mock.call_args_list
        self.assertEqual(run(agent), run(baseline))
        # Instrumentation may add metadata but must preserve original prompts,
        # outcomes, stopping, console output and the eight legacy record values.
        options.instrument = True
        actual = run(agent)
        expected = run(baseline)
        legacy_keys = {'level', 'round', 'reward', 'parts', 'prompt_chars',
                       'reply_chars', 'code', 'feedback'}
        stripped = ''.join(json.dumps({key:value for key,value in json.loads(line).items()
                                       if key in legacy_keys}) + '\n'
                           for line in actual[1].splitlines())
        self.assertEqual((actual[0], stripped, actual[2], actual[3]), expected)


class ReplayTests(unittest.TestCase):
    def records(self, level, rnd):
        return [dict(level=level, round=rnd, reward=.3, code=SOURCE_A, feedback=API),
                dict(level=level, round=rnd, reward=.3, code=SOURCE_B, feedback=DMA)]

    def test_replay_and_repeat_boundaries(self):
        rows = self.records(1, 0) + self.records(1, 1) + self.records(2, 0) + self.records(1, 0)
        report = replay_records(rows, samples=2)
        self.assertEqual(report['rounds_replayed'], 4)
        self.assertEqual(report['changed_choices'], 4)
        self.assertEqual(report['skipped_batches'], [])
        self.assertEqual(report['choices'][1]['candidates'][1]['selection_evidence']['repeated_signature_rounds'], 1)
        self.assertEqual(report['choices'][3]['candidates'][1]['selection_evidence']['repeated_signature_rounds'], 0)
        self.assertEqual(report['choices'][0]['legacy_reward'], report['choices'][0]['diagnostic_reward'])

    def test_partial_batch_or_missing_history_is_not_replayed(self):
        rows = self.records(1, 0)[:1] + self.records(1, 1) + self.records(2, 0)
        report = replay_records(rows, samples=2)
        self.assertEqual(report['rounds_replayed'], 1)
        self.assertEqual(len(report['skipped_batches']), 2)

    def test_empty_replay_and_invalid_samples(self):
        self.assertEqual(replay_records([])['rounds_replayed'], 0)
        with self.assertRaises(ValueError):
            replay_records([], samples=0)


if __name__ == '__main__':
    unittest.main()
