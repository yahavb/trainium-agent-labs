"""Detector and grading regressions; also run by ``python checker.py``."""

import contextlib
import io
import json
import pathlib
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import checker


ROOT = pathlib.Path(checker.__file__).resolve().parent
TB = ROOT / "TestBench/lfsr_tb.v"
REFERENCE = ROOT / "candidates/lfsr.v"


class CheckerTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="checker_test_"))
        self.directory = pathlib.Path(directory)
        self.enterContext(mock.patch.object(checker, "BUILD", self.directory))
        self.enable_principles, self.enable_detectors = False, True
        entries = json.loads((ROOT / "principles.json").read_text())
        self.principles = {entry["id"]: "Principle: " + entry["principle"] for entry in entries}

    def grade(self, *args, **kwargs):
        return checker.grade(*args, principles=self.enable_principles or self.enable_detectors,
                             detectors=self.enable_detectors, **kwargs)

    def principle_lines_for(self, *args):
        return checker._principle_lines(*args, principles=self.enable_principles or self.enable_detectors,
                                       detectors=self.enable_detectors)

    def patterns(self, feedback):
        return [line for line in feedback.splitlines() if line.startswith("Observed pattern: ")]

    def principle_lines(self, feedback):
        return [line for line in feedback.splitlines() if line.startswith("Principle: ")]

    def design(self, name, update):
        path = self.directory / (name + ".v")
        path.write_text("""module lfsr(input clk, input reset_n, output reg [7:0] data);
  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) data <= 8'b10001010;
    else begin
      """ + update + """
    end
  end
endmodule
""")
        return path

    def assert_modes(self, path, expected_score, signatures):
        baseline = None
        for v1, v2 in [(False, False), (True, False), (False, True), (True, True)]:
            with self.subTest(path=path.name, v1=v1, v2=v2):
                self.enable_principles, self.enable_detectors = v1, v2
                score, feedback = self.grade(TB, path)
                self.assertEqual(score, expected_score)
                if not v1 and not v2:
                    baseline = feedback
                plain = "\n".join(line for line in feedback.splitlines()
                                  if not line.startswith(("Principle: ", "Observed pattern: ")))
                self.assertEqual(plain, baseline)
                self.assertEqual(self.patterns(feedback), [checker._v2_observation(s, 8)
                                                          for s in signatures] if v2 else [])
                principles = self.principle_lines(feedback)
                self.assertLessEqual(len(principles), 2)
                if signatures and (v1 or v2):
                    v1_id = ("NEXT_STATE_WRONG_FIRST_EDGE" if "first_step:" in feedback
                             else "NEXT_STATE_DIVERGES_LATER")
                    expected = [self.principles[v1_id]]
                    if v2:
                        expected.append(self.principles[signatures[0]])
                    self.assertEqual(principles, expected)
                elif not v1 and not v2:
                    self.assertEqual(principles, [])
                if score == 1.0:
                    self.assertEqual(feedback, "correct")

    def test_reference_and_existing_bad_designs(self):
        self.assert_modes(REFERENCE, 1.0, [])
        cases = {
            "lfsr_badtap.v": (0.55, ["SHIFTED_TOWARD_MSB", "OFF_BY_ONE_VALUE"]),
            "lfsr_initial_hack.v": (0.8, []),
            "lfsr_syncreset.v": (0.65, []),
            "lfsr_two_drivers.v": (0.65, []),
            "shift_wrong_direction.v": (0.4, ["SHIFTED_TOWARD_LSB"]),
            "inverted_feedback.v": (0.4, ["SHIFTED_TOWARD_MSB", "OFF_BY_ONE_VALUE"]),
        }
        for name, (score, signatures) in cases.items():
            self.assert_modes(ROOT / "candidates/bad" / name, score, signatures)

    def test_every_detector_on_bad_designs(self):
        cases = {
            "partial_x": ("8'bx0010101", ["PARTIAL_X"]),
            "stuck_and_late": ("data", ["OUTPUT_STUCK", "ONE_CYCLE_LATE"]),
            "inverted_output": ("~8'b00010101", ["OUTPUT_INVERTED"]),
            "reversed_output": ("8'b10101000", ["BIT_ORDER_REVERSED"]),
            "shift_lsb": ("{1'b0, data[7:1]}", ["SHIFTED_TOWARD_LSB"]),
            "shift_msb": ("{data[6:0], 1'b0}", ["SHIFTED_TOWARD_MSB", "OFF_BY_ONE_VALUE"]),
            "off_by_one": ("8'b00010110", ["OFF_BY_ONE_VALUE"]),
            "early": ("8'b00101011", ["ONE_CYCLE_EARLY"]),
            "upper_zero": ("8'b00000101", ["UPPER_BITS_ZERO"]),
        }
        covered = set()
        for name, (value, signatures) in cases.items():
            path = self.design(name, f"data <= {value};")
            self.assert_modes(path, 0.4, signatures)
            _, feedback = self.grade(TB, path)
            self.assertNotIn("00010101", feedback)  # The expected value stays private.
            covered.update(signatures)
        self.assertEqual(covered, set(checker.V2_OBSERVATIONS))

    def test_guards_and_reference_samples(self):
        detect = checker._v2_signatures
        for observed in ["", "10101", "000010101", "xxxxxxxx", "zzzzzzzz", "xxxxxxxx!", "00010?01"]:
            with self.subTest(observed=observed):
                self.assertEqual(detect(observed, 0x15, "10001010", 8, 0x8a, 0x2b), [])
        for observed in ["X0010101", "0001010Z", "xz01xz01"]:
            self.assertEqual(detect(observed, 0x15, "10001010", 8, 0x8a, 0x2b), ["PARTIAL_X"])
        for previous in ["", "xxxxxxxx", "1000101x", "1010"]:
            self.assertEqual(detect("00010100", 0x15, previous, 8, 0x8a, 0x2b), ["OFF_BY_ONE_VALUE"])
        state = checker.PROBLEMS["lfsr"]["model"]["reset_state"]
        for _ in range(256):
            expected = checker._lfsr_step(state)
            self.assertEqual(detect(format(expected, "08b"), expected, format(state, "08b"),
                                    8, state, checker._lfsr_step(expected)), [])
            state = expected
        self.assertNotIn("OFF_BY_ONE_VALUE", detect("11111111", 0, "10001010", 8, 0x8a, 1))
        self.assertNotIn("OFF_BY_ONE_VALUE", detect("00000000", 255, "10001010", 8, 0x8a, 1))
        self.assertNotIn("UPPER_BITS_ZERO", detect("00000100", 0x15, "10001010", 8, 0x8a, 0x2b))
        self.assertNotIn("UPPER_BITS_ZERO", detect("00000101", 0x06, "10001010", 8, 0x8a, 0x2b))

    def test_observed_previous_state_and_entered_bit(self):
        detect = checker._v2_signatures
        for observed in ["01000101", "11000101"]:
            self.assertEqual(detect(observed, 0x15, "10001010", 8, 0x12, 0x2b), ["SHIFTED_TOWARD_LSB"])
        for observed in ["00010100", "00010101"]:
            self.assertEqual(detect(observed, 0x77, "10001010", 8, 0x12, 0x2b), ["SHIFTED_TOWARD_MSB"])
        self.assertEqual(detect("10001010", 0x15, "10001010", 8, 0x12, 0x2b), ["OUTPUT_STUCK"])
        self.assertEqual(detect("00010010", 0x15, "10001010", 8, 0x12, 0x2b), ["ONE_CYCLE_LATE"])

    def test_only_first_failing_cycle_is_reported(self):
        path = self.design("changing_patterns", """if (data == 8'h8a) data <= 8'h15;
      else if (data == 8'h15) data <= 8'h2a;
      else data <= 8'bxxxxxxxx;""")
        _, feedback = self.grade(TB, path)
        self.assertIn("at cycle 2 after release", feedback)
        self.assertNotIn("first_step:", feedback)
        self.assertEqual(self.patterns(feedback), [checker._v2_observation(s, 8)
                         for s in ["SHIFTED_TOWARD_MSB", "OFF_BY_ONE_VALUE"]])

    def test_detector_is_never_called_when_disabled(self):
        self.enable_detectors = False
        self.assertEqual(checker._v2_signatures("00010100", 0x15, "10001010", 8, 0x8a, 0x2b, detectors=False), [])
        with mock.patch.object(checker, "_v2_signatures", side_effect=AssertionError("v2 was called")):
            for v1 in [False, True]:
                self.enable_principles = v1
                score, feedback = self.grade(TB, ROOT / "candidates/bad/inverted_feedback.v")
                self.assertEqual(score, 0.4)
                self.assertEqual(self.patterns(feedback), [])

    def test_principle_priority_budget_and_deduplication(self):
        v1 = ["X_BEFORE_FIRST_EDGE", "MULTIPLE_DRIVERS"]
        v2 = ["PARTIAL_X", "OUTPUT_STUCK", "ONE_CYCLE_LATE"]
        self.assertEqual(self.principle_lines_for(v1, v2), [self.principles[s] for s in v1])
        self.assertEqual(self.principle_lines_for(v1[:1] * 2, v2),
                         [self.principles[v1[0]], self.principles[v2[0]]])
        self.assertEqual(self.principle_lines_for([], v2), [self.principles[s] for s in v2[:2]])
        self.enable_principles, self.enable_detectors = True, False
        self.assertEqual(self.principle_lines_for(v1[:1], v2), [self.principles[v1[0]]])
        self.enable_principles = False
        self.assertEqual(self.principle_lines_for(v1, v2), [])

    def test_compile_failure_and_timeout(self):
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="bad syntax")
        compiled = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        cases = [([failed], 0.0, "COMPILE_ERROR"),
                 ([compiled, subprocess.TimeoutExpired("vvp", 1)], 0.1, "TIMEOUT")]
        for responses, score, signature in cases:
            with mock.patch.object(checker.subprocess, "run", side_effect=responses):
                result, feedback = self.grade(TB, REFERENCE)
            self.assertEqual(result, score)
            self.assertEqual(self.patterns(feedback), [])
            self.assertEqual(self.principle_lines(feedback), [self.principles[signature]])

    def test_cli_flags_and_experiment_name(self):
        cases = [([], "mine", False, False),
                 (["--principles"], "mine_principles", True, False),
                 (["--principles-v2"], "mine_principles_v2", True, True)]
        for flags, experiment, v1, v2 in cases:
            response = io.BytesIO(b'{"data": [{"id": "test-model"}]}')
            with contextlib.chdir(self.directory), contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(sys, "argv", ["agent.py", "--repeat", "0"] + flags), \
                 mock.patch("urllib.request.urlopen", return_value=response):
                result = runpy.run_path(str(ROOT / "agent.py"), run_name="__main__")
            self.assertEqual(result["exp"], experiment)
            self.assertEqual(result["a"].principles, v1)
            self.assertEqual(result["a"].detectors, v2)


if __name__ == "__main__":
    unittest.main()
