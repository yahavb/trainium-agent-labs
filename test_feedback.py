"""
test_feedback.py  —  Unit tests for feedback.py.

No EDA tools required: all checker results are mocked inline.
Run with:  python -m pytest test_feedback.py -v
      or:  python test_feedback.py
"""

import json
import os
import tempfile
import unittest
from feedback import (
    generate_feedback,
    generate_feedback_from_text,
    generate_feedback_from_file,
    adapt_teammate_json,
    parse_raw_compiler_text,
    ET_NONE, ET_SYNTAX, ET_FUNCTIONAL, ET_SYNTHESIS, ET_ENVIRONMENT,
    ET_TB_COMPILE, ET_SKIPPED,
)


# ---------------------------------------------------------------------------
# Helpers to build mock checker results
# ---------------------------------------------------------------------------

def _result(passed, compile_ok=True, compile_errors=None,
            sim_ok=True, sim_errors=None, sim_output="",
            synth_ok=True, synth_skipped=False, synth_errors=None):
    return {
        "passed": passed,
        "compile": {
            "ok": compile_ok,
            "errors": compile_errors or [],
        },
        "simulate": {
            "ok": sim_ok,
            "output": sim_output,
            "errors": sim_errors or [],
        },
        "synthesize": {
            "ok": synth_ok,
            "skipped": synth_skipped,
            "errors": synth_errors or [],
        },
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPassedDesign(unittest.TestCase):
    """All stages pass."""

    def setUp(self):
        self.result = generate_feedback(_result(passed=True))

    def test_passed_flag(self):
        self.assertTrue(self.result["passed"])

    def test_error_type_none(self):
        self.assertEqual(self.result["error_type"], ET_NONE)

    def test_no_raw_errors(self):
        self.assertEqual(self.result["raw_errors"], [])

    def test_no_failing_cases(self):
        self.assertEqual(self.result["failing_cases"], [])

    def test_has_suggestion(self):
        self.assertTrue(len(self.result["suggestions"]) > 0)


class TestSyntaxError(unittest.TestCase):
    """iverilog rejects the file."""

    def setUp(self):
        errors = [
            "design.v:5: syntax error",
            "design.v:5: Giving up on module adder.",
        ]
        self.result = generate_feedback(_result(
            passed=False,
            compile_ok=False, compile_errors=errors,
        ))

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_raw_errors_preserved(self):
        self.assertIn("design.v:5: syntax error", self.result["raw_errors"])

    def test_suggestion_mentions_syntax(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("syntax", combined)

    def test_no_failing_cases(self):
        self.assertEqual(self.result["failing_cases"], [])


class TestUndeclaredIdentifier(unittest.TestCase):
    """iverilog: undeclared wire."""

    def setUp(self):
        errors = ["design.v:8: error: 'carry' undeclared identifier"]
        self.result = generate_feedback(_result(
            passed=False,
            compile_ok=False, compile_errors=errors,
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_suggestion_mentions_undeclared(self):
        combined = " ".join(self.result["suggestions"]).lower()
        # Should mention declaration or identifier
        self.assertTrue("declared" in combined or "identifier" in combined)


class TestFunctionalFailures(unittest.TestCase):
    """Simulation runs but several test cases fail (adder_buggy scenario)."""

    FAIL_LINES = [
        "FAIL: 1 + 2 = 31, expected 3",
        "FAIL: 7 + 8 = 31, expected 15",
        "FAIL: 15 + 15 = 0, expected 30",
        "FAIL: 10 + 6 = 4, expected 16",
        "FAILED: 4 test(s) failed.",
    ]

    def setUp(self):
        self.result = generate_feedback(_result(
            passed=False,
            sim_ok=False, sim_errors=self.FAIL_LINES,
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_FUNCTIONAL)

    def test_raw_errors_preserved(self):
        self.assertEqual(self.result["raw_errors"], self.FAIL_LINES)

    def test_failing_cases_extracted(self):
        cases = self.result["failing_cases"]
        # Four FAIL lines should parse; one "FAILED:" summary line won't
        self.assertEqual(len(cases), 4)

    def test_failing_case_fields(self):
        case = self.result["failing_cases"][0]
        self.assertIn("inputs",   case)
        self.assertIn("got",      case)
        self.assertIn("expected", case)

    def test_first_case_values(self):
        case = self.result["failing_cases"][0]
        self.assertEqual(case["inputs"],   "1 + 2")
        self.assertEqual(case["got"],      "31")
        self.assertEqual(case["expected"], "3")

    def test_suggestion_mentions_test_cases(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("test case", combined)

    def test_subtraction_hint_present(self):
        # got=31 is >> expected=3, should trigger the subtraction hint
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("subtraction", combined)


class TestFunctionalSingleFailure(unittest.TestCase):
    """Exactly one simulation test fails."""

    def setUp(self):
        self.result = generate_feedback(_result(
            passed=False,
            sim_ok=False,
            sim_errors=["FAIL: 0 + 15 = 0, expected 15"],
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_FUNCTIONAL)

    def test_one_failing_case(self):
        self.assertEqual(len(self.result["failing_cases"]), 1)

    def test_case_values(self):
        case = self.result["failing_cases"][0]
        self.assertEqual(case["inputs"],   "0 + 15")
        self.assertEqual(case["got"],      "0")
        self.assertEqual(case["expected"], "15")


class TestSynthesisError(unittest.TestCase):
    """Yosys finds a non-synthesizable construct."""

    def setUp(self):
        errors = [
            "ERROR: Module 'adder' contains non-synthesizable 'initial' block at line 12.",
        ]
        self.result = generate_feedback(_result(
            passed=False,
            synth_ok=False, synth_skipped=False, synth_errors=errors,
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTHESIS)

    def test_raw_errors_preserved(self):
        self.assertIn(
            "ERROR: Module 'adder' contains non-synthesizable 'initial' block at line 12.",
            self.result["raw_errors"],
        )

    def test_suggestion_mentions_initial(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("initial", combined)


class TestSynthesisHierarchyError(unittest.TestCase):
    """Yosys cannot find the top module."""

    def setUp(self):
        errors = ["ERROR: Module 'adder' not found in design hierarchy."]
        self.result = generate_feedback(_result(
            passed=False,
            synth_ok=False, synth_skipped=False, synth_errors=errors,
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTHESIS)

    def test_suggestion_mentions_hierarchy(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertTrue("hierarchy" in combined or "module" in combined)


class TestSynthesisSkipped(unittest.TestCase):
    """Yosys not installed: synthesis stage skipped → must return passed=False."""

    def setUp(self):
        # checker.py sets passed=True and skipped=True when yosys is absent;
        # feedback must override this and return passed=False.
        self.result = generate_feedback(_result(
            passed=True,
            synth_ok=True,
            synth_skipped=True,
            synth_errors=[],
        ))

    def test_not_passed(self):
        # Skipped synthesis must never be treated as a pass
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SKIPPED)

    def test_suggestion_mentions_yosys(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("yosys", combined)

    def test_suggestion_does_not_install(self):
        # Feedback must only diagnose; it must not claim it is installing anything
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("installing yosys", combined)
        self.assertNotIn("will install", combined)


class TestEnvironmentTimeoutCompile(unittest.TestCase):
    """iverilog timed out."""

    def setUp(self):
        self.result = generate_feedback(_result(
            passed=False,
            compile_ok=False,
            compile_errors=["iverilog timed out after 30s"],
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_ENVIRONMENT)

    def test_suggestion_mentions_timeout(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("timed out", combined)


class TestEnvironmentTimeoutSimulate(unittest.TestCase):
    """vvp simulation timed out."""

    def setUp(self):
        self.result = generate_feedback(_result(
            passed=False,
            sim_ok=False,
            sim_errors=["vvp timed out after 30s"],
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_ENVIRONMENT)

    def test_suggestion_mentions_infinite_loop(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("infinite loop", combined)


class TestEnvironmentToolNotFound(unittest.TestCase):
    """WSL or tool not found on PATH."""

    def setUp(self):
        self.result = generate_feedback(_result(
            passed=False,
            compile_ok=False,
            compile_errors=["iverilog: command not found"],
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_ENVIRONMENT)


class TestReturnSchemaAlwaysComplete(unittest.TestCase):
    """All required keys are present regardless of result."""

    REQUIRED_KEYS = {
        "passed", "error_type", "raw_errors", "failing_cases",
        "source_file", "error_lines", "retry_target", "suggestions",
    }

    def _check_keys(self, result):
        missing = self.REQUIRED_KEYS - set(result.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_pass_case(self):
        self._check_keys(generate_feedback(_result(passed=True)))

    def test_compile_fail(self):
        self._check_keys(generate_feedback(_result(
            passed=False, compile_ok=False, compile_errors=["syntax error"],
        )))

    def test_sim_fail(self):
        self._check_keys(generate_feedback(_result(
            passed=False, sim_ok=False, sim_errors=["FAIL: 1 + 2 = 99, expected 3"],
        )))

    def test_synth_fail(self):
        self._check_keys(generate_feedback(_result(
            passed=False, synth_ok=False, synth_errors=["ERROR: initial block"],
        )))

    def test_synth_skipped(self):
        self._check_keys(generate_feedback(_result(
            passed=True, synth_ok=True, synth_skipped=True,
        )))


class TestJsonSerializable(unittest.TestCase):
    """Output must be JSON-serializable (no custom objects)."""

    import json as _json

    def test_pass(self):
        import json
        r = generate_feedback(_result(passed=True))
        json.dumps(r)  # raises TypeError if not serializable

    def test_functional_fail(self):
        import json
        r = generate_feedback(_result(
            passed=False,
            sim_ok=False,
            sim_errors=["FAIL: 3 + 4 = 0, expected 7"],
        ))
        json.dumps(r)


# ---------------------------------------------------------------------------
# New fields on existing paths
# ---------------------------------------------------------------------------

class TestNewFieldsOnPassedResult(unittest.TestCase):
    """source_file, error_lines, retry_target default correctly on pass."""

    def setUp(self):
        self.result = generate_feedback(_result(passed=True))

    def test_source_file_none(self):
        self.assertIsNone(self.result["source_file"])

    def test_error_lines_empty(self):
        self.assertEqual(self.result["error_lines"], [])

    def test_retry_target_none(self):
        self.assertIsNone(self.result["retry_target"])


class TestNewFieldsOnDesignCompileError(unittest.TestCase):
    """Design file errors: retry_target=design, error_type=syntax_compile."""

    def setUp(self):
        errors = [
            "adder.v:5: syntax error",
            "adder.v:5: Giving up on module adder.",
        ]
        self.result = generate_feedback(_result(
            passed=False, compile_ok=False, compile_errors=errors,
        ))

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_source_file(self):
        self.assertEqual(self.result["source_file"], "adder.v")

    def test_error_lines_parsed(self):
        # Both "adder.v:5: syntax error" and "adder.v:5: Giving up..." parse
        self.assertEqual(len(self.result["error_lines"]), 2)
        self.assertEqual(self.result["error_lines"][0]["line"], 5)
        self.assertEqual(self.result["error_lines"][0]["file"], "adder.v")


class TestNewFieldsOnSimFailure(unittest.TestCase):
    """Simulation failures point retry_target to design."""

    def setUp(self):
        self.result = generate_feedback(_result(
            passed=False,
            sim_ok=False,
            sim_errors=["FAIL: 1 + 2 = 31, expected 3"],
        ))

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_source_file_none(self):
        self.assertIsNone(self.result["source_file"])

    def test_error_lines_empty(self):
        self.assertEqual(self.result["error_lines"], [])


# ---------------------------------------------------------------------------
# parse_raw_compiler_text
# ---------------------------------------------------------------------------

# The exact example provided by the teammate
TB_RAW = """\
[tb] attempt 1: rejected - it does not compile:
tb.v:27: syntax error
tb.v:27: error: Malformed statement
tb.v:28: syntax error
tb.v:28: error: Malformed statement
"""

class TestParseRawCompilerText(unittest.TestCase):
    """Unit tests for the low-level text parser."""

    def setUp(self):
        self.parsed = parse_raw_compiler_text(TB_RAW)

    def test_is_testbench_error(self):
        self.assertTrue(self.parsed["is_testbench_error"])

    def test_source_file(self):
        self.assertEqual(self.parsed["source_file"], "tb.v")

    def test_raw_errors_count(self):
        # 4 compiler lines; the [tb] header is excluded
        self.assertEqual(len(self.parsed["raw_errors"]), 4)

    def test_raw_errors_preserved_verbatim(self):
        self.assertIn("tb.v:27: syntax error",            self.parsed["raw_errors"])
        self.assertIn("tb.v:27: error: Malformed statement", self.parsed["raw_errors"])

    def test_error_lines_count(self):
        self.assertEqual(len(self.parsed["error_lines"]), 4)

    def test_error_line_fields(self):
        el = self.parsed["error_lines"][0]
        self.assertIn("file",    el)
        self.assertIn("line",    el)
        self.assertIn("message", el)

    def test_error_line_values(self):
        el = self.parsed["error_lines"][0]
        self.assertEqual(el["file"],    "tb.v")
        self.assertEqual(el["line"],    27)
        self.assertEqual(el["message"], "syntax error")

    def test_error_lines_line_numbers(self):
        lines = [el["line"] for el in self.parsed["error_lines"]]
        self.assertEqual(lines, [27, 27, 28, 28])


class TestParseRawNoHeader(unittest.TestCase):
    """Plain iverilog output without a [tb] header → not flagged as testbench."""

    def setUp(self):
        raw = "design.v:10: error: 'out' undeclared identifier\n"
        self.parsed = parse_raw_compiler_text(raw)

    def test_not_testbench(self):
        self.assertFalse(self.parsed["is_testbench_error"])

    def test_source_file(self):
        self.assertEqual(self.parsed["source_file"], "design.v")


# ---------------------------------------------------------------------------
# generate_feedback_from_text — teammate raw text entry point
# ---------------------------------------------------------------------------

class TestFeedbackFromTextTestbenchError(unittest.TestCase):
    """The exact teammate example → testbench_compile."""

    def setUp(self):
        self.result = generate_feedback_from_text(TB_RAW)

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_TB_COMPILE)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "testbench")

    def test_source_file(self):
        self.assertEqual(self.result["source_file"], "tb.v")

    def test_error_lines_count(self):
        self.assertEqual(len(self.result["error_lines"]), 4)

    def test_error_lines_line_27(self):
        line_27 = [el for el in self.result["error_lines"] if el["line"] == 27]
        self.assertEqual(len(line_27), 2)

    def test_error_lines_line_28(self):
        line_28 = [el for el in self.result["error_lines"] if el["line"] == 28]
        self.assertEqual(len(line_28), 2)

    def test_raw_errors_preserved(self):
        self.assertIn("tb.v:27: syntax error",               self.result["raw_errors"])
        self.assertIn("tb.v:27: error: Malformed statement", self.result["raw_errors"])
        self.assertIn("tb.v:28: syntax error",               self.result["raw_errors"])
        self.assertIn("tb.v:28: error: Malformed statement", self.result["raw_errors"])

    def test_header_not_in_raw_errors(self):
        for e in self.result["raw_errors"]:
            self.assertNotIn("[tb]", e)

    def test_suggestion_mentions_testbench(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("testbench", combined)

    def test_suggestion_mentions_malformed(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("malformed", combined)

    def test_no_guessing_root_cause(self):
        # Suggestions must not claim certainty about the root cause
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("the problem is", combined)
        self.assertNotIn("the bug is",     combined)

    def test_failing_cases_empty(self):
        self.assertEqual(self.result["failing_cases"], [])


class TestFeedbackFromTextDesignError(unittest.TestCase):
    """Plain design error text (no [tb] header) → syntax_compile, design target."""

    RAW = (
        "adder.v:3: syntax error\n"
        "adder.v:3: Giving up on module adder.\n"
    )

    def setUp(self):
        self.result = generate_feedback_from_text(self.RAW)

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_source_file(self):
        self.assertEqual(self.result["source_file"], "adder.v")


class TestFeedbackFromTextSchemaComplete(unittest.TestCase):
    """generate_feedback_from_text always returns all required keys."""

    REQUIRED_KEYS = {
        "passed", "error_type", "raw_errors", "failing_cases",
        "source_file", "error_lines", "retry_target", "suggestions",
    }

    def test_schema(self):
        result = generate_feedback_from_text(TB_RAW)
        missing = self.REQUIRED_KEYS - set(result.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_json_serializable(self):
        import json
        result = generate_feedback_from_text(TB_RAW)
        json.dumps(result)  # raises TypeError if not serializable


# ---------------------------------------------------------------------------
# Skipped-stage scenarios
# ---------------------------------------------------------------------------

def _skipped_result(*, compile_present=True, sim_present=True, synth_present=True,
                    synth_skipped=False, synth_note="", compile_ok=True,
                    sim_ok=True, sim_errors=None, synth_ok=True):
    """
    Build a checker result where stages can be absent (empty dict) or present.
    An absent stage simulates the checker never running that stage at all.
    """
    result = {"passed": False}
    if compile_present:
        result["compile"] = {"ok": compile_ok, "errors": [] if compile_ok else ["compile error"]}
    if sim_present:
        result["simulate"] = {
            "ok": sim_ok,
            "output": "",
            "errors": sim_errors or ([] if sim_ok else ["skipped due to compile failure"]),
        }
    if synth_present:
        stage = {"ok": synth_ok, "skipped": synth_skipped, "errors": []}
        if synth_note:
            stage["note"] = synth_note
        result["synthesize"] = stage
    return result


class TestTruePass(unittest.TestCase):
    """passed=True only when ALL three stages explicitly passed (no skips)."""

    def test_all_three_pass(self):
        r = generate_feedback(_result(passed=True, synth_ok=True, synth_skipped=False))
        self.assertTrue(r["passed"])
        self.assertEqual(r["error_type"], ET_NONE)

    def test_synth_skipped_is_not_a_pass(self):
        r = generate_feedback(_result(passed=True, synth_ok=True, synth_skipped=True))
        self.assertFalse(r["passed"])

    def test_checker_passed_true_ignored_when_synth_skipped(self):
        # Even if checker.py says passed=True, we override it
        result = _result(passed=True, synth_ok=True, synth_skipped=True)
        result["synthesize"]["note"] = "yosys not found in WSL; install with: sudo apt-get install -y yosys"
        r = generate_feedback(result)
        self.assertFalse(r["passed"])


class TestCompileStageAbsent(unittest.TestCase):
    """Compilation stage dict entirely missing → ET_SKIPPED, suggest iverilog."""

    def setUp(self):
        self.result = generate_feedback(_skipped_result(compile_present=False))

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SKIPPED)

    def test_suggestion_mentions_iverilog(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("iverilog", combined)

    def test_suggestion_mentions_install(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("install", combined)

    def test_suggestion_does_not_edit_verilog(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("edit", combined)
        self.assertNotIn("fix the verilog", combined)


class TestSimulationStageAbsent(unittest.TestCase):
    """Simulation stage dict entirely missing → ET_SKIPPED, suggest vvp/iverilog."""

    def setUp(self):
        self.result = generate_feedback(_skipped_result(sim_present=False))

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SKIPPED)

    def test_suggestion_mentions_iverilog_or_vvp(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertTrue("iverilog" in combined or "vvp" in combined)


class TestSynthesisStageAbsent(unittest.TestCase):
    """Synthesis stage dict entirely missing → ET_SKIPPED, unknown reason."""

    def setUp(self):
        self.result = generate_feedback(_skipped_result(synth_present=False))

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SKIPPED)

    def test_suggestion_mentions_yosys(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("yosys", combined)

    def test_suggestion_does_not_claim_missing(self):
        # Unknown reason: must not assert yosys is definitely missing
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("yosys is not installed", combined)
        self.assertIn("verify", combined)


class TestSynthesisSkippedWithNote(unittest.TestCase):
    """Yosys absent: checker sets skipped=True and includes a note."""

    NOTE = "yosys not found in WSL; install with: sudo apt-get install -y yosys"

    def setUp(self):
        result = _result(passed=True, synth_ok=True, synth_skipped=True)
        result["synthesize"]["note"] = self.NOTE
        self.result = generate_feedback(result)

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SKIPPED)

    def test_note_in_raw_errors(self):
        self.assertIn(self.NOTE, self.result["raw_errors"])

    def test_suggestion_mentions_yosys(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("yosys", combined)

    def test_suggestion_mentions_install(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("install", combined)

    def test_suggestion_mentions_synthesis_required(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("synthesis", combined)


class TestSynthesisSkippedNoNote(unittest.TestCase):
    """Synthesis skipped but no note → unknown reason, cautious suggestion."""

    def setUp(self):
        result = _result(passed=False, synth_ok=True, synth_skipped=True)
        # No "note" key added
        self.result = generate_feedback(result)

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SKIPPED)

    def test_suggestion_does_not_assume_missing(self):
        combined = " ".join(self.result["suggestions"]).lower()
        # Must not flatly say "yosys is not installed"
        self.assertNotIn("yosys is not installed", combined)

    def test_suggestion_says_verify(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("verify", combined)

    def test_suggestion_says_rerun(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("rerun", combined)


class TestSimulationCascadeSkip(unittest.TestCase):
    """Simulation skipped because compilation failed — compile is the primary issue."""

    def setUp(self):
        # checker.py writes "skipped due to compile failure" into sim errors
        self.result = generate_feedback(_result(
            passed=False,
            compile_ok=False,
            compile_errors=["adder.v:3: syntax error"],
            sim_ok=False,
            sim_errors=["skipped due to compile failure"],
        ))

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_primary_error_is_compile(self):
        # The compile failure is primary; simulation cascade is secondary
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_raw_errors_are_compile_errors(self):
        self.assertIn("adder.v:3: syntax error", self.result["raw_errors"])


class TestSynthesisRanAndFailed(unittest.TestCase):
    """Yosys ran and found an actual design error → direct AI to fix Verilog."""

    def setUp(self):
        self.result = generate_feedback(_result(
            passed=False,
            synth_ok=False,
            synth_skipped=False,
            synth_errors=["ERROR: Module 'adder' contains non-synthesizable 'initial' block."],
        ))

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTHESIS)

    def test_retry_target_is_design(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_suggestion_mentions_verilog_fix(self):
        combined = " ".join(self.result["suggestions"]).lower()
        # Should mention RTL or synthesizable, not install anything
        self.assertTrue("synthesizable" in combined or "rtl" in combined)

    def test_suggestion_does_not_mention_install(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("install", combined)


class TestFeedbackIsReadOnly(unittest.TestCase):
    """Feedback suggestions must never claim to perform actions."""

    ACTION_PHRASES = [
        "i will install",
        "installing",
        "i am editing",
        "calling the ai",
        "running yosys",
        "running iverilog",
        "i have fixed",
    ]

    def _check_no_actions(self, result):
        combined = " ".join(result["suggestions"]).lower()
        for phrase in self.ACTION_PHRASES:
            self.assertNotIn(phrase, combined, f"Suggestion must not claim action: {phrase!r}")

    def test_skipped_synth_with_note(self):
        result = _result(passed=True, synth_ok=True, synth_skipped=True)
        result["synthesize"]["note"] = "yosys not found"
        self._check_no_actions(generate_feedback(result))

    def test_compile_absent(self):
        self._check_no_actions(generate_feedback(_skipped_result(compile_present=False)))

    def test_synth_absent(self):
        self._check_no_actions(generate_feedback(_skipped_result(synth_present=False)))

    def test_synthesis_error(self):
        self._check_no_actions(generate_feedback(_result(
            passed=False,
            synth_ok=False, synth_skipped=False,
            synth_errors=["ERROR: initial block"],
        )))


# ---------------------------------------------------------------------------
# Regression tests: synthesis skipped with note — correct note classification
# ---------------------------------------------------------------------------

def _synth_skipped_result(note: str):
    """Build a checker result where synthesis was skipped with the given note."""
    result = _result(passed=True, synth_ok=True, synth_skipped=True)
    if note:
        result["synthesize"]["note"] = note
    return result


class TestSynthSkipNoteToolMissing(unittest.TestCase):
    """Note explicitly says yosys is not found → recommend install."""

    NOTES = [
        "yosys not found in WSL; install with: sudo apt-get install -y yosys",
        "yosys: command not found",
        "yosys is not installed on this system",
        "not found: yosys",
    ]

    def _check(self, note):
        r = generate_feedback(_synth_skipped_result(note))
        combined = " ".join(r["suggestions"]).lower()
        # Must recommend installing yosys
        self.assertIn("install", combined, f"note={note!r}")
        self.assertIn("yosys",   combined, f"note={note!r}")
        # Must NOT claim it is disabled by config
        self.assertNotIn("disabled by configuration", combined, f"note={note!r}")
        # Must stay passed=False, ET_SKIPPED
        self.assertFalse(r["passed"],              f"note={note!r}")
        self.assertEqual(r["error_type"], ET_SKIPPED, f"note={note!r}")
        self.assertIsNone(r["retry_target"],        f"note={note!r}")

    def test_wsl_not_found(self):       self._check(self.NOTES[0])
    def test_command_not_found(self):   self._check(self.NOTES[1])
    def test_not_installed(self):       self._check(self.NOTES[2])
    def test_not_found_prefix(self):    self._check(self.NOTES[3])


class TestSynthSkipNoteDisabled(unittest.TestCase):
    """Note says synthesis is disabled by config → explain, do not claim missing."""

    NOTES = [
        "synthesis disabled by configuration",
        "synthesis is turned off",
        "synthesis off",
        "skip synthesis (config)",
    ]

    def _check(self, note):
        r = generate_feedback(_synth_skipped_result(note))
        combined = " ".join(r["suggestions"]).lower()
        # Must mention disabled / enabled / configuration
        self.assertTrue(
            "disabled" in combined or "enable" in combined or "configuration" in combined,
            f"Expected config language for note={note!r}, got: {combined!r}",
        )
        # Must NOT say yosys is missing or recommend an install command
        self.assertNotIn("yosys is not installed", combined, f"note={note!r}")
        self.assertNotIn("apt-get install",         combined, f"note={note!r}")
        self.assertNotIn("brew install",            combined, f"note={note!r}")
        # Must still be a failure
        self.assertFalse(r["passed"],                f"note={note!r}")
        self.assertEqual(r["error_type"], ET_SKIPPED, f"note={note!r}")
        self.assertIsNone(r["retry_target"],          f"note={note!r}")

    def test_disabled_by_configuration(self): self._check(self.NOTES[0])
    def test_turned_off(self):                self._check(self.NOTES[1])
    def test_synthesis_off(self):             self._check(self.NOTES[2])
    def test_config_skip(self):               self._check(self.NOTES[3])


class TestSynthSkipNoteUnknown(unittest.TestCase):
    """Note present but matches neither pattern → cautious unknown-reason message."""

    NOTES = [
        "an unexpected condition prevented synthesis",
        "internal error in synthesis wrapper",
        "",   # empty note → same as no note
    ]

    def _check(self, note):
        r = generate_feedback(_synth_skipped_result(note))
        combined = " ".join(r["suggestions"]).lower()
        # Must say verify / check, not claim yosys is missing or disabled
        self.assertIn("verify", combined,                  f"note={note!r}")
        self.assertNotIn("yosys is not installed", combined, f"note={note!r}")
        self.assertNotIn("disabled by configuration", combined, f"note={note!r}")
        self.assertFalse(r["passed"],                f"note={note!r}")
        self.assertEqual(r["error_type"], ET_SKIPPED, f"note={note!r}")
        self.assertIsNone(r["retry_target"],          f"note={note!r}")

    def test_unexpected_condition(self):  self._check(self.NOTES[0])
    def test_internal_error(self):        self._check(self.NOTES[1])
    def test_empty_note(self):            self._check(self.NOTES[2])


class TestSynthSkipNoteMutuallyExclusive(unittest.TestCase):
    """tool_missing and disabled suggestions must not bleed into each other."""

    def test_missing_note_has_no_disabled_language(self):
        r = generate_feedback(_synth_skipped_result("yosys not found"))
        combined = " ".join(r["suggestions"]).lower()
        self.assertNotIn("disabled", combined)
        self.assertNotIn("turned off", combined)

    def test_disabled_note_has_no_install_command(self):
        r = generate_feedback(_synth_skipped_result("synthesis disabled by configuration"))
        combined = " ".join(r["suggestions"]).lower()
        self.assertNotIn("apt-get install", combined)
        self.assertNotIn("brew install",    combined)

    def test_disabled_note_says_enable_not_install(self):
        r = generate_feedback(_synth_skipped_result("synthesis disabled by configuration"))
        combined = " ".join(r["suggestions"]).lower()
        self.assertIn("enable", combined)


# ---------------------------------------------------------------------------
# Teammate JSON format — adapt_teammate_json (pure transformation)
# ---------------------------------------------------------------------------

# The three canonical examples from the spec
_FUNCTIONAL_DATA = {
    "name": "binary_to_gray",
    "spec": "4-bit binary to gray code converter",
    "design": "module binary_to_gray ... endmodule",
    "testbench": "...",
    "score": 0.5,
    "result": "8 of 16 checks failed.",
    "mismatches": [
        "MISMATCH bin=3 expected gray=2 got gray=3",
        "MISMATCH bin=6 expected gray=5 got gray=7",
    ],
}

_COMPILE_DATA = {
    "name": "binary_to_gray",
    "score": 0.0,
    "result": "COMPILE ERROR:\ndut.v:5: syntax error\ndut.v:5: error: Invalid module item.",
    "mismatches": [],
}

_TIMEOUT_DATA = {
    "name": "binary_to_gray",
    "score": 0.0,
    "result": "SIMULATION TIMEOUT (likely an infinite loop)",
    "mismatches": [],
}


class TestAdaptFunctionalMismatches(unittest.TestCase):
    """Exact functional-mismatch example from the spec."""

    def setUp(self):
        self.result = adapt_teammate_json(_FUNCTIONAL_DATA)

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_FUNCTIONAL)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_failing_cases_count(self):
        self.assertEqual(len(self.result["failing_cases"]), 2)

    def test_first_case_inputs(self):
        self.assertEqual(self.result["failing_cases"][0]["inputs"], "bin=3")

    def test_first_case_expected(self):
        self.assertEqual(self.result["failing_cases"][0]["expected"], "gray=2")

    def test_first_case_got(self):
        self.assertEqual(self.result["failing_cases"][0]["got"], "gray=3")

    def test_second_case_inputs(self):
        self.assertEqual(self.result["failing_cases"][1]["inputs"], "bin=6")

    def test_second_case_expected(self):
        self.assertEqual(self.result["failing_cases"][1]["expected"], "gray=5")

    def test_second_case_got(self):
        self.assertEqual(self.result["failing_cases"][1]["got"], "gray=7")

    def test_raw_errors_are_mismatch_lines(self):
        self.assertIn("MISMATCH bin=3 expected gray=2 got gray=3", self.result["raw_errors"])
        self.assertIn("MISMATCH bin=6 expected gray=5 got gray=7", self.result["raw_errors"])

    def test_suggestion_mentions_module_name(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("binary_to_gray", combined)

    def test_suggestion_mentions_gray_code_hint(self):
        # Gray-code specific hint should fire when "gray" appears in cases
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("gray", combined)
        self.assertIn("xor", combined)

    def test_schema_complete(self):
        required = {"passed", "error_type", "raw_errors", "failing_cases",
                    "source_file", "error_lines", "retry_target", "suggestions"}
        self.assertEqual(required - set(self.result.keys()), set())

    def test_json_serializable(self):
        json.dumps(self.result)


class TestAdaptCompileError(unittest.TestCase):
    """Exact compile-error example from the spec."""

    def setUp(self):
        self.result = adapt_teammate_json(_COMPILE_DATA)

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        # dut.v does not match tb/testbench patterns → design
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_source_file(self):
        self.assertEqual(self.result["source_file"], "dut.v")

    def test_error_lines_count(self):
        self.assertEqual(len(self.result["error_lines"]), 2)

    def test_error_line_5_syntax(self):
        el = self.result["error_lines"][0]
        self.assertEqual(el["file"], "dut.v")
        self.assertEqual(el["line"], 5)
        self.assertEqual(el["message"], "syntax error")

    def test_error_line_5_invalid(self):
        el = self.result["error_lines"][1]
        self.assertEqual(el["file"], "dut.v")
        self.assertEqual(el["line"], 5)
        self.assertIn("Invalid module item", el["message"])

    def test_raw_errors_no_header(self):
        # "COMPILE ERROR:" header must not appear in raw_errors
        for e in self.result["raw_errors"]:
            self.assertNotIn("COMPILE ERROR:", e.upper())

    def test_raw_errors_contain_iverilog_lines(self):
        self.assertIn("dut.v:5: syntax error", self.result["raw_errors"])

    def test_failing_cases_empty(self):
        self.assertEqual(self.result["failing_cases"], [])

    def test_suggestion_mentions_syntax(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("syntax", combined)

    def test_schema_complete(self):
        required = {"passed", "error_type", "raw_errors", "failing_cases",
                    "source_file", "error_lines", "retry_target", "suggestions"}
        self.assertEqual(required - set(self.result.keys()), set())


class TestAdaptSimulationTimeout(unittest.TestCase):
    """Exact simulation-timeout example from the spec."""

    def setUp(self):
        self.result = adapt_teammate_json(_TIMEOUT_DATA)

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_ENVIRONMENT)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_raw_errors_contains_timeout_string(self):
        combined = " ".join(self.result["raw_errors"]).upper()
        self.assertIn("SIMULATION TIMEOUT", combined)

    def test_suggestion_mentions_infinite_loop(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("infinite loop", combined)

    def test_suggestion_mentions_always_block(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("always", combined)

    def test_failing_cases_empty(self):
        self.assertEqual(self.result["failing_cases"], [])

    def test_schema_complete(self):
        required = {"passed", "error_type", "raw_errors", "failing_cases",
                    "source_file", "error_lines", "retry_target", "suggestions"}
        self.assertEqual(required - set(self.result.keys()), set())


class TestAdaptMissingOptionalFields(unittest.TestCase):
    """Input with only the required 'result' key — all optional fields absent."""

    def setUp(self):
        self.result = adapt_teammate_json({
            "result": "2 of 4 checks failed.",
            "mismatches": ["MISMATCH a=1 expected b=0 got b=1"],
        })

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_FUNCTIONAL)

    def test_failing_cases_parsed(self):
        self.assertEqual(len(self.result["failing_cases"]), 1)

    def test_uses_unknown_name_gracefully(self):
        # suggestion should not crash when name is missing
        self.assertIsInstance(self.result["suggestions"], list)
        self.assertTrue(len(self.result["suggestions"]) > 0)


class TestAdaptEmptyMismatches(unittest.TestCase):
    """result indicates failure but mismatches list is empty."""

    def setUp(self):
        self.result = adapt_teammate_json({
            "name": "counter",
            "result": "3 of 8 checks failed.",
            "mismatches": [],
            "score": 0.375,
        })

    def test_not_passed(self):
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_FUNCTIONAL)

    def test_no_failing_cases(self):
        self.assertEqual(self.result["failing_cases"], [])

    def test_raw_errors_contains_result_string(self):
        self.assertIn("3 of 8 checks failed.", self.result["raw_errors"])

    def test_suggestion_notes_no_mismatches_provided(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("no", combined)  # "no individual mismatches were provided"


class TestAdaptScoreOnePerfectRun(unittest.TestCase):
    """score=1.0, no mismatches — functionally correct but synthesis not checked."""

    def setUp(self):
        self.result = adapt_teammate_json({
            "name": "adder",
            "result": "All 16 checks passed.",
            "mismatches": [],
            "score": 1.0,
        })

    def test_not_passed(self):
        # Synthesis was not run → must not declare passed=True
        self.assertFalse(self.result["passed"])

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SKIPPED)

    def test_suggestion_mentions_synthesis(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("synthesis", combined)

    def test_retry_target_none(self):
        self.assertIsNone(self.result["retry_target"])


class TestAdaptSynthesisNeverReported(unittest.TestCase):
    """Synthesis is never passed=True from the file adapter."""

    def test_compile_error_never_passes(self):
        r = adapt_teammate_json(_COMPILE_DATA)
        self.assertFalse(r["passed"])

    def test_functional_error_never_passes(self):
        r = adapt_teammate_json(_FUNCTIONAL_DATA)
        self.assertFalse(r["passed"])

    def test_timeout_never_passes(self):
        r = adapt_teammate_json(_TIMEOUT_DATA)
        self.assertFalse(r["passed"])


# ---------------------------------------------------------------------------
# generate_feedback_from_file — file I/O wrapper
# ---------------------------------------------------------------------------

class TestGenerateFeedbackFromFileSuccess(unittest.TestCase):
    """Happy path: write a temp input file, call the function, check output."""

    def _run(self, data):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "for_feedback.json")
            out_path = os.path.join(tmpdir, "feedback_result.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            result = generate_feedback_from_file(in_path, out_path)
            # Output file must exist and contain valid JSON
            with open(out_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        return result, saved

    def test_functional_output_file_written(self):
        result, saved = self._run(_FUNCTIONAL_DATA)
        self.assertEqual(result, saved)

    def test_functional_return_value(self):
        result, _ = self._run(_FUNCTIONAL_DATA)
        self.assertEqual(result["error_type"], ET_FUNCTIONAL)

    def test_compile_output_file_written(self):
        result, saved = self._run(_COMPILE_DATA)
        self.assertEqual(result, saved)

    def test_compile_return_value(self):
        result, _ = self._run(_COMPILE_DATA)
        self.assertEqual(result["error_type"], ET_SYNTAX)

    def test_timeout_output_file_written(self):
        result, saved = self._run(_TIMEOUT_DATA)
        self.assertEqual(result, saved)

    def test_output_schema_complete(self):
        result, _ = self._run(_FUNCTIONAL_DATA)
        required = {"passed", "error_type", "raw_errors", "failing_cases",
                    "source_file", "error_lines", "retry_target", "suggestions"}
        self.assertEqual(required - set(result.keys()), set())

    def test_output_file_is_valid_json(self):
        _, saved = self._run(_FUNCTIONAL_DATA)
        # json.load already proved this; just confirm it's a dict
        self.assertIsInstance(saved, dict)

    def test_original_input_not_modified(self):
        """The input file must be left unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "for_feedback.json")
            out_path = os.path.join(tmpdir, "feedback_result.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump(_FUNCTIONAL_DATA, fh)
            before = open(in_path, encoding="utf-8").read()
            generate_feedback_from_file(in_path, out_path)
            after = open(in_path, encoding="utf-8").read()
        self.assertEqual(before, after)


class TestGenerateFeedbackFromFileMalformedJSON(unittest.TestCase):
    """Malformed JSON must raise ValueError, not crash with a raw exception."""

    def test_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "bad.json")
            out_path = os.path.join(tmpdir, "out.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            with self.assertRaises(ValueError):
                generate_feedback_from_file(in_path, out_path)

    def test_error_message_mentions_malformed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "bad.json")
            out_path = os.path.join(tmpdir, "out.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                fh.write("not json at all")
            try:
                generate_feedback_from_file(in_path, out_path)
                self.fail("Expected ValueError")
            except ValueError as exc:
                self.assertIn("alformed", str(exc))  # "Malformed" or "malformed"


class TestGenerateFeedbackFromFileMissingResultKey(unittest.TestCase):
    """JSON that omits 'result' must raise ValueError."""

    def test_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "no_result.json")
            out_path = os.path.join(tmpdir, "out.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump({"name": "foo", "score": 0.0}, fh)
            with self.assertRaises(ValueError):
                generate_feedback_from_file(in_path, out_path)


class TestGenerateFeedbackFromFileNotADict(unittest.TestCase):
    """Top-level JSON array must raise ValueError."""

    def test_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "array.json")
            out_path = os.path.join(tmpdir, "out.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump(["not", "a", "dict"], fh)
            with self.assertRaises(ValueError):
                generate_feedback_from_file(in_path, out_path)


class TestGenerateFeedbackFromFileMissingInputFile(unittest.TestCase):
    """Missing input file must raise OSError."""

    def test_raises_os_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "does_not_exist.json")
            out_path = os.path.join(tmpdir, "out.json")
            with self.assertRaises(OSError):
                generate_feedback_from_file(in_path, out_path)


# ---------------------------------------------------------------------------
# Input validation — adapt_teammate_json and generate_feedback_from_file
# ---------------------------------------------------------------------------

from feedback import adapt_teammate_json  # already imported above; re-confirm


class TestValidateResultType(unittest.TestCase):
    """'result' must be a string."""

    def test_result_is_int(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": 42, "mismatches": []})
        self.assertIn("result", str(ctx.exception).lower())

    def test_result_is_none(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": None, "mismatches": []})

    def test_result_is_list(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": ["a", "b"], "mismatches": []})

    def test_result_is_dict(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": {}, "mismatches": []})


class TestValidateMismatchesType(unittest.TestCase):
    """'mismatches' must be a list of strings when present."""

    def test_mismatches_is_string(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "mismatches": "MISMATCH a=1 expected b=0 got b=1"})
        self.assertIn("mismatches", str(ctx.exception).lower())

    def test_mismatches_is_dict(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": "fail", "mismatches": {}})

    def test_mismatches_contains_int(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "mismatches": ["MISMATCH a=1 expected b=0 got b=1", 99]})
        self.assertIn("mismatches[1]", str(ctx.exception))

    def test_mismatches_contains_none(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": "fail", "mismatches": [None]})

    def test_mismatches_absent_is_ok(self):
        # Missing mismatches key should not raise
        r = adapt_teammate_json({"result": "all passed.", "score": 1.0})
        self.assertIsInstance(r, dict)

    def test_mismatches_empty_list_is_ok(self):
        r = adapt_teammate_json({"result": "fail", "mismatches": [], "score": 0.0})
        self.assertIsInstance(r, dict)


class TestValidateScoreType(unittest.TestCase):
    """'score' must be a finite number in [0, 1] when present."""

    def test_score_is_string(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "score": "0.5"})
        self.assertIn("score", str(ctx.exception).lower())

    def test_score_is_bool(self):
        # bool is a subclass of int — must be rejected explicitly
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": "fail", "score": True})

    def test_score_is_none_is_ok(self):
        # Missing/None score is treated as 0.0 — should not raise
        r = adapt_teammate_json({"result": "fail", "mismatches": []})
        self.assertIsInstance(r, dict)

    def test_score_negative(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "score": -0.1})
        self.assertIn("0 and 1", str(ctx.exception))

    def test_score_above_one(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "score": 1.1})
        self.assertIn("0 and 1", str(ctx.exception))

    def test_score_infinity(self):
        import math
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "score": math.inf})
        self.assertIn("finite", str(ctx.exception).lower())

    def test_score_nan(self):
        import math
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "score": math.nan})
        self.assertIn("finite", str(ctx.exception).lower())

    def test_score_zero_is_ok(self):
        r = adapt_teammate_json({"result": "fail", "mismatches": [], "score": 0.0})
        self.assertIsInstance(r, dict)

    def test_score_one_is_ok_no_mismatches(self):
        r = adapt_teammate_json({"result": "passed", "mismatches": [], "score": 1.0})
        self.assertIsInstance(r, dict)

    def test_score_integer_zero_is_ok(self):
        r = adapt_teammate_json({"result": "fail", "mismatches": [], "score": 0})
        self.assertIsInstance(r, dict)

    def test_score_integer_one_is_ok(self):
        r = adapt_teammate_json({"result": "all passed", "mismatches": [], "score": 1})
        self.assertIsInstance(r, dict)


class TestValidateContradictoryScore(unittest.TestCase):
    """score=1.0 with non-empty mismatches is a contradictory input."""

    def test_score_one_with_mismatches(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({
                "result": "all passed",
                "score": 1.0,
                "mismatches": ["MISMATCH bin=3 expected gray=2 got gray=3"],
            })
        msg = str(ctx.exception).lower()
        self.assertIn("contradictory", msg)

    def test_score_one_with_multiple_mismatches(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({
                "result": "fail",
                "score": 1.0,
                "mismatches": [
                    "MISMATCH a=0 expected b=1 got b=0",
                    "MISMATCH a=1 expected b=0 got b=1",
                ],
            })
        self.assertIn("2", str(ctx.exception))  # mentions count

    def test_score_point_nine_with_mismatches_is_ok(self):
        # score < 1.0 with mismatches is consistent
        r = adapt_teammate_json({
            "result": "fail",
            "score": 0.9,
            "mismatches": ["MISMATCH a=0 expected b=1 got b=0"],
        })
        self.assertIsInstance(r, dict)

    def test_score_one_integer_with_mismatches(self):
        # score=1 (int) with mismatches is also contradictory
        with self.assertRaises(ValueError):
            adapt_teammate_json({
                "result": "pass",
                "score": 1,
                "mismatches": ["MISMATCH a=0 expected b=1 got b=0"],
            })


class TestValidateViaFile(unittest.TestCase):
    """Validation errors from adapt_teammate_json are surfaced by generate_feedback_from_file."""

    def _write_and_call(self, data):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "in.json")
            out_path = os.path.join(tmpdir, "out.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            generate_feedback_from_file(in_path, out_path)

    def test_invalid_result_type_via_file(self):
        with self.assertRaises(ValueError):
            self._write_and_call({"result": 123, "mismatches": []})

    def test_contradictory_score_via_file(self):
        with self.assertRaises(ValueError):
            self._write_and_call({
                "result": "pass",
                "score": 1.0,
                "mismatches": ["MISMATCH a=0 expected b=1 got b=0"],
            })

    def test_invalid_score_via_file(self):
        with self.assertRaises(ValueError):
            self._write_and_call({"result": "fail", "score": 2.5, "mismatches": []})

    def test_error_message_mentions_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "bad_input.json")
            out_path = os.path.join(tmpdir, "out.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump({"result": 99, "mismatches": []}, fh)
            try:
                generate_feedback_from_file(in_path, out_path)
                self.fail("Expected ValueError")
            except ValueError as exc:
                self.assertIn("bad_input.json", str(exc))


# ---------------------------------------------------------------------------
# Source-analysis tests: design field used to improve suggestions
# ---------------------------------------------------------------------------

# Gray-code design using OR instead of XOR (wrong)
_GRAY_OR_DESIGN = """\
module binary_to_gray(
    input  [3:0] bin,
    output [3:0] gray
);
    assign gray[3] = bin[3];
    assign gray[2] = bin[3] | bin[2];
    assign gray[1] = bin[2] | bin[1];
    assign gray[0] = bin[1] | bin[0];
endmodule
"""

# Gray-code design using XOR (correct)
_GRAY_XOR_DESIGN = """\
module binary_to_gray(
    input  [3:0] bin,
    output [3:0] gray
);
    assign gray[3] = bin[3];
    assign gray[2] = bin[3] ^ bin[2];
    assign gray[1] = bin[2] ^ bin[1];
    assign gray[0] = bin[1] ^ bin[0];
endmodule
"""

# Design with a missing semicolon before endmodule (line 8 is the bad line)
_MISSING_SEMI_DESIGN = """\
module adder(
    input  [3:0] a,
    input  [3:0] b,
    output [4:0] sum
);
    assign sum = a + b
endmodule
"""

_GRAY_SPEC = "4-bit binary to gray code converter"
_ADDER_SPEC = "4-bit adder"

_GRAY_MISMATCHES = [
    "MISMATCH bin=3 expected gray=2 got gray=3",
    "MISMATCH bin=6 expected gray=5 got gray=7",
]


class TestGrayCodeOrInsteadOfXor(unittest.TestCase):
    """Gray-code design using | instead of ^: specific OR→XOR suggestion expected."""

    def setUp(self):
        self.result = adapt_teammate_json({
            "name": "binary_to_gray",
            "spec": _GRAY_SPEC,
            "design": _GRAY_OR_DESIGN,
            "result": "8 of 16 checks failed.",
            "mismatches": _GRAY_MISMATCHES,
            "score": 0.5,
        })

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_FUNCTIONAL)

    def test_or_xor_suggestion_present(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("|", combined)
        self.assertIn("^", combined)
        self.assertIn("replace", combined)

    def test_suggestion_cites_a_source_line(self):
        # Should mention at least one "line N:" reference
        combined = " ".join(self.result["suggestions"])
        self.assertRegex(combined, r"line \d+")

    def test_suggestion_does_not_mention_wrong_module(self):
        # Should not fire generic adder hints
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("carry bit", combined)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")

    def test_failing_cases_preserved(self):
        self.assertEqual(len(self.result["failing_cases"]), 2)


class TestGrayCodeXorNoFalseWarning(unittest.TestCase):
    """Gray-code design already using ^: no OR warning should appear."""

    def setUp(self):
        self.result = adapt_teammate_json({
            "name": "binary_to_gray",
            "spec": _GRAY_SPEC,
            "design": _GRAY_XOR_DESIGN,
            "result": "1 of 16 checks failed.",
            "mismatches": ["MISMATCH bin=1 expected gray=1 got gray=0"],
            "score": 0.9375,
        })

    def test_no_or_xor_suggestion(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("replace '|' with '^'", combined)

    def test_still_has_generic_suggestion(self):
        # Generic mismatch suggestion must still appear
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("mismatch", combined)

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_FUNCTIONAL)


class TestNonGrayModuleNoOrWarning(unittest.TestCase):
    """A non-Gray-code module using | in assigns: no OR warning (spec has no 'gray')."""

    def setUp(self):
        # An OR-based design for a non-gray module
        design = """\
module flag_or(input a, input b, output f);
    assign f = a | b;
endmodule
"""
        self.result = adapt_teammate_json({
            "name": "flag_or",
            "spec": "simple OR gate",
            "design": design,
            "result": "1 of 4 checks failed.",
            "mismatches": ["MISMATCH a=0 expected f=0 got f=1"],
            "score": 0.75,
        })

    def test_no_or_xor_suggestion(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("replace '|' with '^'", combined)
        self.assertNotIn("gray", combined)


class TestCompileMissingSemicolon(unittest.TestCase):
    """Compile error with design source showing a missing semicolon before endmodule."""

    def setUp(self):
        # Error reported on line 7 (endmodule), bad statement is on line 6
        self.result = adapt_teammate_json({
            "name": "adder",
            "spec": _ADDER_SPEC,
            "design": _MISSING_SEMI_DESIGN,
            "result": (
                "COMPILE ERROR:\n"
                "dut.v:7: syntax error\n"
                "dut.v:7: error: Invalid module item."
            ),
            "mismatches": [],
            "score": 0.0,
        })

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_specific_semicolon_suggestion(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("semicolon", combined)

    def test_suggestion_cites_source_line(self):
        combined = " ".join(self.result["suggestions"])
        self.assertRegex(combined, r"[Ll]ine \d+")

    def test_suggestion_quotes_offending_statement(self):
        # The offending statement text should appear in the suggestion
        combined = " ".join(self.result["suggestions"])
        self.assertIn("assign sum = a + b", combined)

    def test_original_error_preserved(self):
        self.assertIn("dut.v:7: syntax error", self.result["raw_errors"])

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "design")


class TestCompileNoDesignSource(unittest.TestCase):
    """Compile error without design source: generic fallback must still be present."""

    def setUp(self):
        self.result = adapt_teammate_json({
            "name": "adder",
            "result": (
                "COMPILE ERROR:\n"
                "dut.v:5: syntax error\n"
                "dut.v:5: error: Invalid module item."
            ),
            "mismatches": [],
            "score": 0.0,
            # No "design" key
        })

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_SYNTAX)

    def test_generic_suggestion_present(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertTrue(
            "syntax" in combined or "compile" in combined or "design" in combined,
            f"Expected generic suggestion, got: {combined!r}",
        )

    def test_raw_errors_preserved(self):
        self.assertIn("dut.v:5: syntax error", self.result["raw_errors"])


class TestTestbenchCompileError(unittest.TestCase):
    """Compile error in tb.v: retry_target must be 'testbench', not 'design'."""

    def setUp(self):
        self.result = adapt_teammate_json({
            "name": "adder",
            "design": _MISSING_SEMI_DESIGN,   # design present but error is in tb.v
            "result": (
                "COMPILE ERROR:\n"
                "tb.v:12: syntax error\n"
                "tb.v:12: error: Malformed statement"
            ),
            "mismatches": [],
            "score": 0.0,
        })

    def test_error_type(self):
        self.assertEqual(self.result["error_type"], ET_TB_COMPILE)

    def test_retry_target(self):
        self.assertEqual(self.result["retry_target"], "testbench")

    def test_suggestion_targets_testbench(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertIn("testbench", combined)

    def test_suggestion_does_not_say_fix_design(self):
        combined = " ".join(self.result["suggestions"]).lower()
        self.assertNotIn("fix the design", combined)
        self.assertNotIn("resubmit the design", combined)

    def test_source_file_is_tb(self):
        self.assertEqual(self.result["source_file"], "tb.v")

    def test_no_semicolon_suggestion_for_tb_error(self):
        # Source analysis of design must not fire when error is in tb.v
        combined = " ".join(self.result["suggestions"]).lower()
        # The specific "assign sum = a + b" line from _MISSING_SEMI_DESIGN
        # must not appear — that's a design-file detail irrelevant to tb errors
        self.assertNotIn("assign sum = a + b", combined)


class TestInvalidDesignSpecTestbenchFields(unittest.TestCase):
    """'design', 'spec', and 'testbench' must be strings when present."""

    def test_design_is_int(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "design": 42})
        self.assertIn("design", str(ctx.exception).lower())

    def test_design_is_list(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": "fail", "design": ["line1"]})

    def test_design_is_none_ok(self):
        # None / absent is allowed — treated as empty
        r = adapt_teammate_json({"result": "fail", "mismatches": [], "score": 0.0})
        self.assertIsInstance(r, dict)

    def test_spec_is_int(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "spec": 7})
        self.assertIn("spec", str(ctx.exception).lower())

    def test_spec_is_dict(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": "fail", "spec": {}})

    def test_testbench_is_int(self):
        with self.assertRaises(ValueError) as ctx:
            adapt_teammate_json({"result": "fail", "testbench": 99})
        self.assertIn("testbench", str(ctx.exception).lower())

    def test_testbench_is_list(self):
        with self.assertRaises(ValueError):
            adapt_teammate_json({"result": "fail", "testbench": [1, 2]})

    def test_testbench_absent_ok(self):
        r = adapt_teammate_json({"result": "fail", "mismatches": [], "score": 0.0})
        self.assertIsInstance(r, dict)


class TestSchemaAndSavingWithDesignField(unittest.TestCase):
    """Output schema is preserved when design/spec/testbench are provided."""

    REQUIRED_KEYS = {"passed", "error_type", "raw_errors", "failing_cases",
                     "source_file", "error_lines", "retry_target", "suggestions"}

    def _full_data(self):
        return {
            "name": "binary_to_gray",
            "spec": _GRAY_SPEC,
            "design": _GRAY_OR_DESIGN,
            "testbench": "// testbench source here",
            "result": "8 of 16 checks failed.",
            "mismatches": _GRAY_MISMATCHES,
            "score": 0.5,
        }

    def test_schema_complete(self):
        r = adapt_teammate_json(self._full_data())
        missing = self.REQUIRED_KEYS - set(r.keys())
        self.assertEqual(missing, set())

    def test_json_serializable(self):
        r = adapt_teammate_json(self._full_data())
        json.dumps(r)

    def test_output_file_written_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "for_feedback.json")
            out_path = os.path.join(tmpdir, "feedback_result.json")
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump(self._full_data(), fh)
            result = generate_feedback_from_file(in_path, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(result, saved)

    def test_input_file_not_modified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "for_feedback.json")
            out_path = os.path.join(tmpdir, "feedback_result.json")
            data = self._full_data()
            with open(in_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            before = open(in_path, encoding="utf-8").read()
            generate_feedback_from_file(in_path, out_path)
            after = open(in_path, encoding="utf-8").read()
            self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
