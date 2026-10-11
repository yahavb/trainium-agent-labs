"""
feedback.py  —  Feedback engine for the AI-powered Verilog debugging agent.

Accepts either:
  (a) the structured result dict produced by checker.py, or
  (b) raw compiler output text from a teammate's build system, or
  (c) the JSON format produced by a teammate's evaluation harness.

Public API
----------
    generate_feedback(checker_result: dict) -> dict
    generate_feedback_from_text(raw_text: str) -> dict
    generate_feedback_from_file(input_path="for_feedback.json",
                                output_path="feedback_result.json") -> dict
    parse_raw_compiler_text(raw_text: str) -> dict

Output schema
-------------
    {
        "passed":        bool,
        "error_type":    str,        # one of the ET_* constants below
        "raw_errors":    list[str],
        "failing_cases": list[{"inputs": str, "got": str, "expected": str}],
        "source_file":   str | None, # file that caused the failure
        "error_lines":   list[{"file": str, "line": int, "message": str}],
        "retry_target":  str | None, # "design", "testbench", or None
        "suggestions":   list[str]
    }

passed=True is ONLY returned when compilation, simulation, and synthesis all
explicitly passed.  A skipped stage always produces passed=False.

Integration note (generate_feedback_from_file)
-----------------------------------------------
The teammate's JSON format does not include a synthesis result.  Therefore
passed=True is never returned by generate_feedback_from_file — synthesis
was neither run nor skipped explicitly, so the design cannot be declared
complete.  The caller should run a separate synthesis check if needed.
"""

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Error type constants
# ---------------------------------------------------------------------------

ET_NONE        = "none"               # all stages passed
ET_SYNTAX      = "syntax_compile"     # design file rejected by iverilog
ET_TB_COMPILE  = "testbench_compile"  # testbench file rejected by iverilog
ET_FUNCTIONAL  = "functional_sim"     # simulation ran but tests failed
ET_SYNTHESIS   = "synthesis"          # yosys found non-synthesizable constructs
ET_ENVIRONMENT = "environment"        # tool timeout / tool not found / WSL error
ET_SKIPPED     = "stage_skipped"      # a required verification stage was not run

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# iverilog error line: "filename.v:27: some message"
_IVERILOG_LINE_RE = re.compile(
    r'^(?P<file>[^\s:]+\.v(?:h)?):(?P<line>\d+):\s*(?P<message>.+)$'
)

# Testbench attempt header: "[tb] attempt N: rejected - it does not compile:"
_TB_HEADER_RE = re.compile(
    r'\[tb\].*attempt\s+\d+.*rejected.*compile',
    re.IGNORECASE,
)

# Simulation FAIL line: "FAIL: 1 + 2 = 31, expected 3"
_FAIL_RE = re.compile(
    r'FAIL[:\s]+(.+?)\s*=\s*(\S+),\s*expected\s*(\S+)',
    re.IGNORECASE,
)

# Environment-error patterns
_ENV_PATTERNS = [
    re.compile(r'timed out',        re.IGNORECASE),
    re.compile(r'command not found',re.IGNORECASE),
    re.compile(r'not recognized',   re.IGNORECASE),
    re.compile(r'no such file',     re.IGNORECASE),
    re.compile(r'wsl.*error',       re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Raw compiler text parser
# ---------------------------------------------------------------------------

def parse_raw_compiler_text(raw_text: str) -> dict[str, Any]:
    """
    Parse freeform compiler output (e.g. from a teammate's build system) into
    a structured dict.

    Handles blocks like:
        [tb] attempt 1: rejected - it does not compile:
        tb.v:27: syntax error
        tb.v:27: error: Malformed statement

    Returns
    -------
    {
        "is_testbench_error": bool,
        "source_file":        str | None,   # dominant filename seen in errors
        "error_lines":        list[{"file", "line", "message"}],
        "raw_errors":         list[str],    # every non-blank, non-header line
    }
    """
    lines = raw_text.splitlines()

    is_tb_error = False
    error_lines: list[dict] = []
    raw_errors:  list[str]  = []
    files_seen:  list[str]  = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect the [tb] header — marks this whole block as a testbench error
        if _TB_HEADER_RE.search(stripped):
            is_tb_error = True
            # Don't add the header itself to raw_errors
            continue

        raw_errors.append(stripped)

        m = _IVERILOG_LINE_RE.match(stripped)
        if m:
            fname = m.group("file")
            files_seen.append(fname)
            error_lines.append({
                "file":    fname,
                "line":    int(m.group("line")),
                "message": m.group("message").strip(),
            })

    # Pick the dominant source file (most frequently mentioned)
    source_file: str | None = None
    if files_seen:
        source_file = max(set(files_seen), key=files_seen.count)

    return {
        "is_testbench_error": is_tb_error,
        "source_file":        source_file,
        "error_lines":        error_lines,
        "raw_errors":         raw_errors,
    }


# ---------------------------------------------------------------------------
# Source-file classifier (for checker.py list-of-strings errors)
# ---------------------------------------------------------------------------

# Filenames that indicate a testbench
_TB_FILE_RE = re.compile(r'\btb[\w_]*\.v\b|\btestbench[\w_]*\.v\b', re.IGNORECASE)
# Filenames that indicate a generated design
_DESIGN_FILE_RE = re.compile(r'\bdesign[\w_]*\.v\b|\badder[\w_]*\.v\b', re.IGNORECASE)

def _classify_compile_source(errors: list[str]) -> str:
    """
    Inspect filenames in error messages and return 'testbench' or 'design'.
    Falls back to 'design' when ambiguous (design agent is the repair target).
    """
    joined = "\n".join(errors)
    has_tb     = bool(_TB_FILE_RE.search(joined))
    has_design = bool(_DESIGN_FILE_RE.search(joined))

    if has_tb and not has_design:
        return "testbench"
    # If both appear, or only design, or neither → direct to design agent
    return "design"


def _extract_error_lines(errors: list[str]) -> list[dict[str, Any]]:
    """Parse iverilog-style 'file:line: message' entries from a list of strings."""
    result = []
    for line in errors:
        m = _IVERILOG_LINE_RE.match(line.strip())
        if m:
            result.append({
                "file":    m.group("file"),
                "line":    int(m.group("line")),
                "message": m.group("message").strip(),
            })
    return result


# ---------------------------------------------------------------------------
# Environment-error detector
# ---------------------------------------------------------------------------

def _is_env_error(messages: list[str]) -> bool:
    joined = " ".join(messages)
    return any(p.search(joined) for p in _ENV_PATTERNS)


# ---------------------------------------------------------------------------
# Skipped-stage helpers
# ---------------------------------------------------------------------------

# Phrases that checker.py writes when a stage was skipped because an earlier
# stage already failed (not a missing-tool situation).
_CASCADE_SKIP_RE = re.compile(
    r'skipped due to (compile|simulation|earlier) fail',
    re.IGNORECASE,
)

def _is_cascade_skip(stage: dict) -> bool:
    """True when a stage was skipped because a prior stage failed."""
    errors = stage.get("errors", [])
    return any(_CASCADE_SKIP_RE.search(e) for e in errors)


def _suggestions_skipped_compile() -> list[str]:
    return [
        "The compilation stage did not run. "
        "Icarus Verilog (iverilog) may not be installed or is not on the PATH. "
        "Install it with: sudo apt-get install -y iverilog  (Debian/Ubuntu) "
        "or the equivalent for your OS, then rerun the checker."
    ]


def _suggestions_skipped_simulate() -> list[str]:
    return [
        "The simulation stage did not run. "
        "The vvp simulator (part of Icarus Verilog) may not be installed or is not on the PATH. "
        "Install Icarus Verilog with: sudo apt-get install -y iverilog  (Debian/Ubuntu) "
        "or the equivalent for your OS, then rerun the checker."
    ]


# Patterns that confirm yosys itself is absent
_YOSYS_MISSING_RE = re.compile(
    r'yosys\b.*(not found|not installed|missing|command not found)|'
    r'(not found|not installed|missing).*\byosys\b',
    re.IGNORECASE,
)

# Patterns that indicate synthesis was deliberately disabled by configuration
_SYNTH_DISABLED_RE = re.compile(
    r'disabled|turned off|synthesis.*off|skip.*config|config.*skip',
    re.IGNORECASE,
)


def _classify_skip_note(note: str) -> str:
    """
    Inspect the synthesize.note string and return one of:
      "tool_missing"  — note clearly says yosys is absent
      "disabled"      — note says synthesis was disabled by configuration
      "unknown"       — note is present but doesn't match either pattern,
                        or note is empty
    """
    if not note:
        return "unknown"
    if _YOSYS_MISSING_RE.search(note):
        return "tool_missing"
    if _SYNTH_DISABLED_RE.search(note):
        return "disabled"
    return "unknown"


def _suggestions_skipped_synthesize(skip_reason: str) -> list[str]:
    """
    Return a suggestion list for a skipped synthesis stage.

    Parameters
    ----------
    skip_reason : str
        One of "tool_missing", "disabled", or "unknown".
    """
    if skip_reason == "tool_missing":
        return [
            "The synthesis stage was skipped because Yosys is not installed. "
            "Install it with: sudo apt-get install -y yosys  (Debian/Ubuntu) "
            "or the equivalent for your OS, then rerun the checker. "
            "Synthesis must pass before the design is considered complete."
        ]
    if skip_reason == "disabled":
        return [
            "The synthesis stage was skipped because it is disabled by configuration. "
            "Enable synthesis in the checker configuration and rerun. "
            "Synthesis must pass before the design is considered complete. "
            "Do not assume Yosys is missing — the tool may already be installed."
        ]
    # "unknown"
    return [
        "The synthesis stage was skipped for an unknown reason. "
        "Check that Yosys is installed and accessible, then rerun the checker. "
        "Do not assume Yosys is missing — verify the tool is present before "
        "attempting an installation."
    ]


# ---------------------------------------------------------------------------
# Per-stage suggestion builders
# ---------------------------------------------------------------------------

def _suggestions_compile(errors: list[str], retry_target: str) -> list[str]:
    suggestions = []
    joined = "\n".join(errors)

    if retry_target == "testbench":
        suggestions.append(
            "The testbench file failed to compile. "
            "Inspect the lines listed in 'error_lines' — do not modify the generated design. "
            "Fix the testbench syntax and resubmit."
        )
    
    if re.search(r"syntax error", joined, re.IGNORECASE):
        suggestions.append(
            "A syntax error was reported. Without seeing the source file it is not "
            "possible to pinpoint the exact cause. Check for missing semicolons, "
            "unmatched begin/end or module/endmodule pairs, and malformed statements "
            "near the line numbers in 'error_lines'."
        )
    if re.search(r"malformed statement", joined, re.IGNORECASE):
        suggestions.append(
            "A 'Malformed statement' error typically means a statement is incomplete "
            "or uses syntax iverilog does not recognise at that point. "
            "Check the token immediately before the reported line for a missing "
            "semicolon, mismatched parenthesis, or unsupported construct."
        )
    if re.search(r"undeclared|unknown identifier|not declared", joined, re.IGNORECASE):
        suggestions.append(
            "One or more identifiers are undeclared. Ensure every wire, reg, and "
            "parameter is declared before use and that module port names match exactly."
        )
    if re.search(r"\bport\b", joined, re.IGNORECASE):
        suggestions.append(
            "A port-related error was detected. Verify the module port list matches "
            "the instantiation (name, direction, and width)."
        )
    if re.search(r"width|size|bit", joined, re.IGNORECASE):
        suggestions.append(
            "A bit-width mismatch was reported. Check that output/input port widths "
            "agree between the module definition and its instantiation."
        )
    if not suggestions:
        first = errors[0] if errors else "unknown compile error"
        suggestions.append(
            f"Compilation failed. Address the following iverilog error: {first!r}"
        )
    return suggestions


def _suggestions_simulate(errors: list[str], failing_cases: list[dict]) -> list[str]:
    suggestions = []

    if failing_cases:
        n = len(failing_cases)
        suggestions.append(
            f"{n} test case(s) produced wrong output. "
            "Review the 'failing_cases' list for exact input/got/expected values."
        )

        # Carry/overflow hint
        for c in failing_cases:
            try:
                got = int(c["got"])
                exp = int(c["expected"])
                if got > exp and (got - exp) in (16, 32, 64, 128, 256):
                    suggestions.append(
                        "Some results are larger than expected by a power of 2. "
                        "Check that the output port is wide enough to hold the carry bit."
                    )
                    break
                if got < exp and (exp - got) in (16, 32, 64, 128, 256):
                    suggestions.append(
                        "Some results are smaller than expected by a power of 2. "
                        "A possible overflow/truncation — widen the output port."
                    )
                    break
            except ValueError:
                pass

        # Subtraction/wrap hint
        for c in failing_cases:
            try:
                got = int(c["got"])
                exp = int(c["expected"])
                if exp > 0 and got > exp * 1.5:
                    suggestions.append(
                        "Got values are much larger than expected, which can indicate "
                        "subtraction instead of addition (two's-complement wrap-around). "
                        "Confirm the arithmetic operator in your assign/always block."
                    )
                    break
            except ValueError:
                pass
    else:
        first = errors[0] if errors else "unknown simulation error"
        suggestions.append(
            f"Simulation failed with: {first!r}. "
            "Check that the testbench's $fatal/$finish calls are reachable and "
            "that there are no infinite loops or unresolved signals."
        )
    return suggestions


def _suggestions_synthesize(errors: list[str]) -> list[str]:
    suggestions = []
    joined = "\n".join(errors)

    if re.search(r"initial|always.*@\s*\*", joined, re.IGNORECASE):
        suggestions.append(
            "Yosys flagged a construct that is not synthesizable (e.g. 'initial' block "
            "or an unsupported 'always' sensitivity list). "
            "Remove simulation-only blocks and use only synthesizable RTL."
        )
    if re.search(r"hierarchy|module.*not found|no such module", joined, re.IGNORECASE):
        suggestions.append(
            "Yosys could not resolve the module hierarchy. "
            "Ensure the top-level module name matches the --top argument and that "
            "all sub-modules are included in the file list."
        )
    if re.search(r"multiple driver|multi.driver", joined, re.IGNORECASE):
        suggestions.append(
            "Multiple drivers were detected on the same net. "
            "Ensure each wire/reg is driven by exactly one always block or assign statement."
        )
    if not suggestions:
        first = errors[0] if errors else "unknown synthesis error"
        suggestions.append(
            f"Synthesis failed. Address the following Yosys error: {first!r}"
        )
    return suggestions


def _suggestions_environment(errors: list[str]) -> list[str]:
    joined = " ".join(errors)
    if re.search(r"timed out", joined, re.IGNORECASE):
        return [
            "A tool timed out. The design may contain an infinite loop or unresolved "
            "clock/reset logic that prevents simulation from terminating. "
            "Check all always blocks and ensure the testbench calls $finish or $fatal."
        ]
    if re.search(r"iverilog|vvp|yosys", joined, re.IGNORECASE):
        return [
            "An EDA tool was not found. Confirm that iverilog/vvp/yosys are installed "
            "and accessible in the execution environment."
        ]
    return [
        "An environment or tool error occurred. Check the raw_errors list for details "
        "and verify the EDA toolchain is correctly installed."
    ]


# ---------------------------------------------------------------------------
# Internal result builder — keeps both entry points DRY
# ---------------------------------------------------------------------------

def _build_result(
    passed: bool,
    error_type: str,
    raw_errors: list[str],
    failing_cases: list[dict],
    suggestions: list[str],
    source_file: str | None = None,
    error_lines: list[dict] | None = None,
    retry_target: str | None = None,
) -> dict[str, Any]:
    return {
        "passed":        passed,
        "error_type":    error_type,
        "raw_errors":    raw_errors,
        "failing_cases": failing_cases,
        "source_file":   source_file,
        "error_lines":   error_lines if error_lines is not None else [],
        "retry_target":  retry_target,
        "suggestions":   suggestions,
    }


# ---------------------------------------------------------------------------
# Public entry point 1: structured checker.py dict
# ---------------------------------------------------------------------------

def generate_feedback(checker_result: dict[str, Any]) -> dict[str, Any]:
    """
    Interpret a checker.py result dict and return structured feedback.

    Parameters
    ----------
    checker_result : dict
        The dict returned by checker.check() — must contain keys:
        "passed", "compile", "simulate", "synthesize".

    Returns
    -------
    dict  (see module docstring for full schema)

    Notes
    -----
    passed=True is only returned when ALL three stages (compile, simulate,
    synthesize) explicitly passed.  A skipped stage always produces
    passed=False even if checker.py itself set passed=True.
    """
    compile_stage  = checker_result.get("compile",    {})
    sim_stage      = checker_result.get("simulate",   {})
    synth_stage    = checker_result.get("synthesize", {})

    compile_ok    = bool(compile_stage.get("ok",   False))
    sim_ok        = bool(sim_stage.get("ok",        False))
    synth_ok      = bool(synth_stage.get("ok",      False))
    synth_skipped = bool(synth_stage.get("skipped", False))

    # A stage is "present" (ran and produced a result) when its dict is
    # non-empty.  An absent stage dict means the checker never attempted it.
    compile_present = bool(compile_stage)
    sim_present     = bool(sim_stage)
    synth_present   = bool(synth_stage)

    compile_errors = compile_stage.get("errors", [])
    sim_errors     = sim_stage.get("errors",     [])
    synth_errors   = synth_stage.get("errors",   [])

    # ------------------------------------------------------------------ #
    # True pass: all three stages present and explicitly ok.              #
    # synth_skipped is NOT a pass — yosys was not available.              #
    # ------------------------------------------------------------------ #
    all_passed = (
        compile_present and compile_ok
        and sim_present and sim_ok
        and synth_present and synth_ok and not synth_skipped
    )
    if all_passed:
        return _build_result(
            passed=True,
            error_type=ET_NONE,
            raw_errors=[],
            failing_cases=[],
            suggestions=["All checks passed. The design is correct and ready."],
        )

    # ------------------------------------------------------------------ #
    # Compile stage                                                        #
    # ------------------------------------------------------------------ #

    # Stage entirely absent
    if not compile_present:
        return _build_result(
            passed=False,
            error_type=ET_SKIPPED,
            raw_errors=["Compilation stage was not run."],
            failing_cases=[],
            suggestions=_suggestions_skipped_compile(),
        )

    # Stage ran but failed
    if not compile_ok:
        if _is_env_error(compile_errors):
            return _build_result(
                passed=False,
                error_type=ET_ENVIRONMENT,
                raw_errors=compile_errors,
                failing_cases=[],
                suggestions=_suggestions_environment(compile_errors),
            )
        retry_target = _classify_compile_source(compile_errors)
        error_type   = ET_TB_COMPILE if retry_target == "testbench" else ET_SYNTAX
        error_lines  = _extract_error_lines(compile_errors)
        source_file  = error_lines[0]["file"] if error_lines else None
        return _build_result(
            passed=False,
            error_type=error_type,
            raw_errors=compile_errors,
            failing_cases=[],
            suggestions=_suggestions_compile(compile_errors, retry_target),
            source_file=source_file,
            error_lines=error_lines,
            retry_target=retry_target,
        )

    # ------------------------------------------------------------------ #
    # Simulation stage                                                     #
    # ------------------------------------------------------------------ #

    # Stage entirely absent
    if not sim_present:
        return _build_result(
            passed=False,
            error_type=ET_SKIPPED,
            raw_errors=["Simulation stage was not run."],
            failing_cases=[],
            suggestions=_suggestions_skipped_simulate(),
        )

    # Stage ran but failed
    if not sim_ok:
        # Cascade skip: compile already failed and sim was never attempted
        if _is_cascade_skip(sim_stage):
            return _build_result(
                passed=False,
                error_type=ET_SKIPPED,
                raw_errors=sim_errors,
                failing_cases=[],
                suggestions=[
                    "Simulation was skipped because compilation failed. "
                    "Fix the compilation errors first, then rerun the full check."
                ],
            )
        if _is_env_error(sim_errors):
            return _build_result(
                passed=False,
                error_type=ET_ENVIRONMENT,
                raw_errors=sim_errors,
                failing_cases=[],
                suggestions=_suggestions_environment(sim_errors),
            )
        failing_cases = _parse_failing_cases(sim_errors)
        return _build_result(
            passed=False,
            error_type=ET_FUNCTIONAL,
            raw_errors=sim_errors,
            failing_cases=failing_cases,
            suggestions=_suggestions_simulate(sim_errors, failing_cases),
            retry_target="design",
        )

    # ------------------------------------------------------------------ #
    # Synthesis stage                                                      #
    # ------------------------------------------------------------------ #

    # Stage entirely absent
    if not synth_present:
        return _build_result(
            passed=False,
            error_type=ET_SKIPPED,
            raw_errors=["Synthesis stage was not run."],
            failing_cases=[],
            suggestions=_suggestions_skipped_synthesize(skip_reason="unknown"),
        )

    # Stage was explicitly skipped (yosys absent or disabled)
    if synth_skipped:
        note = synth_stage.get("note", "")
        skip_reason = _classify_skip_note(note)
        return _build_result(
            passed=False,
            error_type=ET_SKIPPED,
            raw_errors=[note] if note else ["Synthesis stage was skipped."],
            failing_cases=[],
            suggestions=_suggestions_skipped_synthesize(skip_reason=skip_reason),
        )

    # Stage ran but yosys found errors
    if not synth_ok:
        if _is_env_error(synth_errors):
            return _build_result(
                passed=False,
                error_type=ET_ENVIRONMENT,
                raw_errors=synth_errors,
                failing_cases=[],
                suggestions=_suggestions_environment(synth_errors),
            )
        return _build_result(
            passed=False,
            error_type=ET_SYNTHESIS,
            raw_errors=synth_errors,
            failing_cases=[],
            suggestions=_suggestions_synthesize(synth_errors),
            retry_target="design",
        )

    # ------------------------------------------------------------------ #
    # Edge case: all stages present and ok, but overall passed=False      #
    # ------------------------------------------------------------------ #
    return _build_result(
        passed=False,
        error_type=ET_ENVIRONMENT,
        raw_errors=["checker reported failure but no stage errors were found"],
        failing_cases=[],
        suggestions=[
            "The overall result is 'failed' but no specific errors were captured. "
            "Re-run the checker with verbose logging to diagnose the issue."
        ],
    )


# ---------------------------------------------------------------------------
# Public entry point 2: raw compiler text from a teammate's build system
# ---------------------------------------------------------------------------

def generate_feedback_from_text(raw_text: str) -> dict[str, Any]:
    """
    Parse freeform compiler output text and return structured feedback.

    Handles output such as:
        [tb] attempt 1: rejected - it does not compile:
        tb.v:27: syntax error
        tb.v:27: error: Malformed statement

    Parameters
    ----------
    raw_text : str
        The raw stdout/stderr string from the external build system.

    Returns
    -------
    dict  (same schema as generate_feedback)
    """
    parsed = parse_raw_compiler_text(raw_text)

    is_tb      = parsed["is_testbench_error"]
    raw_errors = parsed["raw_errors"]
    error_lines = parsed["error_lines"]
    source_file = parsed["source_file"]

    if _is_env_error(raw_errors):
        return _build_result(
            passed=False,
            error_type=ET_ENVIRONMENT,
            raw_errors=raw_errors,
            failing_cases=[],
            suggestions=_suggestions_environment(raw_errors),
            source_file=source_file,
            error_lines=error_lines,
        )

    if is_tb:
        retry_target = "testbench"
        error_type   = ET_TB_COMPILE
    else:
        retry_target = _classify_compile_source(raw_errors)
        error_type   = ET_TB_COMPILE if retry_target == "testbench" else ET_SYNTAX

    return _build_result(
        passed=False,
        error_type=error_type,
        raw_errors=raw_errors,
        failing_cases=[],
        suggestions=_suggestions_compile(raw_errors, retry_target),
        source_file=source_file,
        error_lines=error_lines,
        retry_target=retry_target,
    )


# ---------------------------------------------------------------------------
# Simulation failure line parser (internal)
# ---------------------------------------------------------------------------

def _parse_failing_cases(errors: list[str]) -> list[dict[str, str]]:
    cases = []
    for line in errors:
        m = _FAIL_RE.search(line)
        if m:
            cases.append({
                "inputs":   m.group(1).strip(),
                "got":      m.group(2).strip(),
                "expected": m.group(3).strip(),
            })
    return cases


# ---------------------------------------------------------------------------
# Teammate JSON format adapter
# ---------------------------------------------------------------------------

# Matches: "MISMATCH bin=3 expected gray=2 got gray=3"
# Groups:  (1) all input key=value tokens before "expected"
#          (2) the expected key=value token
#          (3) the got key=value token
_MISMATCH_RE = re.compile(
    r'MISMATCH\s+'
    r'(?P<inputs>(?:\w+=\S+\s+)+?)'      # one or more key=val pairs (inputs)
    r'expected\s+(?P<expected>\w+=\S+)'  # expected key=val
    r'\s+got\s+(?P<got>\w+=\S+)',        # got key=val
    re.IGNORECASE,
)

# Prefix that marks a compile-error block in the teammate's "result" field
_COMPILE_ERROR_PREFIX = re.compile(r'^COMPILE\s+ERROR\s*:', re.IGNORECASE)

# Timeout marker in the "result" field
_TIMEOUT_RE = re.compile(r'SIMULATION\s+TIMEOUT', re.IGNORECASE)


def _parse_mismatch_lines(mismatches: list[str]) -> list[dict[str, str]]:
    """
    Convert teammate MISMATCH lines into the standard failing_cases format.

    Input:  ["MISMATCH bin=3 expected gray=2 got gray=3", ...]
    Output: [{"inputs": "bin=3", "expected": "gray=2", "got": "gray=3"}, ...]
    """
    cases = []
    for line in mismatches:
        m = _MISMATCH_RE.search(line)
        if m:
            cases.append({
                "inputs":   m.group("inputs").strip(),
                "expected": m.group("expected").strip(),
                "got":      m.group("got").strip(),
            })
    return cases


def _analyze_design_for_compile(
    design_src: str,
    error_lines: list[dict],
) -> list[str]:
    """
    Inspect the design source for evidence that explains a compile error.

    Currently detects:
    - A missing semicolon on the statement immediately before ``endmodule``.
      iverilog reports this as a syntax error on or near the ``endmodule`` line.

    Parameters
    ----------
    design_src : str
        Full Verilog source text of the generated design.
    error_lines : list[dict]
        Parsed error-line dicts (file, line, message) from the compile error.

    Returns
    -------
    list[str]
        Zero or more specific suggestion strings grounded in the source.
        Empty when no evidence is strong enough to make a confident claim.
    """
    if not design_src or not error_lines:
        return []

    suggestions = []
    src_lines = design_src.splitlines()

    # Collect all 1-based line numbers mentioned in compiler errors
    error_linenos = {el["line"] for el in error_lines}

    # Check every error line: if the source line just before it (or the error
    # line itself) is a non-empty statement that does not end with ';', ')',
    # '{', or a block keyword, and the next non-blank line is 'endmodule',
    # report a missing semicolon.
    _BLOCK_END_RE  = re.compile(r'^\s*(end|endmodule|endcase|endfunction|endtask)\b', re.IGNORECASE)
    _BLOCK_START_RE = re.compile(r'\b(begin|end|module|endmodule|always|initial|if|else|case)\b', re.IGNORECASE)
    _ENDS_WITH_SEMI = re.compile(r';\s*(?://.*)?$')

    for el in error_lines:
        lineno = el["line"]  # 1-based

        # Look at the source line at and just before the error
        for candidate_1based in (lineno - 1, lineno):
            idx = candidate_1based - 1  # convert to 0-based
            if idx < 0 or idx >= len(src_lines):
                continue
            src_line = src_lines[idx].rstrip()
            if not src_line.strip():
                continue
            # Skip pure block keyword lines — they don't take semicolons
            if _BLOCK_START_RE.match(src_line.strip()):
                continue
            # Check: looks like a statement AND does not end with semicolon
            if not _ENDS_WITH_SEMI.search(src_line):
                # Confirm something that looks like `endmodule` follows nearby
                for lookahead in src_lines[idx + 1: idx + 4]:
                    if re.match(r'\s*endmodule\b', lookahead, re.IGNORECASE):
                        suggestions.append(
                            f"Line {candidate_1based} appears to be missing a "
                            f"semicolon: {src_line.strip()!r}. "
                            "Add ';' at the end of this statement."
                        )
                        break
                if suggestions:
                    break  # one finding is enough
        if suggestions:
            break

    return suggestions


# Matches an assign statement that uses | but not ^ on the RHS.
# We only flag lines where | appears and ^ does not, since ^ is the correct
# Gray-code operator and | would produce wrong results.
_ASSIGN_OR_NOT_XOR_RE = re.compile(
    r'^\s*assign\b[^;]*\|[^;^]*;',   # assign ... | ... ; with no ^ on same line
    re.IGNORECASE,
)

# Recognises that the spec describes a Gray-code conversion
_GRAY_SPEC_RE = re.compile(r'\bgray\b', re.IGNORECASE)


def _analyze_design_for_functional(
    design_src: str,
    spec: str,
    failing_cases: list[dict],
) -> list[str]:
    """
    Inspect the design source for evidence that explains functional mismatches.

    Currently detects (only for Gray-code modules):
    - ``assign`` statements that use bitwise OR (``|``) where XOR (``^``)
      is expected for a binary-to-Gray conversion.

    The analysis is intentionally narrow: it only runs when the spec
    explicitly mentions "gray" and only reports lines where ``|`` is present
    and ``^`` is absent on the same assign statement.  No suggestion is
    produced for unrelated modules.

    Parameters
    ----------
    design_src : str
        Full Verilog source text of the generated design.
    spec : str
        The natural-language specification string (e.g. "4-bit binary to gray
        code converter").
    failing_cases : list[dict]
        Parsed mismatch dicts (inputs, expected, got).

    Returns
    -------
    list[str]
        Zero or more specific suggestion strings grounded in the source.
    """
    if not design_src:
        return []

    # Only run Gray-code analysis when the spec says so
    if not _GRAY_SPEC_RE.search(spec):
        return []

    src_lines = design_src.splitlines()
    offending: list[tuple[int, str]] = []   # (1-based line number, stripped text)

    for i, line in enumerate(src_lines, start=1):
        if _ASSIGN_OR_NOT_XOR_RE.match(line):
            offending.append((i, line.strip()))

    if not offending:
        return []

    lines_desc = "; ".join(f"line {ln}: {txt!r}" for ln, txt in offending)
    return [
        f"The following assign statement(s) use bitwise OR ('|') where XOR ('^') "
        f"is expected for a binary-to-Gray conversion: {lines_desc}. "
        "Replace '|' with '^' in these statements."
    ]


def _suggestions_mismatch(failing_cases: list[dict], name: str, spec: str) -> list[str]:
    """Produce targeted suggestions for functional mismatches from the teammate format."""
    suggestions = []
    n = len(failing_cases)

    if n:
        suggestions.append(
            f"{n} mismatch(es) detected for module '{name}' ({spec}). "
            "Review the 'failing_cases' list for exact input/expected/got values "
            "and correct the logic in the generated design."
        )
    else:
        suggestions.append(
            f"The result field indicates failure for module '{name}' but no "
            "individual mismatches were provided. Review the full result string "
            "in 'raw_errors' for details."
        )

    # Gray-code specific hint: XOR shift pattern
    if any("gray" in c.get("expected", "").lower() or
           "gray" in c.get("got", "").lower()
           for c in failing_cases):
        suggestions.append(
            "Gray code outputs are wrong. Verify the conversion formula: "
            "for a binary input B, the gray code G satisfies G[i] = B[i] XOR B[i+1] "
            "(with B[MSB+1] = 0). A common mistake is using the wrong XOR shift."
        )

    return suggestions


def _suggestions_compile_from_file(error_lines: list[dict], retry_target: str) -> list[str]:
    """Suggestions for compile errors originating from the teammate's JSON format."""
    suggestions = []
    if retry_target == "design":
        suggestions.append(
            "The design file failed to compile. "
            "Fix the syntax errors listed in 'error_lines' and resubmit the design."
        )
    # Delegate to the standard compile suggestion builder for pattern-matched hints
    raw = [f"{el['file']}:{el['line']}: {el['message']}" for el in error_lines]
    if not raw and retry_target == "design":
        return suggestions  # already have one generic suggestion
    suggestions.extend(_suggestions_compile(raw, retry_target))
    return suggestions


def _validate_teammate_data(data: dict[str, Any]) -> None:
    """
    Validate the fields of a teammate evaluation dict before processing.

    Raises
    ------
    ValueError
        With a descriptive message for each of:
        - "result" is not a str
        - "mismatches" is not a list, or contains non-str elements
        - "score" is not a real number, is not finite, or is outside [0, 1]
        - "score" is 1.0 but "mismatches" is non-empty (contradictory)
    """
    import math

    # --- design ---
    design = data.get("design")
    if design is not None and not isinstance(design, str):
        raise ValueError(
            f"'design' must be a string, got {type(design).__name__!r}"
        )

    # --- spec ---
    spec = data.get("spec")
    if spec is not None and not isinstance(spec, str):
        raise ValueError(
            f"'spec' must be a string, got {type(spec).__name__!r}"
        )

    # --- testbench ---
    testbench = data.get("testbench")
    if testbench is not None and not isinstance(testbench, str):
        raise ValueError(
            f"'testbench' must be a string, got {type(testbench).__name__!r}"
        )

    # --- result ---
    result = data.get("result", "")
    if not isinstance(result, str):
        raise ValueError(
            f"'result' must be a string, got {type(result).__name__!r}: {result!r}"
        )

    # --- mismatches ---
    mismatches = data.get("mismatches")
    if mismatches is not None:
        if not isinstance(mismatches, list):
            raise ValueError(
                f"'mismatches' must be a list, got {type(mismatches).__name__!r}"
            )
        for i, item in enumerate(mismatches):
            if not isinstance(item, str):
                raise ValueError(
                    f"'mismatches[{i}]' must be a string, "
                    f"got {type(item).__name__!r}: {item!r}"
                )

    # --- score ---
    score = data.get("score")
    if score is not None:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(
                f"'score' must be a number, got {type(score).__name__!r}: {score!r}"
            )
        if not math.isfinite(score):
            raise ValueError(
                f"'score' must be a finite number, got {score!r}"
            )
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"'score' must be between 0 and 1 inclusive, got {score!r}"
            )

    # --- contradictory: score=1.0 with non-empty mismatches ---
    if score is not None and float(score) == 1.0:
        effective_mismatches = mismatches if mismatches is not None else []
        if effective_mismatches:
            raise ValueError(
                f"Contradictory input: 'score' is 1.0 but 'mismatches' contains "
                f"{len(effective_mismatches)} item(s). "
                "A perfect score is inconsistent with reported mismatches."
            )


def adapt_teammate_json(data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a teammate evaluation JSON dict into the standard feedback output
    schema.  This is the pure transformation — no file I/O.

    Parameters
    ----------
    data : dict
        Must contain at least "result".  Optional keys: "name", "spec",
        "design", "score", "mismatches".

    Returns
    -------
    dict  (standard feedback schema — same as generate_feedback output)

    Notes
    -----
    passed is always False because the teammate format does not include a
    synthesis result, and synthesis must pass for the design to be complete.
    Even score=1.0 with no mismatches only confirms functional correctness;
    it does not confirm synthesizability.
    """
    _validate_teammate_data(data)

    result_str: str   = data.get("result",     "")
    mismatches: list  = data.get("mismatches", []) or []
    name:       str   = data.get("name",       "unknown")
    spec:       str   = data.get("spec",       "")
    design_src: str   = data.get("design",     "") or ""
    score:      float = float(data.get("score", 0.0))

    # ------------------------------------------------------------------ #
    # Case 1: Simulation timeout                                           #
    # ------------------------------------------------------------------ #
    if _TIMEOUT_RE.search(result_str):
        return _build_result(
            passed=False,
            error_type=ET_ENVIRONMENT,
            raw_errors=[result_str.strip()],
            failing_cases=[],
            suggestions=[
                f"Simulation timed out for module '{name}'. "
                "The design likely contains an infinite loop or a clock/reset "
                "condition that never terminates. "
                "Check all always blocks and ensure every loop has a reachable exit."
            ],
            retry_target="design",
        )

    # ------------------------------------------------------------------ #
    # Case 2: Compile error                                                #
    # ------------------------------------------------------------------ #
    if _COMPILE_ERROR_PREFIX.search(result_str):
        # Strip the "COMPILE ERROR:" header line and pass the rest as raw text
        body = _COMPILE_ERROR_PREFIX.sub("", result_str, count=1).strip()
        error_lines_parsed = _extract_error_lines(body.splitlines())
        raw_errors = [line.strip() for line in body.splitlines() if line.strip()]

        # dut.v → design; tb.v / testbench → testbench; fallback → design
        retry_target = _classify_compile_source(raw_errors)
        error_type   = ET_TB_COMPILE if retry_target == "testbench" else ET_SYNTAX
        source_file  = error_lines_parsed[0]["file"] if error_lines_parsed else None

        # Build suggestions: source-grounded findings first, then generic fallback
        src_suggestions = []
        if retry_target == "design" and design_src:
            src_suggestions = _analyze_design_for_compile(design_src, error_lines_parsed)
        generic_suggestions = _suggestions_compile_from_file(error_lines_parsed, retry_target)
        # Prepend specific findings; generic suggestions always follow as fallback
        all_suggestions = src_suggestions + generic_suggestions

        return _build_result(
            passed=False,
            error_type=error_type,
            raw_errors=raw_errors,
            failing_cases=[],
            suggestions=all_suggestions,
            source_file=source_file,
            error_lines=error_lines_parsed,
            retry_target=retry_target,
        )

    # ------------------------------------------------------------------ #
    # Case 3: Functional mismatches                                        #
    # ------------------------------------------------------------------ #
    failing_cases = _parse_mismatch_lines(mismatches)
    raw_errors    = mismatches if mismatches else ([result_str.strip()] if result_str.strip() else [])

    # score == 1.0 AND no mismatches → functional tests passed, but synthesis
    # was not checked, so we still return passed=False with a clear note.
    if score == 1.0 and not mismatches:
        return _build_result(
            passed=False,
            error_type=ET_SKIPPED,
            raw_errors=[],
            failing_cases=[],
            suggestions=[
                f"All functional tests passed for module '{name}' (score=1.0). "
                "However, synthesis was not checked in this evaluation format. "
                "Run a synthesis check with Yosys before marking the design complete."
            ],
            retry_target=None,
        )

    return _build_result(
        passed=False,
        error_type=ET_FUNCTIONAL,
        raw_errors=raw_errors,
        failing_cases=failing_cases,
        suggestions=_suggestions_mismatch(failing_cases, name, spec)
                    + _analyze_design_for_functional(design_src, spec, failing_cases),
        retry_target="design",
    )


def generate_feedback_from_file(
    input_path:  str = "for_feedback.json",
    output_path: str = "feedback_result.json",
) -> dict[str, Any]:
    """
    Read a teammate evaluation JSON file, generate structured feedback, and
    write the result to an output JSON file.

    Parameters
    ----------
    input_path : str
        Path to the input JSON file (default: "for_feedback.json").
        Must contain at least a "result" key.  Optional keys: "name", "spec",
        "design", "score", "mismatches".
        The original file is never modified.

    output_path : str
        Path to write the feedback JSON (default: "feedback_result.json").

    Returns
    -------
    dict  (standard feedback schema — same as generate_feedback output)

    Raises
    ------
    ValueError
        If the file cannot be parsed as JSON or is missing required keys.
    OSError
        If the input file cannot be read or the output file cannot be written.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {input_path!r}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{input_path!r} must contain a JSON object (dict), "
            f"got {type(data).__name__}"
        )
    if "result" not in data:
        raise ValueError(f"{input_path!r} is missing the required 'result' key")

    try:
        _validate_teammate_data(data)
    except ValueError as exc:
        raise ValueError(f"Invalid field in {input_path!r}: {exc}") from exc

    feedback = adapt_teammate_json(data)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(feedback, fh, indent=2)

    return feedback
