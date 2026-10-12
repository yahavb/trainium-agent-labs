"""Prompt templates S1-S5. DESIGN.md 6.5.

Deliberately short and positive. Measured on these seats: a list of prohibitions makes a model audit
itself against each one and return nothing. The rules live in the checker; the prompt asks for the
module, and the repair prompt asks for exactly one named change.
"""
from __future__ import annotations

import re

FENCE = "```"

S1 = """{spec}
Write the complete SystemVerilog module TopModule for this specification.
Reply with one ```verilog code block and nothing else."""

S2 = """{spec}
First write the truth table or state-transition table for TopModule as Verilog comments, then the complete module.
Reply with one ```verilog code block and nothing else."""

S4 = """{spec}
Start from the exact port list above, declare every output with its type, then write the complete module TopModule.
Reply with one ```verilog code block and nothing else."""

S3 = """Specification:
{spec}
This SystemVerilog module TopModule is not correct yet:

```verilog
{code}
```
{ledger}
A checker reports:
{feedback}

Rewrite the complete module TopModule so that what the checker names is fixed. Your module must differ from the code above.
Reply with one ```verilog code block."""

S3_NO_SPEC = """This SystemVerilog module TopModule is not correct yet:

```verilog
{code}
```
{ledger}
A checker reports:
{feedback}

Rewrite the complete module TopModule so that what the checker names is fixed. Your module must differ from the code above.
Reply with one ```verilog code block."""

# D-014: after a repair that handed back the code unchanged, start over from the spec with the
# feedback as a hint instead of asking for another edit of the same code.
S5 = """{spec}
A previous attempt at TopModule failed. A checker reported:
{feedback}
{ledger}
Write the complete SystemVerilog module TopModule for this specification so that this problem does not occur.
Reply with one ```verilog code block and nothing else."""

FRESH = {"S1": S1, "S2": S2, "S4": S4}

# D-016: the ledger. Dev runs C-2 and C-3 showed fixes that do not stick (Prob050_kmap1 went
# 0.82 -> 0.88 -> 0.93 -> 0.87 -> 0.82 -> 0.88): each repair fixed the row it was told about and broke
# one fixed earlier, because the prompt carried only the latest finding. The ledger repeats what
# earlier checks already told the model, so it reveals nothing new. Same mechanism in B and C.
LEDGER_ITEMS = 4
LEDGER_ITEM_CHARS = 240


def ledger_text(earlier) -> str:
    items = [" ".join(m.split())[:LEDGER_ITEM_CHARS] for m in (earlier or []) if m][-LEDGER_ITEMS:]
    if not items:
        return ""
    return ("\nEarlier versions of this module also failed these checks. Your new module must not "
            "fail them again:\n" + "".join(f"- {m}\n" for m in items))


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    return re.sub(r"\n\s*\n+", "\n", code)


def build(strategy: str, problem, code: str = None, feedback: str = None,
          max_prompt_chars: int = 4 * (8192 - 64 - 1000), earlier: list = None) -> tuple:
    """(prompt, note). `note` says what was dropped to fit, or '' -- the agent logs it.

    The budget leaves ~1,000 tokens for the answer at the 4-characters-per-token estimate that
    workers.py uses to cap max_tokens.
    """
    if strategy in FRESH:
        return FRESH[strategy].format(spec=problem.spec), ""
    led = ledger_text(earlier)
    if strategy == "S5":
        hint = feedback or ""
        prompt = S5.format(spec=problem.spec, feedback=hint, ledger=led)
        if len(prompt) <= max_prompt_chars:
            return prompt, ""
        prompt = S5.format(spec=problem.spec, feedback=hint, ledger="")
        if len(prompt) <= max_prompt_chars:
            return prompt, "dropped ledger"
        keep = max(0, len(hint) - (len(prompt) - max_prompt_chars))
        return S5.format(spec=problem.spec, feedback=hint[:keep], ledger=""), "dropped ledger, shortened feedback"
    if strategy != "S3":
        raise ValueError(f"unknown strategy {strategy!r}")
    code = code if code and code.strip() else "// (your previous reply contained no Verilog code)"
    prompt = S3.format(spec=problem.spec, code=code.rstrip(), feedback=feedback or "", ledger=led)
    if len(prompt) <= max_prompt_chars:
        return prompt, ""
    prompt = S3.format(spec=problem.spec, code=code.rstrip(), feedback=feedback or "", ledger="")
    if len(prompt) <= max_prompt_chars:
        return prompt, "dropped ledger"
    prompt = S3_NO_SPEC.format(code=code.rstrip(), feedback=feedback or "", ledger="")
    if len(prompt) <= max_prompt_chars:
        return prompt, "dropped ledger and spec"
    prompt = S3_NO_SPEC.format(code=strip_comments(code).rstrip(), feedback=feedback or "", ledger="")
    return prompt, "dropped ledger, spec and code comments"
