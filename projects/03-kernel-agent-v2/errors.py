#!/usr/bin/env python3
"""
errors.py — the failure taxonomy, and the verdict-to-instruction translation.

Two jobs, both demanded by the challenge's scoring rubric:

1. THE TAXONOMY. Every failed attempt gets one named failure mode, so a whole campaign of
   runs can be grouped and counted ("which levels did it fail, and WHY"). This is the
   artifact the rubric says it would most want to keep, and the dashboard chart
   footer reads from it.

2. THE TRANSLATION. "The single most valuable thing your agent does" (challenge, measured
   twice in project 02) is converting "this is wrong" into "change exactly this". So every
   failure carries an INSTRUCTION -- imperative, one named change, never the answer -- and
   enrich() upgrades raw exception text with the fix, because a verdict reproduced verbatim
   reproduces the same violation.

Each mode: a one-line description (for the dashboard), a detect() when detection is
mechanical, and instruction fragments used by verifier.py / agent.py.
"""

# --- failure modes -------------------------------------------------------------
# kind: where it is caught. 'static' = rule scan, 'numeric' = numerics check,
#       'runtime' = raised during execution, 'model' = the model's answer itself.

TAXONOMY = {
    # static
    "parse-error":      ("the reply is not parseable Python", "static"),
    "banned-call":      ("calls a function that does the whole operation", "static"),
    "fancy-indexing":   ("boolean-mask or integer-array indexing", "static"),
    "whole-array-op":   ("arithmetic on the whole input instead of per-tile", "static"),
    "no-tile-loop":     ("one tile only -- no explicit loop over tiles", "static"),
    # numeric
    "wrong-shape":      ("output shape does not match the reference", "numeric"),
    "non-finite":       ("NaN or Inf in the output (overflow / uninitialised)", "numeric"),
    "modified-input":   ("wrote into the tensor it was given", "numeric"),
    "ragged-edge":      ("wrong only in the final partial tile", "numeric"),
    "partial-coverage": ("some tiles written, others left zero/untouched", "numeric"),
    "core-arithmetic":  ("wrong across the interior, not at an edge", "numeric"),
    "nondeterministic": ("two runs on the same input disagree", "numeric"),
    # runtime
    "raised":           ("the kernel raised during execution", "runtime"),
    "timeout":          ("the kernel ran past its time budget", "runtime"),
    # model-level (agent.py)
    "empty-answer":     ("no code came back at all", "model"),
    "truncated":        ("finish_reason=length -- the answer was cut off", "model"),
    "no-improvement":   ("same failure repeated; the prompt stopped changing", "model"),
    "confidently-wrong":("reported high confidence and failed verification", "model"),
}


def is_known(label):
    return label in TAXONOMY


# --- exception -> instruction --------------------------------------------------
#
# Measured in project 02 (agent.py enrich(), 15-round loops on single messages): a raw
# exception names the mistake and never the fix. These patterns are NumPy-stage now; the
# NKI ones live in 02's agent.py if Stage B is ever wired in.

def enrich_exception(text):
    """Append the fix to a raw exception message, when the fix is mechanical."""
    t = text or ""

    import re as _re
    m = _re.search(r"name '(\w+)' is not defined", t)
    if m:
        name = m.group(1)
        if name in ("np", "numpy"):
            return t + (" FIX: put `import numpy as np` as the FIRST line of the file. "
                        "Everything else can stay exactly as it is.")
        return t + (f" FIX: `{name}` is used but never defined. Check the spelling, or "
                    f"define it before use.")
    if "only size-1 arrays can be converted to Python scalars" in t or \
       "cannot convert" in t and "scalar" in t:
        return t + (" FIX: a tile is an array, not a number. Keep it an array of the same "
                    "shape as the output slice you are writing into.")
    if "operands could not be broadcast together" in t:
        return t + (" FIX: the two sides have different shapes. Slice both to the SAME "
                    "tile -- a[r0:r0+128, c0:c0+512] on each -- rather than relying on "
                    "broadcasting, which is banned here.")
    if "index " in t and "is out of bounds" in t:
        return t + (" FIX: your loop overruns the array. The final tile is PARTIAL -- clamp "
                    "the slice with min(r0+128, rows) or let the slice cut itself: "
                    "x[r0:r0+128] already stops at the end. Derive bounds from x.shape, "
                    "never from the constant 128.")
    if "too many indices" in t or "too many indices for array" in t:
        return t + (" FIX: you indexed with more subscripts than the array has dimensions. "
                    "Check which axis you are actually reducing over.")
    if "can only annotate" in t or "invalid syntax" in t:
        return t
    if "division by zero" in t or "divide by zero" in t:
        return t + (" FIX: a denominator hit 0. For a norm or variance, add the level's "
                    "epsilon inside the sqrt; for softmax, note the denominator is a sum "
                    "of exponentials and cannot be 0 if you subtracted the row max first.")
    if "maximum recursion depth" in t:
        return t + " FIX: use loops, not recursion -- tiles are iterated with `for`."
    if "Unable to allocate" in t or "MemoryError" in t:
        return t + (" FIX: you allocated something the size of the whole INPUT per "
                    "iteration. Allocate the small tile inside the loop instead.")
    if "object is not callable" in t:
        return t + " FIX: parentheses after a value that is not a function. Check the name."
    if "unsupported operand" in t:
        return t + (" FIX: check dtypes -- float32 tiles and python floats mix fine, but "
                    "None or a tuple on one side means a previous line returned something "
                    "else than you assumed.")
    return t


def instruction_from_failure(failure, case_label=""):
    """The instruction the model receives. One named change, imperative, no answer.

    `failure` is a dict with at least `taxonomy` and `verdict`. The templates say WHAT TO
    CHANGE; the verdict supplies WHERE. Measured: 'line 16: calls banned max' reproduced
    the violation verbatim; 'Replace the np.max call with an explicit python loop over the
    columns' fixed it in one round.
    """
    tax = failure.get("taxonomy", "core-arithmetic")
    v = failure.get("verdict", "")
    loc = f" ({case_label})" if case_label else ""

    # A raised exception whose enrich_exception() found a mechanical FIX: the enriched
    # text IS the instruction (the generic template below says only 'fix the exception',
    # which measured three straight rounds of `NameError: name 'np' is not defined`).
    if tax == "raised":
        enriched = enrich_exception(v)
        if "FIX:" in enriched:
            return f"{enriched}{loc} Change nothing else."

    T = {
        "parse-error":
            "Send ONE complete python code block that parses. No prose, no commentary "
            "outside the block.",
        "banned-call":
            "Replace each banned call named above with an explicit python loop over the "
            "tile's rows or columns that computes the same quantity. Change only that.",
        "fancy-indexing":
            "Replace every boolean-mask or list index above with slice notation and plain "
            "arithmetic on the slice. Slices and arithmetic only.",
        "whole-array-op":
            "Do that arithmetic on a TILE, not on the whole input: slice "
            "x[r0:r0+128, c0:c0+512] inside your loop and operate on the slice.",
        "no-tile-loop":
            "Wrap the work in explicit `for` loops that step across the array in tiles of "
            "at most 128 rows x 512 columns, and compute each tile's piece inside.",
        "wrong-shape":
            "Recompute the output size from the INPUT shape with the formula named above "
            "and allocate the output from that. The arithmetic lines are right; the size "
            "is not.",
        "non-finite":
            "Apply the fix named above (usually: subtract the row max before exp, or "
            "initialise every output tile before reading it). Nothing else is wrong.",
        "modified-input":
            "Allocate a NEW output array with np.empty_like / np.zeros and write your "
            "result there; return that. Never write into the arguments.",
        "ragged-edge":
            "Handle the FINAL PARTIAL tile separately: it is smaller than the others, so "
            "clamp every slice to the array's real shape and do not assume the tile is "
            "full-size.",
        "partial-coverage":
            "Every output tile must be written exactly once, at ITS OWN index "
            "out[r0:r0+128, c0:c0+512]. You are writing some tiles to the wrong place or "
            "not at all; check the loop bounds against the output shape.",
        "core-arithmetic":
            "The formula itself is wrong in the interior. Recompute it for one tile BY "
            "HAND (a SCRATCH: line comparing your expression to the reference on a 3x3 "
            "array is the fastest way to see which term differs), then fix that term.",
        "nondeterministic":
            "Your kernel gives different answers on identical calls: you are reading "
            "uninitialised memory or carrying state in a global. Allocate fresh output "
            "per call.",
        "raised":
            "Fix the exception above, and change nothing else.",
        "timeout":
            "You are doing far more work than the operation needs -- usually a loop over "
            "elements where a loop over tiles belongs, or an accidental O(n^2) over the "
            "whole array. Restructure to one pass per tile.",
        "empty-answer":
            "Reply with one python code block. Do not restate the task.",
        "truncated":
            "Your previous answer was cut off. Write the code SHORTER and plainer, and "
            "send it in one block.",
        "no-improvement":
            "Every approach in the ledger above has failed. Pick a structurally DIFFERENT "
            "one: different loop nesting, or reduce incrementally (compute tile 0, check "
            "it with a SCRATCH: line, then generalise).",
        "confidently-wrong":
            "You marked this high confidence and it is wrong. Verify your index "
            "arithmetic with a SCRATCH: line on a small array before answering again.",
    }
    base = T.get(tax)
    if base is None:  # unknown mode -- degrade to the verdict, never to silence
        base = "Fix what the verdict names."
    return f"{base}{loc} {v}".strip()


def classify_verdict(verdict):
    """Map a verifier verdict string to a taxonomy label. Used when replaying old traces
    and by the selftest to check the classifier stays in step with the verifier."""
    v = verdict or ""
    if "WRONG SHAPE" in v:
        return "wrong-shape"
    if "NON-FINITE" in v:
        return "non-finite"
    if "MODIFIED ITS INPUT" in v:
        return "modified-input"
    if "ragged edge" in v or "ragged-edge" in v:
        return "ragged-edge"
    if "% ZERO" in v or "of the output is zero" in v or "is zero" in v and "block" in v:
        return "partial-coverage"
    if "most elements are wrong" in v or "NUMERICAL MISMATCH" in v:
        return "core-arithmetic"
    if "disagree" in v.lower():
        return "nondeterministic"
    if "RAISED" in v:
        return "raised"
    if "TIMEOUT" in v:
        return "timeout"
    return "core-arithmetic"
