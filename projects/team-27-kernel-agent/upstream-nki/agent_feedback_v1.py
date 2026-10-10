"""Run the existing kernel agent with targeted scalar-multiply repair feedback.

Place this file beside the pod's agent.py. All original CLI options remain valid.
Only missing nki.isa.multiply/scalar_mul errors receive different feedback.
"""

import re
import sys


MISSING_SCALE_API = re.compile(
    r"module ['\"]nki\.isa['\"] has no attribute ['\"](?:multiply|scalar_mul)['\"]"
)

SCALING_HINT = (
    " For elementwise scaling by a constant, use "
    "nisa.tensor_scalar(dst=scaled, data=tile, op0=nl.multiply, operand0=scale). "
    "First allocate scaled = nl.ndarray(tile.shape, dtype=tile.dtype, buffer=nl.sbuf). "
    "Here tile is an existing on-chip tile, scale is a compile-time scalar constant, "
    "and scaled has the same shape as tile. nl.multiply names the operator; "
    "tensor_scalar executes it and writes into dst. Replace the missing call "
    "with this operation, using your own tile and scalar variable names."
)


def install_feedback(base):
    """Change the feedback hook in this process; keep the original handler otherwise."""
    original = base.enrich

    def enrich(error_text):
        if MISSING_SCALE_API.search(error_text):
            return error_text + SCALING_HINT
        return original(error_text)

    base.enrich = enrich
    return original


def check_feedback(base):
    """Check observed error routing and the installed SDK's keyword contract."""
    import inspect
    import nki.isa as nisa
    import nki.language as nl

    original = install_feedback(base)
    try:
        for name in ("multiply", "scalar_mul"):
            error = f"raised AttributeError: module 'nki.isa' has no attribute '{name}'"
            repaired = base.enrich(error)
            assert repaired == error + SCALING_HINT
            print(f"{name}: targeted replacement feedback OK")
        for error in (
            "raised AssertionError: dst must be in ['psum'], got sbuf",
            "raised AttributeError: module 'nki.isa' has no attribute 'unknown_api'",
            "raised AttributeError: module 'nki.language' has no attribute 'multiply'",
        ):
            assert base.enrich(error) == original(error)
        print("Other errors: original feedback preserved")
        inspect.signature(nisa.tensor_scalar).bind(
            dst=None, data=None, op0=nl.multiply, operand0=0.5
        )
        print("Installed tensor_scalar: example keywords accepted")
        print("FEEDBACK CHECK PASSED (routing and signature only; no kernel/model run)")
    finally:
        base.enrich = original


if __name__ == "__main__":
    import agent as base

    if sys.argv[1:] == ["--check-feedback"]:
        check_feedback(base)
    else:
        install_feedback(base)
        print("Feedback experiment v1: targeted scalar-multiply API repair", flush=True)
        base.main()
