"""The organizers' checker as a final gate (v4, notes E29).

Our harness diagnoses; kernelbench.py is how the work is judged: its signatures, references, cases
and per-element relative tolerance 1e-4. Under it, kernels our float32 bound accepts can fail (float32
products near zero outputs, L7/L10) and our L4/L8 signatures failed outright. A kernel is reported
verified only if it also passes this gate. The file is the organizers', so it is loaded from the seat
pod (KERNELBENCH_DIR) rather than copied; without it the gate is skipped and says so.
"""
import contextlib
import io
import os
import sys

KERNELBENCH_DIR = os.environ.get("KERNELBENCH_DIR", "/workspace/projects/02-kernel-agent")
_kb = None


def available():
    global _kb
    if _kb is None:
        if not os.path.exists(os.path.join(KERNELBENCH_DIR, "kernelbench.py")):
            return False
        sys.path.insert(0, KERNELBENCH_DIR)
        import kernelbench
        _kb = kernelbench
    return True


def check(level_num, source):
    """(passed, message): passed is None when the checker is unavailable (never counts as a pass)."""
    if not available():
        # red team PR #9: passing here made a kernel "verified 0.95" with no final check at all
        return None, "organizers' checker not found: NOT judged"
    ns = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "kernel.py", "exec"), ns)
        res = _kb.verify(ns["kernel"], level_num, stop_early=True)
    except Exception as e:
        return False, f"RAISED: {type(e).__name__}: {e}"
    rules = _kb.check_rules(source, level_num)
    if res["ok"] and not rules:
        return True, f"organizers' checker: {res['passed']}/{res['total']} cases"
    if rules:
        return False, "RULES: " + "; ".join(rules)
    label, msg = res["failures"][0]
    return False, f"{label}: {msg}"
