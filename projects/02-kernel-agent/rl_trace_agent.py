#!/usr/bin/env python3
"""
Verifier-grounded online RL controller for projects/02-kernel-agent  (v5).

Reuses agent.py (prompts, grader) and nkibench.py (verifier) UNCHANGED. Everything new lives here.

What it learns: which prompt strategy and sampling temperature earn the most verifier reward, per
level and per failure category, plus a bank of API facts and lessons the verifier has confirmed. This
is online policy learning over prompts; it does NOT update model weights. It exports group-structured
(prompt, reply, reward, advantage) rows for a later GRPO / rejection-sampling run (--export-groups,
--export-sft).

Run from projects/02-kernel-agent:
  python rl_trace_agent.py --level 2 --rounds 8 --samples 4 --episodes 3 \
      --context 8192 --seed-references --log rl_level2_v5.jsonl

See README-RL-TRACE.md for what changed from v4.1 and why.
"""

import argparse
import ast
import concurrent.futures as cf
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
from collections import defaultdict

import numpy as np

import agent as base_agent
import nkibench

SCHEMA_VERSION = 6
MODEL = os.environ.get("KERNEL_AGENT_MODEL", getattr(base_agent, "MODEL", "Qwen/Qwen3-8B"))

STRATEGIES = ("direct", "contract_trace", "counterexample", "repair_diagnosis", "exemplar", "reflect")
TEMPERATURES = ("t=0.3", "t=0.6", "t=0.9")
MODES = ("reflect", "bandit", "plain")

TRACE_FIELDS = (
    "contract", "tile_strategy", "memory_flow", "edge_cases",
    "predicted_failure", "verification_checks", "diagnosis",
)
TRACE_RE = re.compile(r"<TRACE>\s*(\{.*?\})\s*</TRACE>", re.S)
TRACE_STRIP_RE = re.compile(r"<TRACE>.*?(?:</TRACE>|\Z)", re.S)
FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)
NON_CODE_LANGS = {"json", "text", "txt", "bash", "sh", "shell", "console", "yaml", "yml", "markdown"}

# Failure categories for which re-running the kernel can say WHERE or HOW it is wrong.
LOCATABLE = {"api_signature", "runtime_error", "api_name", "tile_limits", "correctness"}

# ---------------------------------------------------------------------------- prompt cards

FILE_RULES_CARD = r"""
NKI RULES — follow exactly:
- Imports, and only these: `import nki`, `import nki.language as nl`, `import nki.isa as nisa`.
  `nl` is an alias; NEVER write `import nl` or `import nki.nl`.
- `nl.sbuf`, `nl.psum`, `nl.shared_hbm` are memory-region VALUES, not functions.
  Correct: `nl.ndarray(shape=(P, F), dtype=nl.float32, buffer=nl.sbuf)`. Wrong: `nl.sbuf(...)`.
- Do not use NumPy to implement the kernel, and do not apply host-side `.T` to an input.
- Use `@nki.jit` on the required entry point and keep its exact name and signature.
- The checker imports your file and then calls the kernel itself. The file may contain ONLY imports
  and the function definition(s): no kernel call, no `nl.*` or `nisa.*` call at module level, no
  example usage, no `print`, no `if __name__ == "__main__":`.
"""

# Language documentation (what the calls ARE), not an algorithm. The v4.1 trace failed on `nl.ds`
# eight rounds running because the prompt named it and never gave its signature.
API_SIGS_CARD = r"""
CALL SIGNATURES (exact):
- nl.ds(start, size)                   two separate integers; builds a slice. It is used INSIDE an
                                       index and is never called with a tuple: `t[:, nl.ds(j, 8)]`
                                       means every row, columns j .. j+7.
- t[a:b, c:d]  or  t[:, nl.ds(c, n)]   slicing a tile gives a view; pass views as dst= / src=.
- nisa.tensor_copy(dst=, src=)         dst and src must have the same shape. There is NO `ds=`,
                                       `dst_idx=` or `src_idx=` keyword: the slice IS the index.
- nisa.dma_copy(dst=, src=)            HBM <-> SBUF; same number of elements on both sides.
- nl.ndarray(shape, dtype=, buffer=)   allocate; the first dimension of an sbuf/psum tile is <= 128.
- nl.affine_range(n)                   the loop; its variable is usable inside nl.ds(...).
"""

LEVEL2_ALGO_HINT = r"""
LEVEL 2 TRANSPOSE NOTE:
The input is [P, F1*F2]; each row is a flattened row-major F1-by-F2 matrix. The output row is its
transpose, flattened:  output[p, f2*F1 + f1] = input[p, f1*F2 + f2].
Do not call `reshape` or `transpose` on NKI tensors. Load the input into an SBUF tile with
nisa.dma_copy, then in nested nl.affine_range loops copy ONE column at a time with
nisa.tensor_copy(dst=out_tile[:, nl.ds(dst_col, 1)], src=in_tile[:, nl.ds(src_col, 1)]),
then nisa.dma_copy the result to the shared-HBM output.
"""

REFLECT_PROMPT = """REFLECTION TASK. You are reviewing a FAILED attempt at an AWS Neuron NKI kernel
for: {op}. Do NOT write the kernel.

```python
{code}
```

Checker report:
{feedback}

{known}Only use calls that exist: nl.ds(start, size), nisa.tensor_copy(dst=, src=), nisa.dma_copy(dst=, src=),
nl.ndarray, nl.affine_range. Do not invent functions or keyword arguments.
Answer in exactly this plain-text format, under 90 words in total:
CAUSE: <the specific line or construct that makes it fail>
CHANGE: <the exact edit that fixes it, as code in backticks>
RULE: <one general rule for writing NKI kernels, starting with a verb>
"""
REFLECT_RE = {k: re.compile(rf"{k}:\s*(.+?)(?=\n\s*(?:CAUSE|CHANGE|RULE):|\Z)", re.S)
              for k in ("CAUSE", "CHANGE", "RULE")}

FEEDBACK_NOTES = {
    "no_backend": (
        "WHY: something at module level (a call to the kernel, to nl.*, or to nisa.*) ran while "
        "the file was being imported, before the simulator exists. FIX: the file must contain only "
        "imports and the @nki.jit function. Delete every top-level call, example, test, print, and "
        "`if __name__ == '__main__'` block. The checker calls the kernel for you."),
    "parse": (
        "FIX: reply with exactly one ```python fenced block that contains the complete file and "
        "nothing else inside the fence."),
    "truncated": (
        "The previous reply was CUT OFF by the token limit. Write the shortest correct kernel: no "
        "comments, no docstrings, no explanation outside the code block."),
}

# One line of difference per variant. Sample 0 always gets the canonical prompt.
VARIANT_HINTS = (
    "",
    "Variation request: choose the simplest loop structure you can think of.",
    "Variation request: compute the index arithmetic for each copy into named variables first, "
    "and use a different loop order from the most obvious one.",
    "Variation request: use as few nisa calls as you can, and do not follow the most common "
    "pattern.",
)
DUP_NOTE = ("\n\nAnother attempt at this exact prompt produced identical code. Do NOT reproduce it: "
            "change the structure (loop order, how the index is computed, how data is moved).")

REVEAL_FLOW = True      # level 2: tell the model where its output values actually came from


def compact(value, limit=500):
    return str(value or "").strip().replace("\n", " ")[:limit]


# ---------------------------------------------------------------------------- extraction, hygiene

def parse_trace(reply):
    match = TRACE_RE.search(reply or "")
    if not match:
        return {}, False
    try:
        result = json.loads(match.group(1))
        return (result, True) if isinstance(result, dict) else ({}, False)
    except (json.JSONDecodeError, TypeError):
        return {}, False


def _is_json_object(text):
    try:
        return isinstance(json.loads(text), dict)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def extract_kernel_code(reply, entry):
    """Pick the fenced block that is the kernel: trace cut out, JSON/shell blocks ignored, blocks
    ranked by how much they look like the kernel (v3 took the longest block, which could be JSON)."""
    text = TRACE_STRIP_RE.sub("", reply or "")
    scored = []
    for lang, body in FENCE_RE.findall(text):
        body = body.strip()
        if not body or lang.lower() in NON_CODE_LANGS or _is_json_object(body):
            continue
        score = 2 * (entry in body) + ("@nki.jit" in body) + ("import nki" in body)
        scored.append((score, len(body), body))
    code = max(scored)[2] if scored else base_agent.extract_code(text)
    lines = code.splitlines()
    if lines and lines[0].strip().lower() == "python":
        code = "\n".join(lines[1:])
    return code.strip()


def _is_static_expr(node):
    if node is None:
        return True
    allowed = (ast.Constant, ast.Tuple, ast.List, ast.UnaryOp, ast.BinOp, ast.Name, ast.Attribute,
               ast.operator, ast.unaryop, ast.expr_context)
    return all(isinstance(n, allowed) for n in ast.walk(node))


def sanitize_module(source):
    """Make the file safe to import: drop module-level statements that execute at import time and
    remove type annotations (evaluated at import). Returns (source, dropped_descriptions); the source
    is byte-for-byte unchanged when nothing needed changing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    keep, dropped, changed = [], [], False
    for node in tree.body:
        is_docstring = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
        if isinstance(node, ast.AnnAssign) and node.value is not None and _is_static_expr(node.value):
            node = ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)
            changed = True
            keep.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)) or is_docstring:
            keep.append(node)
        elif isinstance(node, ast.Assign) and _is_static_expr(node.value):
            keep.append(node)
        else:
            try:
                shown = ast.unparse(node).replace("\n", " ")[:80]
            except Exception:
                shown = type(node).__name__
            dropped.append(f"line {node.lineno}: `{shown}`")
    module = ast.Module(body=keep, type_ignores=[])
    for fn in ast.walk(module):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = fn.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                if arg is not None and arg.annotation is not None:
                    arg.annotation, changed = None, True
            if fn.returns is not None:
                fn.returns, changed = None, True
    if not dropped and not changed:
        return source, []
    return ast.unparse(ast.fix_missing_locations(module)) + "\n", dropped


def compact_exemplar(source):
    """Shrink a verified kernel for use as a prompt exemplar."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def strip_docstring(body):
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            return body[1:] or [ast.Pass()]
        return body

    tree.body = strip_docstring(tree.body)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.body = strip_docstring(node.body)
        if isinstance(node, ast.ImportFrom) and node.module == "nki.typing":
            continue
        if isinstance(node, ast.Import) and all(a.name == "numpy" and (a.asname or a.name) not in used
                                                for a in node.names):
            continue
        kept.append(node)
    tree.body = kept
    try:
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    except Exception:
        return source


def code_fingerprint(code):
    try:
        norm = ast.dump(ast.parse(code))
    except SyntaxError:
        norm = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------- static API lint
#
# The v4.1 trace: `ds() missing 1 required positional argument: 'size'` came back 40+ times and the
# eight-billion-parameter reviewer answered with `ds(nl.ds((1, 1)))`. The checker names the mistake
# and never the line, and a reviewer with the same blind spot cannot find it. Python can: bind every
# nki call in the candidate against the REAL signature (inspect) and report line, call and fix.

def _fb_ds(start, size):                                  # seen in the v4.1 verifier output
    pass


def _fb_tensor_copy(dst, src, engine=None, name=None):    # printed by the verifier in the same run
    pass


FALLBACKS = {"nki.language.ds": _fb_ds, "nki.isa.tensor_copy": _fb_tensor_copy}


def _module_aliases(tree):
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                alias[a.asname or a.name] = f"{node.module}.{a.name}"
    return alias


def _dotted_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve(dotted, alias):
    if not dotted:
        return None
    head, _, rest = dotted.partition(".")
    if head not in alias:
        return None
    full = alias[head] + ("." + rest if rest else "")
    return full if full.split(".")[0] == "nki" else None


@functools.lru_cache(maxsize=None)
def _lookup(full):
    """-> (status, callable_or_None). status: 'ok' | 'missing' | 'unknown'."""
    mod_name, _, attr = full.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        fb = FALLBACKS.get(full)
        return ("ok", fb) if fb else ("unknown", None)
    if not hasattr(mod, attr):
        return "missing", None
    fn = getattr(mod, attr)
    return "ok", (fn if callable(fn) else None)


def _signature(full, fn):
    for candidate in (fn, FALLBACKS.get(full)):
        if candidate is None:
            continue
        try:
            return inspect.signature(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _call_hint(leaf):
    if leaf == "ds":
        return (" `nl.ds` takes two separate integers, `nl.ds(start, size)`, and is used inside an "
                "index such as `tile[:, nl.ds(start, size)]`. It is never called with a tuple.")
    if leaf in ("tensor_copy", "dma_copy"):
        return (" Pass views through dst= and src=, e.g. "
                "`nisa.tensor_copy(dst=a[0:64, nl.ds(j, 8)], src=b[0:64, nl.ds(i, 8)])`. There is "
                "no keyword for an index: the slice is the index.")
    return ""


def lint_api(code):
    """Check every nki call and attribute in `code` against the real API. Returns a list of dicts
    {line, src, msg, name, sig}. Advisory only: the simulator stays the judge, and a kernel that
    passes it is never penalised for a lint finding."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    alias = _module_aliases(tree)
    lines = code.splitlines()
    issues, seen = [], set()

    def add(node, msg, name="", sig=""):
        n = getattr(node, "lineno", 0)
        key = (n, msg)
        if key in seen:
            return
        seen.add(key)
        src = lines[n - 1].strip()[:100] if 0 < n <= len(lines) else ""
        issues.append({"line": n, "src": src, "msg": msg, "name": name, "sig": sig})

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            full = _resolve(_dotted_name(node), alias)
            if full and _lookup(full)[0] == "missing":
                try:
                    names = base_agent.available_names(full)
                except Exception:
                    names = ""
                add(node, f"`{ast.unparse(node)}` does not exist.{names}")
        if isinstance(node, ast.Call):
            written = _dotted_name(node.func)
            full = _resolve(written, alias)
            if not full:
                continue
            status, fn = _lookup(full)
            if status != "ok":
                continue
            sig = _signature(full, fn)
            if sig is None:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in
                                                                         node.keywords):
                continue
            try:
                sig.bind(*[None] * len(node.args), **{k.arg: None for k in node.keywords})
            except TypeError as e:
                leaf = full.rsplit(".", 1)[-1]
                extra = ""
                if leaf == "ds" and len(node.args) == 1 and isinstance(node.args[0], ast.Tuple):
                    extra = " You passed ONE tuple; pass its two elements as separate arguments."
                add(node, f"`{written}{sig}` was called with the wrong arguments ({e})." + extra
                    + _call_hint(leaf), name=written, sig=f"{written}{sig}")
    issues.sort(key=lambda i: i["line"])
    return issues


def format_lint(issues, limit=3):
    if not issues:
        return ""
    return "STATIC CHECK (against the real API): " + " ".join(
        f"[line {i['line']}: `{i['src']}`] {i['msg']}" for i in issues[:limit])


def diagnosis_is_sane(text):
    """Reject a model-written CHANGE that itself breaks the API (v4.1 proposed `ds(nl.ds((1, 1)))`)."""
    for span in re.findall(r"`([^`]+)`", text or ""):
        span = span.strip()
        try:
            ast.parse(span, mode="eval")
        except SyntaxError:
            continue
        if lint_api("import nki.language as nl\nimport nki.isa as nisa\n" + span):
            return False
    return True


# ---------------------------------------------------------------------------- verifier feedback

def classify_feedback(feedback, passed, truncated=False):
    text = (feedback or "").lower()
    if passed:
        return "verified"
    if "no backend set" in text:
        return "no_backend"
    if "no module named 'nl'" in text or 'no module named "nl"' in text or "there is no module named" in text:
        return "bad_import"
    if "reshape() takes" in text or ("reshape" in text and "positional arguments" in text):
        return "reshape_api"
    if "does not parse" in text or "no code came back" in text:
        return "truncated" if truncated else "parse"
    if "rule violations" in text or "not decorated" in text:
        return "rules"
    if "memoryregion" in text or "object is not callable" in text:
        return "buffer_api"
    if any(t in text for t in ("has no attribute", "unexpected keyword argument",
                               "got multiple values", "must be in [")):
        return "api_name"
    if "required positional argument" in text or "missing a required argument" in text:
        return "api_signature"
    if any(t in text for t in ("exceeds maximum", "must have at least 2 dimensions",
                               "same number of elements", "could not be broadcast",
                               "out-of-bound", "exceeds pmax")):
        return "tile_limits"
    if "could not be loaded" in text:
        return "load_error"
    if "raised " in text and "mismatch" not in text and "wrong shape" not in text:
        return "runtime_error"          # v4.1 filed these under "correctness", which hid the signal
    if any(t in text for t in ("mismatch", "wrong shape", "shapes passed", "raised ",
                               "correct on cpu but wrong", "non-finite output", "modified its input",
                               "output is", "of the output is zero")):
        return "correctness"
    if "cannot simulate" in text or "cannot import" in text:
        return "environment"
    return "other"


def locate_load_error(code):
    """Import the file the way the checker does and report WHERE it fails."""
    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError:
        return None
    try:
        exec(compiled, {"__name__": "candidate_probe"})
    except Exception as e:
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<candidate>"]
        if not frames:
            return type(e).__name__, str(e), None, ""
        n = frames[-1].lineno
        lines = code.splitlines()
        return type(e).__name__, str(e), n, lines[n - 1].strip() if 0 < n <= len(lines) else ""
    return None


def enrich_feedback(feedback, category, dropped, code=""):
    parts = [feedback or ""]
    if category in ("load_error", "no_backend") and code.strip():
        found = locate_load_error(code)
        if found:
            etype, msg, lineno, src = found
            where = f"line {lineno}: `{src}`" if lineno else "a statement run at import"
            parts.append(f"LOCATION: the error is raised at {where}.")
            name = re.search(r"name '(\w+)' is not defined", msg)
            if etype == "NameError" and name:
                n = name.group(1)
                parts.append(
                    f"`{n}` is used while the file is being imported (a default argument, "
                    f"decorator argument, or top-level statement) but is never defined there. "
                    f"Define `{n}` INSIDE the function body, for example "
                    f"`{n}, F = in_tensor.shape` as its first line, and remove it from the "
                    f"signature, defaults and decorator.")
    if category in FEEDBACK_NOTES:
        parts.append(FEEDBACK_NOTES[category])
    if dropped:
        parts.append("NOTE: these module-level statements were removed before grading because they "
                     "run at import time: " + "; ".join(dropped[:3]) + ". Do not write them.")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------- located analysis
#
# Re-run a failing kernel once and say WHERE it raised, or, for a wrong answer, WHERE each output
# value actually came from. Inputs are random floats, so every output value can be traced back to
# the input element it was copied from: "81.8% of elements are wrong" becomes a concrete mapping.

def explain_permutation(got, inp, shape2D, rows=4, examples=3):
    """Pure numpy. `got` and `inp` are [P, F1*F2]; returns a short diagnosis or ''."""
    got, inp = np.asarray(got), np.asarray(inp)
    if got.shape != inp.shape or inp.ndim != 2:
        return ""
    F1, F2 = shape2D
    P, F = inp.shape
    if F1 * F2 != F:
        return ""
    total = ok = ident = unmatched = swapped = 0
    srcs, wrong = set(), []
    for p in range(min(P, rows)):
        lookup = {float(v): i for i, v in enumerate(inp[p])}
        for j in range(F):
            f2, f1 = divmod(j, F1)
            want_src = f1 * F2 + f2
            c, r = divmod(j, F2)
            swapped += (r * F1 + c) == lookup.get(float(got[p, j]), -1)
            src = lookup.get(float(got[p, j]))
            total += 1
            if src is None:
                unmatched += 1
            else:
                srcs.add((p, src))
                ident += src == j
            if src == want_src:
                ok += 1
            elif len(wrong) < examples:
                wrong.append((p, j, f1, f2, want_src, src))
    bad = total - ok
    if bad == 0:
        return ""
    out = [f"FLOW CHECK (traced each output value back to the input element it was copied from, "
           f"first {min(P, rows)} rows, shape2D={F1}x{F2}): {ok}/{total} output positions are right."]
    for p, j, f1, f2, want_src, src in wrong:
        held = f"input[{p}, {src}]" if src is not None else "a value that is not in the input row"
        out.append(f"output[{p}, {j}] should hold input[{p}, {want_src}] (f1={f1}, f2={f2}) but holds {held}.")
    if ident >= 0.8 * bad:
        out.append("Almost every wrong position holds the element with the SAME index: the elements "
                   "were copied but never moved.")
    elif swapped >= 0.8 * total:
        out.append(f"The output is a transpose that used the dimensions the wrong way round "
                   f"({F2}x{F1} instead of {F1}x{F2}).")
    elif unmatched >= 0.3 * bad:
        out.append(f"{unmatched} positions hold values that are not in the input row at all "
                   f"(zeros or garbage): those destinations were never written, or an index "
                   f"pointed outside the row. Every one of the {F} destinations must be written.")
    elif len(srcs) < 0.5 * total:
        out.append("Many outputs hold the same input element: some source index is not advancing.")
    else:
        out.append("The elements are permuted, but not by the required index formula.")
    return " ".join(out)


def probe_kernel(level, code):
    """Run `code` on the simulator case by case. Returns a dict with kind in
    'exception' | 'mismatch' | 'ok' | 'unavailable'."""
    spec = nkibench.LEVELS[level]
    try:
        import nki  # noqa: F401
    except ImportError:
        return {"kind": "unavailable"}
    fd, path = tempfile.mkstemp(suffix=".py", prefix="_rl_probe_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        kernel = nkibench.load_kernel(path, spec["entry"])
        for case in spec["shapes"]:
            args, _ = nkibench.make_inputs(case, level)
            want = spec["ref"](*args)
            try:
                got, _ = nkibench.simulate_and_count(kernel, args)
            except Exception as e:
                frames = [fr for fr in traceback.extract_tb(e.__traceback__) if fr.filename == path]
                lineno = frames[-1].lineno if frames else None
                lines = code.splitlines()
                src = lines[lineno - 1].strip() if lineno and 0 < lineno <= len(lines) else ""
                return {"kind": "exception", "etype": type(e).__name__, "msg": str(e),
                        "lineno": lineno, "src": src, "case": case}
            if nkibench.describe_mismatch(got, want):
                return {"kind": "mismatch", "case": case, "args": args, "got": np.asarray(got)}
        return {"kind": "ok"}
    except Exception:
        return {"kind": "unavailable"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def located_analysis(level, code, probe, reveal_flow=True):
    if probe["kind"] == "exception":
        where = f"line {probe['lineno']}: `{probe['src']}`" if probe["lineno"] else "inside the kernel"
        return (f"LOCATION: {probe['etype']} is raised at {where}: {compact(probe['msg'], 200)}")
    if probe["kind"] == "mismatch" and level == 2 and reveal_flow:
        args = probe["args"]
        return explain_permutation(probe["got"], args[0], args[1])
    return ""


# ---------------------------------------------------------------------------- prompts

def trace_instructions(use_trace):
    if not use_trace:
        return ""
    return """
Before the code, emit a short engineering record in this exact format:
<TRACE>
{"contract":"input/output shapes and operation","tile_strategy":"tile and loop plan",
 "memory_flow":"where data is allocated and copied","edge_cases":"partial/boundary tiles",
 "predicted_failure":"one likely risk","verification_checks":["shape check","correctness check"],
 "diagnosis":"initial hypothesis or latest failure diagnosis"}
</TRACE>
Keep fields concise and factual. Do not claim the checker passed before it runs.
Then give the code in ONE ```python block. Do not put the TRACE inside a code fence.
"""


def strategy_instructions(action, history):
    if action == "direct":
        return "Implement the operation directly; avoid unnecessary abstractions."
    if action == "contract_trace":
        return ("State the tensor shapes, dtype, output contract, loop bounds, and how partial "
                "tiles are handled (as short code comments). Keep the kernel simple.")
    if action == "counterexample":
        return ("Prioritize dimensions smaller than a tile, non-divisible dimensions, and final "
                "partial tiles. Do not read or write out of bounds.")
    if action == "repair_diagnosis":
        ledger = "\n".join("- " + compact(item, 200) for item in list(history)[-3:])
        return ("Use the actual checker feedback below to make a minimal repair. Do not repeat "
                "the same failed API usage:\n" + (ledger or "- No previous failures."))
    if action == "exemplar":
        return ("Follow the import style, buffer allocation, and nisa call conventions of the "
                "verified example below. Copy its conventions, not its algorithm.")
    if action == "reflect":
        return "Apply the change above exactly. Keep everything else that already works."
    return ""


def lessons_block(lessons):
    if not lessons:
        return ""
    return ("LESSONS from earlier attempts (each one raised the checker score when applied):\n" +
            "\n".join(f"- {r}" for r in lessons) + "\n")


def facts_block(facts):
    if not facts:
        return ""
    return ("API FACTS the checker enforced on earlier attempts:\n" +
            "\n".join(f"- {f}" for f in facts) + "\n")


def build_prompt(args, level, action, anchor_code="", anchor_feedback="", last_feedback="",
                 last_regressed=False, history=(), exemplar=None, stuck=False, diagnosis="",
                 diag_src="", lessons=(), facts=(), rejected=()):
    if args.mode == "plain":                       # exactly what agent.py would send
        if anchor_code:
            return base_agent.repair_prompt(level, anchor_code, anchor_feedback)
        return base_agent.first_prompt(level, args.terse)

    spec = nkibench.LEVELS[level]
    if anchor_code and not stuck:
        core = base_agent.repair_prompt(level, anchor_code, anchor_feedback)
        if last_regressed and last_feedback:
            core += ("\n\nA LATER attempt made things worse and was discarded; do not do this: "
                     + compact(last_feedback, 300))
    else:
        core = base_agent.first_prompt(level, args.terse)
        if last_feedback:
            core += "\n\nYour previous reply failed: " + compact(last_feedback, 500)
    if stuck:
        core += ("\n\nSeveral replies in a row failed the same way. Do not resubmit that code. Write "
                 "the kernel again with a different structure.")
    if rejected:
        core += ("\n\nThese exact lines were rejected by the checker; do not write them again:\n" +
                 "\n".join(f"- `{r}`" for r in rejected[-3:]))

    exemplar_block = ""
    if exemplar:
        ex_level, ex_code = exemplar
        exemplar_block = (f"\nVERIFIED WORKING EXAMPLE (a different operation: "
                          f"{nkibench.LEVELS[ex_level]['op']}):\n```python\n{ex_code.strip()}\n```\n")
    diagnosis_block = ""
    if diagnosis:
        head = ("CHECKER ANALYSIS, computed from the real API signatures and the simulator (not "
                "guessed). Apply it:\n" if diag_src == "grounded" else
                "A REVIEWER diagnosed the failure. Apply this change:\n")
        diagnosis_block = "\n" + head + diagnosis + "\n"

    cards = FILE_RULES_CARD
    if args.hints in ("api", "algo"):
        cards += API_SIGS_CARD
    if args.hints == "algo" and level == 2:
        cards += LEVEL2_ALGO_HINT
    return (
        core + "\n\n" + cards + "\n" + facts_block(facts) + lessons_block(lessons) +
        exemplar_block + diagnosis_block + "\n" + strategy_instructions(action, history) + "\n" +
        trace_instructions(args.trace) +
        f"\nRequired entry point: {spec['entry']}. Reply with ONE ```python code block containing "
        f"only the imports and the function."
    )


def reflect(args, level, code, feedback, lessons, tried):
    """Last resort, used only when the verifier, the lint and the located analysis are all silent.
    Returns (diagnosis_text, rule). A diagnosis that breaks the API or repeats an earlier one is
    discarded: v4.1 fed `ds(nl.ds((1, 1)))` back to the model three times."""
    known = ""
    if lessons:
        known += "Lessons already known (do not repeat them):\n" + "\n".join(f"- {r}" for r in lessons) + "\n"
    if tried:
        known += "Changes already tried that did NOT fix it:\n" + "\n".join(f"- {t}" for t in tried[-3:]) + "\n"
    prompt = REFLECT_PROMPT.format(op=nkibench.LEVELS[level]["op"], code=code.strip()[:6000],
                                   feedback=compact(feedback, 900), known=known + ("\n" if known else ""))
    reply, _, _, _ = ask_model(args, prompt, 0.3, max_tokens=260)
    found = {k: (m.group(1).strip() if (m := rx.search(reply)) else "") for k, rx in REFLECT_RE.items()}
    if not (found["CAUSE"] and found["CHANGE"]):
        return "", ""
    change = compact(found["CHANGE"], 250)
    if not diagnosis_is_sane(change) or change in tried:
        return "", ""
    return f"CAUSE: {compact(found['CAUSE'], 250)}\nCHANGE: {change}", compact(found["RULE"], 200)


# ---------------------------------------------------------------------------- reward

def trace_score(trace, level, valid_format):
    if not valid_format:
        return 0.0, {"format": 0.0, "coverage": 0.0, "checks_quality": 0.0}
    coverage = sum(bool(str(trace.get(k, "")).strip()) for k in TRACE_FIELDS) / len(TRACE_FIELDS)
    checks = trace.get("verification_checks", [])
    if not isinstance(checks, list):
        checks = []
    check_text = " ".join(str(item).lower() for item in checks)
    required = ["shape", "correct"]
    if level >= 3:
        required.extend(["tile", "memory"])
    check_quality = sum(term in check_text for term in required) / len(required)
    claim_text = " ".join(str(trace.get(k, "")) for k in TRACE_FIELDS).lower()
    forbidden = ("all tests pass", "verified correct", "checker passed", "successfully tested")
    honesty = 0.0 if any(term in claim_text for term in forbidden) else 1.0
    score = 0.65 * coverage + 0.25 * check_quality + 0.10 * honesty
    return score, {"format": 1.0, "coverage": coverage,
                   "checks_quality": check_quality, "honesty": honesty}


MISMATCH_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)% of elements are outside tolerance")


def progress_score(parts, feedback, passed):
    """Dense signal INSIDE a failure bucket. v4.1 scored a crash on every shape 0.30 whatever the
    fix, so 40 rounds of ds() errors earned the same RL reward as pure noise. A kernel that runs and
    gets 60% of its elements right is closer than one that raises."""
    if passed:
        return 1.0
    m = MISMATCH_PCT_RE.search(feedback or "")
    if m:
        return max(0.0, 1.0 - float(m.group(1)) / 100.0)
    return 0.3 if parts.get("runs") else 0.0


def shaped_reward(kernel, trace, progress, lint_clean, hygiene, improved, use_trace):
    """The verifier dominates; small terms add gradient inside a failure bucket."""
    if use_trace:
        r = (0.65 * kernel + 0.10 * trace + 0.10 * progress + 0.05 * lint_clean
             + 0.05 * hygiene + 0.05 * improved)
    else:
        r = 0.75 * kernel + 0.10 * progress + 0.05 * lint_clean + 0.05 * hygiene + 0.05 * improved
    return min(1.0, max(0.0, r))


def group_advantages(rewards):
    """Raw (r - mean) and GRPO-style normalised (r - mean) / std. Both 0 for a group of identical
    rewards, which is exactly the v4.1 trace: 4 identical samples, so no learning signal at all."""
    mean = sum(rewards) / len(rewards)
    raw = [r - mean for r in rewards]
    std = math.sqrt(sum(x * x for x in raw) / len(rewards))
    norm = [x / std if std > 1e-9 else 0.0 for x in raw]
    return raw, norm


# ---------------------------------------------------------------------------- policy

class ShrunkUCB:
    """UCB bandit with hierarchical shrinkage: global -> level -> level+failure_category."""
    def __init__(self, actions, exploration=0.5, prior_strength=2.0, prior_mean=0.5):
        self.actions = tuple(actions)
        self.c = float(exploration)
        self.k = float(prior_strength)
        self.prior = float(prior_mean)
        self.stats = defaultdict(lambda: [0, 0.0])          # "scope||action" -> [n, sum]

    @staticmethod
    def scopes(context):
        parts = context.split("|")
        return ["*"] + ["|".join(parts[:i]) for i in range(1, len(parts) + 1)]

    def _key(self, scope, action):
        return f"{scope}||{action}"

    def estimate(self, context, action):
        v, n = self.prior, 0
        for scope in self.scopes(context):
            n, s = self.stats[self._key(scope, action)]
            v = (s + self.k * v) / (n + self.k)
        return v, n

    def select(self, context, rng, legal=None):
        legal = tuple(legal or self.actions)
        total = sum(self.stats[self._key(context, a)][0] for a in self.actions)
        scores = {}
        for a in legal:
            v, n = self.estimate(context, a)
            scores[a] = v + self.c * math.sqrt(math.log(total + 2) / (n + 1))
        best = max(scores.values())
        return rng.choice([a for a, s in scores.items() if abs(s - best) < 1e-12])

    def update(self, context, action, reward):
        for scope in self.scopes(context):
            cell = self.stats[self._key(scope, action)]
            cell[0] += 1
            cell[1] += float(reward)

    def to_json(self):
        return {k: v for k, v in self.stats.items() if v[0] > 0}

    def load_json(self, blob):
        for k, v in (blob or {}).items():
            self.stats[k] = [int(v[0]), float(v[1])]

    def table(self, context):
        return [(a, *self.estimate(context, a)) for a in self.actions]


class State:
    """Policy counts, verified kernels, lesson bank and API facts, persisted as one JSON file."""
    def __init__(self, path):
        self.path = path
        self.strategy = None
        self.temperature = None
        self.verified = {}                                   # str(level) -> source
        self.lessons = {}                                    # rule -> {"uses": n, "gain": float}
        self.facts = {}                                      # "nl.ds(start, size)" -> {"hits": n}

    def load(self, strategy, temperature):
        self.strategy, self.temperature = strategy, temperature
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    blob = json.load(f)
                strategy.load_json(blob.get("strategy"))
                temperature.load_json(blob.get("temperature"))
                self.verified = dict(blob.get("verified", {}))
                self.lessons = dict(blob.get("lessons", {}))
                self.facts = dict(blob.get("facts", {}))
                pulls = sum(v[0] for k, v in strategy.stats.items() if k.startswith("*||"))
                print(f"loaded state from {self.path} ({pulls} prior strategy pulls, "
                      f"{len(self.lessons)} lessons, {len(self.facts)} API facts, "
                      f"verified kernels for levels {sorted(self.verified) or 'none'})")
            except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
                print(f"warning: could not read {self.path} ({e}); starting fresh")

    def credit_lesson(self, rule, gain):
        if not rule:
            return
        cell = self.lessons.setdefault(rule, {"uses": 0, "gain": 0.0})
        cell["uses"] += 1
        cell["gain"] += float(gain)

    def proven_lessons(self, k=3):
        good = [(c["gain"] / c["uses"], r) for r, c in self.lessons.items()
                if c["uses"] and c["gain"] > 0]
        return [r for _, r in sorted(good, reverse=True)[:k]]

    def note_fact(self, sig):
        if sig:
            self.facts.setdefault(sig, {"hits": 0})["hits"] += 1

    def relevant_facts(self, k=4):
        ranked = sorted(self.facts.items(), key=lambda kv: -kv[1]["hits"])
        return [sig for sig, c in ranked[:k] if c["hits"] > 0]

    def save(self):
        if not self.path:
            return
        blob = {"version": SCHEMA_VERSION, "strategy": self.strategy.to_json(),
                "temperature": self.temperature.to_json(), "verified": self.verified,
                "lessons": self.lessons, "facts": self.facts}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp, self.path)


def load_exemplar_pool(state, level, seed_references, here):
    """Verified kernels from OTHER levels. Never the current level: that would replay the answer."""
    pool = {int(k): v for k, v in state.verified.items() if int(k) != level}
    if seed_references:
        for n in nkibench.LEVELS:
            path = os.path.join(here, f"reference_level{n}.py")
            if n != level and n not in pool and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    pool[n] = f.read()
    return pool


def pick_exemplar(pool, level):
    if not pool:
        return None
    n = min(pool, key=lambda m: (abs(m - level), m))
    return n, compact_exemplar(pool[n])


# ---------------------------------------------------------------------------- the model

def ask_model(args, prompt, temperature, max_tokens=None, seed=None):
    """Same request as agent.ask, with the temperature as a policy decision, a per-request seed,
    and the finish reason returned so a token-budget cut-off is not mistaken for a model failure."""
    import httpx
    est_prompt = len(prompt) // 4
    cap = max_tokens or args.max_tokens
    budget = min(cap, max(128, args.context - est_prompt - 64))
    body = dict(model=args.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=temperature, top_p=0.95,
                chat_template_kwargs={"enable_thinking": args.think})
    if seed is not None and not getattr(args, "no_seed", False):
        body["seed"] = int(seed)
    url = f"{args.base.rstrip('/')}/chat/completions"
    r = httpx.post(url, json=body, timeout=900, verify=False)
    if r.status_code == 400 and "seed" in body and "seed" in r.text.lower():
        args.no_seed = True                                  # this server rejects per-request seeds
        body.pop("seed")
        r = httpx.post(url, json=body, timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    ch = r.json()["choices"][0]
    content = (ch.get("message") or {}).get("content") or ""
    return content, ch.get("finish_reason"), budget, est_prompt


def sample_temps(base, n):
    """A spread around the policy's temperature. v4.1 sampled all four at one value and, at 0.3 and
    0.6, got four identical kernels per round: no diversity, so no group baseline and 4x the cost."""
    offsets = (-0.1, 0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7)
    return [round(min(1.1, max(0.2, base + offsets[i % len(offsets)])), 2) for i in range(n)]


def sample_prompts(prompt, n, variants):
    out = []
    for i in range(n):
        hint = VARIANT_HINTS[i % len(VARIANT_HINTS)] if (variants and n > 1) else ""
        out.append(prompt + ("\n\n" + hint if hint else ""))
    return out


def generate(args, prompts, temps, seeds):
    """One reply per prompt, in parallel; the server batches them."""
    started = time.perf_counter()
    if len(prompts) == 1:
        results = [ask_model(args, prompts[0], temps[0], None, seeds[0])]
    else:
        with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            futs = [ex.submit(ask_model, args, p, t, None, s) for p, t, s in zip(prompts, temps, seeds)]
            results = [f.result() for f in futs]
    return results, time.perf_counter() - started


# ---------------------------------------------------------------------------- the loop

class Run:
    """Everything one invocation shares across levels and episodes."""
    def __init__(self, args, strategy, temperature, state, log, sft, groups, rng, here):
        self.args, self.strategy, self.temperature, self.state = args, strategy, temperature, state
        self.log, self.sft, self.groups, self.rng, self.here = log, sft, groups, rng, here
        self.grade_cache = {}
        self.probe_cache = {}

    def grade(self, level, code, fp):
        key = (level, fp) if fp else None
        if key in self.grade_cache:
            return self.grade_cache[key], True
        res = base_agent.grade(code, level)
        if key:
            self.grade_cache[key] = res
        return res, False

    def probe(self, level, code, fp):
        key = (level, fp)
        if key not in self.probe_cache:
            self.probe_cache[key] = probe_kernel(level, code)
        return self.probe_cache[key]


def _failure_key(category, feedback):
    return category + "|" + re.sub(r"[\d.]+", "#", compact(feedback, 90))


def run_episode(run, level, episode, run_id):
    args, state = run.args, run.state
    spec = nkibench.LEVELS[level]
    print(f"\n========== RL trace agent v5 [{args.mode}, hints={args.hints}]: level {level} "
          f"({spec['op']}) episode {episode + 1}/{args.episodes} ==========")
    plain = args.mode == "plain"
    pool = {} if plain else load_exemplar_pool(state, level, args.seed_references, run.here)
    best = {"kernel": 0.0, "code": "", "feedback": "", "analysis": ""}   # best parsed attempt
    last = {"code": "", "feedback": "", "analysis": "", "category": "initial", "regressed": False}
    history, seen, tried_changes, rejected = [], set(), [], []
    fail_streak, last_key = 0, None
    stuck = low_div = verified = False
    no_improve = 0

    for round_i in range(args.rounds):
        context = f"level={level}|failure={last['category']}"
        if plain:
            action = "plain"
        elif args.mode == "reflect":
            action = "reflect" if history else "direct"
        else:
            legal = [a for a in STRATEGIES if a != "reflect"
                     and not (a == "repair_diagnosis" and not history)
                     and not (a == "exemplar" and not pool)]
            action = run.strategy.select(context, run.rng, legal)
        temp_name = run.temperature.select(context, run.rng)
        if stuck or low_div:
            temp_name = "t=0.9"                              # identical outputs: widen the sampling
        base_temp = float(temp_name.split("=")[1])
        use_exemplar = action == "exemplar" or (args.mode == "reflect" and args.seed_references)
        exemplar = pick_exemplar(pool, level) if use_exemplar else None
        lessons = state.proven_lessons() if args.mode == "reflect" else []
        facts = [] if plain else state.relevant_facts()

        anchor = last if plain else (best if best["code"] and not stuck else last)

        # Diagnosis. The verifier-grounded analysis comes first; the model is asked only when the
        # verifier, the lint and the located analysis are all silent.
        diagnosis, diag_src, rule, reflect_s = "", "", "", 0.0
        if action == "reflect" and anchor["code"].strip():
            if anchor["analysis"]:
                diagnosis, diag_src = anchor["analysis"], "grounded"
            else:
                t0 = time.perf_counter()
                diagnosis, rule = reflect(args, level, anchor["code"], anchor["feedback"], lessons,
                                          tried_changes)
                diag_src = "model" if diagnosis else ""
                reflect_s = time.perf_counter() - t0
                if diagnosis:
                    tried_changes.append(compact(diagnosis.split("CHANGE:")[-1], 200))
            print(f"  analysis ({diag_src or 'none'}, {reflect_s:.0f}s): "
                  + (compact(diagnosis, 300) if diagnosis else "nothing usable; plain repair"))
        elif action == "repair_diagnosis":
            diag_src = ""
        prompt_action = action if (action != "reflect" or diagnosis) else "repair_diagnosis"
        prompt = build_prompt(args, level, prompt_action, anchor["code"],
                              anchor["feedback"], last["feedback"], last["regressed"], history,
                              exemplar, stuck, diagnosis, diag_src, lessons, facts, rejected)
        was_stuck, was_low_div = stuck, low_div

        n = args.samples
        temps = sample_temps(base_temp, n)
        seeds = [(args.seed * 1000003 + episode * 10007 + round_i * 101 + i) % (2 ** 31)
                 for i in range(n)]
        prompts = sample_prompts(prompt, n, not args.no_variants and not plain)
        replies, wall = generate(args, prompts, temps, seeds)

        def extract(reply):
            if plain:
                raw = base_agent.extract_code(reply)
                return raw, raw, []
            raw = extract_kernel_code(reply, spec["entry"])
            code, dropped = sanitize_module(raw)
            return raw, code, dropped

        extracted = [extract(r[0]) for r in replies]
        fps = [code_fingerprint(c) if c.strip() else "" for _, c, _ in extracted]

        # Duplicates, within the group or of an earlier failure, teach nothing. Ask again once with
        # a structure-changing note and a hotter, re-seeded sample.
        resampled = 0
        if not args.no_variants and not plain and n > 1:
            first_seen, redo = set(), []
            for i, fp in enumerate(fps):
                if fp and (fp in first_seen or fp in seen):
                    redo.append(i)
                first_seen.add(fp)
            if redo:
                with cf.ThreadPoolExecutor(max_workers=len(redo)) as ex:
                    futs = {i: ex.submit(ask_model, args, prompts[i] + DUP_NOTE,
                                         min(1.1, temps[i] + 0.3), None, seeds[i] + 7919)
                            for i in redo}
                    for i, fut in futs.items():
                        prompts[i] = prompts[i] + DUP_NOTE
                        replies[i] = fut.result()
                        extracted[i] = extract(replies[i][0])
                        fps[i] = code_fingerprint(extracted[i][1]) if extracted[i][1].strip() else ""
                        temps[i] = min(1.1, temps[i] + 0.3)
                resampled = len(redo)

        results = []
        for i, ((reply, finish, budget, est_prompt), (raw_code, code, dropped), fp) in enumerate(
                zip(replies, extracted, fps)):
            truncated = finish == "length"
            trace, trace_valid = parse_trace(reply)
            (kernel, parts, feedback), cached = run.grade(level, code, fp)
            passed = bool(parts.get("correct"))
            category = classify_feedback(feedback, passed, truncated)
            lint = [] if (plain or passed or not code.strip()) else lint_api(code)
            if not passed and not plain:
                feedback = enrich_feedback(feedback, category, dropped, code)
                if lint:
                    feedback = (feedback + " " + format_lint(lint)).strip()
            repeated = bool(fp) and (fp in seen or fps.index(fp) != i)
            hygiene = 1.0 if (parts.get("parses") and not dropped and not truncated) else 0.0
            progress = progress_score(parts, feedback, passed)
            results.append(dict(
                reply=reply, finish=finish, truncated=truncated, trace=trace, trace_valid=trace_valid,
                raw_code=raw_code, code=code, dropped=dropped, kernel=float(kernel), parts=parts,
                feedback=feedback, category=category, fingerprint=fp, repeated=repeated,
                hygiene=hygiene, progress=progress, lint=lint, cached=cached, passed=passed,
                temperature=temps[i], prompt=prompts[i], analysis=""))

        for r in results:
            r["improved"] = 1.0 if r["kernel"] > best["kernel"] + 1e-9 else 0.0
            r["trace_reward"], r["trace_parts"] = trace_score(r["trace"], level, r["trace_valid"])
            r["rl"] = shaped_reward(r["kernel"], r["trace_reward"], r["progress"],
                                    0.0 if r["lint"] else 1.0, r["hygiene"], r["improved"], args.trace)
        raw_adv, norm_adv = group_advantages([r["rl"] for r in results])
        distinct = len({r["fingerprint"] for r in results})

        top = max(results, key=lambda r: (r["kernel"], r["rl"]))
        if not plain and not top["passed"] and top["kernel"] > 0 and top["category"] in LOCATABLE:
            # One re-run of the best failing sample, to say WHERE it fails or HOW its output is wrong.
            probe = run.probe(level, top["code"], top["fingerprint"])
            located = located_analysis(level, top["code"], probe, REVEAL_FLOW and not args.no_flow)
            parts_ = [p for p in (format_lint(top["lint"]), located) if p]
            top["analysis"] = " ".join(parts_)
            if located:
                top["feedback"] = (top["feedback"] + " " + located).strip()
        elif top["lint"]:
            top["analysis"] = format_lint(top["lint"])

        for i, r in enumerate(results):
            run.strategy.update(context, action, r["rl"])
            run.temperature.update(context, temp_name, r["rl"])
            for issue in r["lint"]:
                state.note_fact(issue["sig"])
                if issue["src"] and issue["src"] not in rejected:
                    rejected.append(issue["src"])
            row = {
                "schema_version": SCHEMA_VERSION, "run_id": run_id, "episode": episode,
                "mode": args.mode, "hints": args.hints, "level": level, "round": round_i, "sample": i,
                "action": action, "temperature": r["temperature"], "temp_arm": temp_name,
                "context": context, "stuck": was_stuck, "low_diversity": was_low_div,
                "diagnosis": diagnosis, "diagnosis_source": diag_src, "lesson": rule,
                "kernel_reward": r["kernel"], "trace_reward": r["trace_reward"],
                "progress": r["progress"], "lint_clean": not r["lint"], "lint": r["lint"],
                "rl_reward": r["rl"], "advantage": raw_adv[i], "advantage_norm": norm_adv[i],
                "hygiene": r["hygiene"], "improved": r["improved"], "repeated": r["repeated"],
                "group_distinct": distinct, "resampled": resampled, "grade_cached": r["cached"],
                "trace_enabled": bool(args.trace), "trace_parts": r["trace_parts"],
                "trace": r["trace"], "trace_valid_json": r["trace_valid"],
                "verifier_parts": r["parts"], "failure_category": r["category"],
                "feedback": r["feedback"], "analysis": r["analysis"], "code": r["code"],
                "raw_code_differs": r["raw_code"] != r["code"], "sanitizer_dropped": r["dropped"],
                "truncated": r["truncated"], "finish_reason": r["finish"],
                "prompt": r["prompt"], "reply": r["reply"][:12000],
                "elapsed_s": round(wall, 3), "reflect_s": round(reflect_s, 3),
            }
            run.log.write(json.dumps(row, ensure_ascii=False) + "\n")
            if run.sft and r["kernel"] >= args.sft_min_reward and r["code"].strip():
                run.sft.write(json.dumps({"messages": [{"role": "user", "content": r["prompt"]},
                                                       {"role": "assistant", "content": r["reply"]}],
                                          "reward": r["kernel"], "level": level}) + "\n")
            flags = "".join([" REPEAT" if r["repeated"] else "", " CUT" if r["truncated"] else "",
                             " STUCK-RESTART" if was_stuck else "",
                             " LINT" if r["lint"] else "", " cached" if r["cached"] else "",
                             f" stripped={len(r['dropped'])}" if r["dropped"] else ""])
            print(f"  r{round_i}.{i} {action:16s} t={r['temperature']:.2f} kernel={r['kernel']:.2f} "
                  f"prog={r['progress']:.2f} RL={r['rl']:.3f} adv={raw_adv[i]:+.3f} {wall:.0f}s "
                  f"failure={r['category']}{flags}")
            print("    verifier:", compact(r["feedback"], 420))
            if args.verbose and (r["category"] in ("parse", "truncated") or not r["code"].strip()):
                print("    reply head:", compact(r["reply"], 240))
        run.log.flush()
        if run.groups is not None and n > 1:
            run.groups.write(json.dumps({
                "level": level, "episode": episode, "round": round_i, "action": action,
                "distinct": distinct,
                "completions": [{"prompt": r["prompt"], "reply": r["reply"], "kernel": r["kernel"],
                                 "reward": r["rl"], "advantage": norm_adv[i],
                                 "category": r["category"]} for i, r in enumerate(results)]}) + "\n")
            run.groups.flush()
        if run.sft:
            run.sft.flush()
        print(f"  group: {distinct}/{n} distinct kernels"
              + (f", {resampled} re-asked after duplicates" if resampled else "")
              + (f", std(RL)={math.sqrt(sum(a * a for a in raw_adv) / n):.3f}"))

        gain = top["kernel"] - best["kernel"]
        if rule:
            state.credit_lesson(rule, gain)
        regressed = bool(best["code"]) and top["kernel"] < best["kernel"] - 1e-9
        if top["kernel"] > 0 and top["kernel"] >= best["kernel"] - 1e-9 and top["code"].strip():
            best = {"kernel": top["kernel"], "code": top["code"], "feedback": top["feedback"],
                    "analysis": top["analysis"]}
        no_improve = 0 if gain > 1e-9 else no_improve + 1

        key = _failure_key(top["category"], top["feedback"])
        fail_streak = fail_streak + 1 if key == last_key else 1
        last_key = key
        low_div = distinct < n
        stuck = (all(r["repeated"] for r in results) or (n > 1 and distinct == 1 and not top["passed"])
                 or fail_streak >= 3)
        for r in results:
            if r["fingerprint"]:
                seen.add(r["fingerprint"])
        last = {"code": top["code"], "feedback": top["feedback"], "analysis": top["analysis"],
                "category": top["category"], "regressed": regressed}
        history.append(top["feedback"])
        state.save()

        winner = next((r for r in results if r["passed"]), None)
        if winner:
            verified = True
            state.verified[str(level)] = winner["code"]
            state.save()
            print("  VERIFIED: every checker shape passed.")
            break
        if args.patience and no_improve >= args.patience:
            print(f"  no improvement for {no_improve} rounds; ending the episode early "
                  f"(--patience {args.patience}).")
            break

    print(f"level {level} episode {episode + 1}: best kernel reward={best['kernel']:.3f}"
          f"{'  (verified)' if verified else ''}")
    return best["kernel"], verified


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1,
                        help="replies per round, generated in parallel; >1 gives a group baseline")
    parser.add_argument("--episodes", type=int, default=1,
                        help="independent attempts per level; the policy carries over between them")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL") or
                        os.environ.get("GPTOSS_BASE_URL"))
    parser.add_argument("--path", default="", help="optional endpoint suffix such as /agg/v1")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--think", action="store_true",
                        help="not recommended; may consume output budget")
    parser.add_argument("--terse", type=int, default=0, choices=(0, 1, 2),
                        help="first-prompt length, as in agent.py (0 is right for Qwen3-8B)")
    parser.add_argument("--mode", choices=MODES, default="reflect",
                        help="reflect (default): verifier-grounded analysis, model reflection only "
                             "as a fallback, lesson bank. bandit: UCB over fixed prompt strategies. "
                             "plain: agent.py's own prompts through this harness, for an A/B on the "
                             "same logging.")
    parser.add_argument("--hints", choices=("none", "api", "algo"), default="api",
                        help="none: file rules only. api: + exact call signatures (documentation). "
                             "algo: + the level-2 algorithm note. Ablate this; 'algo' is close to "
                             "giving the answer.")
    parser.add_argument("--trace", action="store_true",
                        help="ask for the structured TRACE note and include it in the reward")
    parser.add_argument("--seed-references", action="store_true",
                        help="show reference_level{n}.py of OTHER levels as API-usage examples")
    parser.add_argument("--no-variants", action="store_true",
                        help="send the identical prompt to every sample and skip duplicate re-asks")
    parser.add_argument("--no-flow", action="store_true",
                        help="level 2: do not tell the model where its output values came from")
    parser.add_argument("--patience", type=int, default=5,
                        help="end an episode after this many rounds without a better kernel (0 = off)")
    parser.add_argument("--state", default="rl_state.json",
                        help="policy counts, verified kernels, lessons, API facts; '' disables")
    parser.add_argument("--export-sft", default="",
                        help="(prompt, reply) rows at or above --sft-min-reward, for rejection sampling")
    parser.add_argument("--export-groups", default="",
                        help="one JSON row per sample group with normalised advantages, for GRPO")
    parser.add_argument("--sft-min-reward", type=float, default=0.999)
    parser.add_argument("--log", default="rl_trace_attempts.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exploration", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.no_seed = False

    if not args.base:
        sys.exit("Set KERNEL_AGENT_BASE_URL (or GPTOSS_BASE_URL) or pass --base.")
    args.base = args.base.rstrip("/") + args.path
    if not args.base.startswith(("http://", "https://")):
        sys.exit(f"Invalid base URL: {args.base!r}")
    if min(args.rounds, args.samples, args.episodes) < 1:
        sys.exit("--rounds, --samples and --episodes must all be >= 1.")

    here = os.path.dirname(os.path.abspath(__file__))
    rng = random.Random(args.seed)
    strategy = ShrunkUCB(STRATEGIES, exploration=args.exploration)
    temperature = ShrunkUCB(TEMPERATURES, exploration=args.exploration)
    state = State(args.state)
    state.load(strategy, temperature)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    levels = sorted(nkibench.LEVELS) if args.all else [args.level or 1]

    sft = open(args.export_sft, "a", encoding="utf-8") if args.export_sft else None
    groups = open(args.export_groups, "a", encoding="utf-8") if args.export_groups else None
    try:
        with open(args.log, "a", encoding="utf-8") as log:
            run = Run(args, strategy, temperature, state, log, sft, groups, rng, here)
            for level in levels:
                for episode in range(args.episodes):
                    run_episode(run, level, episode, run_id)
            print("\nLearned values (shrunk estimate, pulls):")
            for level in levels:
                ctx = f"level={level}"
                if args.mode == "bandit":
                    print(f"  level {level}: " + "  ".join(
                        f"{a}={v:.2f}({n})" for a, v, n in strategy.table(ctx)))
                print(f"  level {level} temperature: " + "  ".join(
                    f"{a}={v:.2f}({n})" for a, v, n in temperature.table(ctx)))
    finally:
        for fh in (sft, groups):
            if fh:
                fh.close()

    if state.facts:
        print("\nAPI facts the checker enforced (hits):")
        for sig, c in sorted(state.facts.items(), key=lambda kv: -kv[1]["hits"])[:6]:
            print(f"  [{c['hits']}] {sig}")
    if state.lessons:
        print("\nLessons the model wrote (uses, total verifier gain):")
        for rule, c in sorted(state.lessons.items(), key=lambda kv: -kv[1]["gain"])[:8]:
            print(f"  [{c['uses']}, {c['gain']:+.2f}] {rule}")
    print(f"\nAttempt log: {args.log}")
    if args.state:
        print(f"Policy state: {args.state}  (delete it for a from-scratch comparison)")
    print("This learns prompt-level values online; it does not update model weights.")


if __name__ == "__main__":
    main()