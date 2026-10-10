#!/usr/bin/env python3
"""
tools.py — the two tools the model can aim itself, over a text-marker protocol.

WHY TOOLS. Measured in project 01: given only directional feedback, the model guesses a
pattern and repeats itself byte for byte; given the target values, it copies them and
derives nothing. What worked was the SPLIT: the model decides WHAT to compute (the
reasoning), a tool does the arithmetic it cannot do reliably. Project 02's version is the
same lesson arriving for kernels: it invents API names because it cannot check, and it
cannot check because it has no REPL.

THE PROTOCOL (the endpoint's tools= parameter is a no-op, so this is hand-rolled in the
prompt -- the same mechanism as project 01's COMPUTE:):

    the reply may contain lines like
        SCRATCH: <python expression or statement>
        DOCS: <what to look up>
    the harness executes them and returns the results; the model continues.
    a reply with a ```python code block instead is an ATTEMPT.
    every attempt ends with a confidence line:  CONFIDENCE: high | medium | low

The REPL is PERSISTENT for the whole level: variables from earlier SCRATCH lines are
still there later, so the model can build up a small experiment across exchanges. That
is the prime-agent idea -- a consistent interpreter, not a fresh eval each time.
"""

import io
import re
import signal
import warnings
from contextlib import redirect_stdout

import numpy as np

MAX_OUTPUT_CHARS = 1500
MAX_SCRATCH_LEN = 1200

def _guard_import(name, *args, **kwargs):
    """numpy's internals probe __import__ on some paths (argmax on a float array hit a
    KeyError('__import__') in the first live run). Allow the harmless stdlib, nothing
    that reaches the network or the filesystem."""
    if name in ("math", "itertools", "functools", "collections", "operator", "warnings") \
            or name.startswith("numpy") or name.startswith("math"):
        return __import__(name, *args, **kwargs)
    raise ImportError("SCRATCH allows only numpy (already loaded as np) and basic stdlib")


# Builtins the REPL gets. Deliberately generous -- SCRATCH is for CHECKING, so range,
# len, sorted, abs are all welcome -- minus everything that reaches outside the process.
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "hash": hash, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "map": map, "max": max, "min": min, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
    "__import__": _guard_import,
    "ArithmeticError": ArithmeticError, "IndexError": IndexError,
    "TypeError": TypeError, "ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError,
}

_FORBIDDEN = re.compile(r"\b(import|__import__|open|exec|eval|compile|input|globals|locals"
                        r"|__builtins__|getattr|setattr|delattr|breakpoint)\b")


class Scratch:
    """One persistent Python session. One per level, fresh per level."""

    def __init__(self):
        self.globals = {"__builtins__": _SAFE_BUILTINS, "np": np, "numpy": np}
        self.history = []          # (code, output) -- compacted by context.py

    def _eval_or_exec(self, code):
        """Expression -> its value; statement -> run it; and the case the model writes
        most: `x = np.arange(12); x[0:2, 1:3].shape` -- an assignment AND a trailing
        expression. eval() rejects the whole line, plain exec() hides the value, and the
        model was told '(no output)' for exactly the checks it most needed to see
        (measured, first live run). So: exec the prefix, eval the last segment."""
        try:
            return eval(code, self.globals)
        except SyntaxError:
            pass
        if ";" in code:
            prefix, _, last = code.rpartition(";")
            try:
                exec(prefix, self.globals)
                return eval(last.strip(), self.globals)
            except SyntaxError:
                pass            # last segment is itself a statement: run it all
        exec(code, self.globals)
        return None

    def run(self, code):
        code = (code or "").strip().rstrip(";")
        if not code:
            return "nothing to run"
        if len(code) > MAX_SCRATCH_LEN:
            return f"too long ({len(code)} chars, limit {MAX_SCRATCH_LEN}) -- run a smaller check"
        if _FORBIDDEN.search(code):
            return ("rejected: SCRATCH runs pure numpy arithmetic only (np is already "
                    "loaded). No import/open/eval/getattr.")
        buf = io.StringIO()
        try:
            use_signal = hasattr(signal, "setitimer")
            if use_signal:
                try:
                    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
                    signal.setitimer(signal.ITIMER_REAL, 5)
                except ValueError:
                    use_signal = False
            try:
                with warnings.catch_warnings():
                    # 'overflow encountered in exp' is information the RESULT already
                    # carries (inf); the printed block is not, and it leaks to stderr
                    # where nobody driving an agent loop will see it.
                    warnings.simplefilter("ignore")
                    with redirect_stdout(buf):
                        result = self._eval_or_exec(code)
            finally:
                if use_signal:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        except _Timeout:
            out = f"TIMED OUT after 5s: {code[:80]}"
            self.history.append((code, out))
            return out
        except Exception as e:
            # The exception IS the result -- 'shape mismatch (3,) (3,1)' is exactly what
            # the model needed to see. Not an error of the tool.
            out = f"{type(e).__name__}: {e}"
            self.history.append((code, out))
            return out
        out = buf.getvalue().rstrip("\n")
        if result is not None:
            shown = repr(result)
            out = (out + "\n" if out else "") + shown
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(out)} chars total]"
        self.history.append((code, out))
        return out or "(no output -- assign to a variable or print it)"


class _Timeout(Exception):
    pass


# ---------------------------------------------------------------- docs retrieval

def load_cards(docs_dir):
    """docs/*.md -> list of (card_name, section_title, text). Sections are the retrieval
    units: small enough to drop straight into a prompt."""
    import pathlib
    cards = []
    for path in sorted(pathlib.Path(docs_dir).glob("*.md")):
        card = path.stem
        sections, cur_title, cur = [], "(top)", []
        for line in path.read_text().splitlines():
            if line.startswith("## "):
                if cur:
                    sections.append((cur_title, "\n".join(cur).strip()))
                cur_title, cur = line[3:].strip(), []
            else:
                cur.append(line)
        if cur:
            sections.append((cur_title, "\n".join(cur).strip()))
        for title, text in sections:
            if text:
                cards.append((card, title, text))
    return cards


def _tokens(text):
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())


class Docs:
    """BM25-lite retrieval over the doc cards. No dependencies; the corpus is a dozen
    sections, so term overlap with idf weighting is all the ranking it needs."""

    def __init__(self, docs_dir):
        self.sections = load_cards(docs_dir)
        # The section TITLE is part of the searchable text: a model asking about "argmax"
        # must reach the section titled "Ties (argmax-style outputs)" even if its body
        # never repeats the word.
        self.doc_tokens = [set(_tokens(f"{t} {x}")) for _c, t, x in self.sections]
        self.doc_all = [_tokens(f"{t} {x}") for _c, t, x in self.sections]
        n = len(self.sections) or 1
        self.idf = {}
        df = {}
        for toks in self.doc_tokens:
            for w in toks:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: np.log((n - d + 0.5) / (d + 0.5) + 1) for w, d in df.items()}

    def search(self, query, k=2, max_chars=900):
        q = _tokens(query)
        if not q:
            return "empty query"
        scored = []
        for i, toks in enumerate(self.doc_all):
            s = 0.0
            for w in q:
                tf = toks.count(w)
                if tf:
                    s += self.idf.get(w, 1.0) * tf * 2.2 / (tf + 1.2)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        if not scored:
            return (f"nothing in the docs about {query!r}. Cards available: "
                    + ", ".join(sorted({c for c, _t, _x in self.sections})))
        out = []
        used = 0
        for s, i in scored[:k]:
            card, title, text = self.sections[i]
            block = f"[{card} / {title}]\n{text}"
            if used + len(block) > max_chars:
                block = block[:max(0, max_chars - used)] + "..."
            out.append(block)
            used += len(block)
            if used >= max_chars:
                break
        return "\n\n---\n\n".join(out)

    def list(self):
        return "\n".join(f"- {c} / {t}" for c, t, _x in self.sections)


# ---------------------------------------------------------------- parsing + protocol text

SCRATCH_RE = re.compile(r"^\s*SCRATCH:\s*(.+?)\s*$", re.M)
DOCS_RE = re.compile(r"^\s*DOCS:\s*(.+?)\s*$", re.M)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(high|medium|low|m|h|l)\b", re.I)
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

TOOL_INSTRUCTIONS = """You have two tools. To use one, include lines of exactly this form in \
your reply (the harness runs them and sends back the results; then you continue):

  SCRATCH: <one python statement or expression>   -- a PERSISTENT numpy session (np is \
preloaded; no import). State survives between your SCRATCH lines, so you can build an \
experiment across several. Use it to CHECK index arithmetic, tile slicing, output sizes and \
numerical identities on tiny arrays BEFORE you commit to an answer.
  DOCS: <topic>                                    -- search the local reference cards \
(tiling patterns, numeric gotchas, rules).

Example:
  SCRATCH: x = np.arange(12).reshape(3, 4); x[0:2, 1:3].shape
  DOCS: softmax overflow

When you are ready to answer, reply with ONE ```python code block defining `kernel`, and end \
your reply with a line: CONFIDENCE: high (or medium, or low) -- how sure you are that the \
code is correct. A wrong 'high' is scored worse than an honest 'low'."""


def parse_reply(text):
    """Split a model reply into (code|None, scratch_asks, docs_asks, confidence)."""
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    code = max(blocks, key=len).strip() if blocks else None
    scratch = [m.group(1) for m in SCRATCH_RE.finditer(text)][:6]
    docs = [m.group(1) for m in DOCS_RE.finditer(text)][:4]
    m = CONFIDENCE_RE.search(text)
    conf = None
    if m:
        val = m.group(1).lower()
        conf = {"h": "high", "m": "medium", "l": "low"}.get(val, val)
    return code, scratch, docs, conf


def extract_code(text):
    """The longest fenced block, else nothing -- never prose (project 02 measured a
    numbered list reaching the compiler as 'invalid decimal literal')."""
    blocks = CODE_BLOCK.findall(text or "")
    return max(blocks, key=len).strip() if blocks else ""
