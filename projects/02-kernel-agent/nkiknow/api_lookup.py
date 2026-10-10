"""Exact NKI signatures from the INSTALLED nki, attached to feedback when an error names a function.

Reads only live objects (inspect.signature / __doc__); never opens files, so nothing under a
'downloads' directory can be read.
"""
import ast
import difflib
import importlib
import inspect
import re

MAX_NAMES = 3
MAX_CHARS = 1200          # ~300 tokens at chars/4
DOC_LINES = 6
PREFIXES = {"nl": "nki.language", "nisa": "nki.isa", "nki": "nki"}


def _modules():
    mods = {}
    for short, full in PREFIXES.items():
        try:
            mods[short] = importlib.import_module(full)
        except Exception:
            pass
    return mods


def _names_in_error(text):
    found = [m.group(0) for m in re.finditer(r"\b(?:nl|nisa|nki)\.[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*", text)]
    for m in re.finditer(r"\b([A-Za-z_]\w*)\(\)", text):   # "nc_matmul() got an unexpected ..."
        found.append(m.group(1))
    return found


def _names_in_code(code):
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [m.group(0) for m in re.finditer(r"\b(?:nl|nisa|nki)\.[A-Za-z_]\w*", code)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts, f = [], node.func
            while isinstance(f, ast.Attribute):
                parts.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name) and f.id in PREFIXES and parts:
                out.append(".".join([f.id] + parts[::-1]))
    return out


def _resolve(name, mods):
    """Return (display_name, object) or (name, None)."""
    if "." in name:
        head, *rest = name.split(".")
        obj = mods.get(head)
        for r in rest:
            if obj is None:
                break
            obj = getattr(obj, r, None)
        return name, obj
    for short in ("nisa", "nl", "nki"):
        m = mods.get(short)
        if m is not None and hasattr(m, name):
            return f"{short}.{name}", getattr(m, name)
    return name, None


def _describe(name, obj, mods):
    if obj is None:
        pool, leaf = [], name.split(".")[-1]
        for short, m in mods.items():
            pool += [f"{short}.{n}" for n in dir(m) if not n.startswith("_")]
        close = difflib.get_close_matches(name, pool, n=4, cutoff=0.5) or \
            [p for p in pool if leaf in p][:4]
        return (f"`{name}` does not exist in the installed nki." +
                (f" Close real names: {', '.join(close)}." if close else ""))
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return f"`{name}` is a {type(obj).__name__} value, not a function; it cannot be called."
    doc = inspect.getdoc(obj) or ""
    lines = [l.strip() for l in doc.splitlines() if l.strip()][:DOC_LINES]
    return f"`{name}{sig}`" + ("\n  " + "\n  ".join(lines) if lines else "")


def signatures_for(error_text, code):
    mods = _modules()
    if not mods:
        return ""
    ordered, seen = [], set()
    err = _names_in_error(error_text or "")
    codes = _names_in_code(code or "")
    resolved = {}
    for n in err + codes:
        if n not in resolved:
            resolved[n] = _resolve(n, mods)
    bad_in_code = [n for n in codes if resolved[n][1] is None
                   or not callable(resolved[n][1])]
    # existing names from the error first, then broken names in the code, then other called names
    for n in [x for x in err if resolved[x][1] is not None] + bad_in_code + codes + err:
        key = resolved[n][0]
        if key not in seen:
            seen.add(key)
            ordered.append(n)
    parts, used = [], 0
    for n in ordered[:MAX_NAMES]:
        disp, obj = resolved[n]
        text = _describe(disp, obj, mods)
        if used + len(text) > MAX_CHARS:
            text = text[:max(0, MAX_CHARS - used)]
        if not text:
            break
        parts.append(text)
        used += len(text)
    if not parts:
        return ""
    return "Real signatures from the installed nki:\n" + "\n".join(parts)
