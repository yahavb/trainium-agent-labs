"""tools.py -- tools the model AIMS ITSELF, the mechanism that solved Project 1.

Two tools, both answering a question the model chose to ask, both resolved against the INSTALLED
nki package rather than a static card:

  DOC: <name>     the real signature + the head of the docstring of an nki function/attr
  PROBE: <code>   run a tiny snippet under nki.simulate in a subprocess, return stdout/shapes/errors

The point (measured in Project 1): directional feedback alone makes a model guess; a tool it aims
itself lets it DISCOVER the API and the shapes instead of inventing them. DOC mirrors the
calculator -- it answers exactly what was asked. PROBE lets the model run a 5-line experiment to
see what a slice or an .ap() view actually is at trace time.

Both are OFF by default and enabled per-run with --tools. DOC needs only `import nki` to be useful;
PROBE needs the Neuron SDK to actually execute (it degrades to a clear message otherwise).
"""
import re
import subprocess
import sys
import textwrap

DOC_RE = re.compile(r"^\s*DOC:\s*([A-Za-z_][\w.]*)\s*$", re.M)
PROBE_RE = re.compile(r"```probe\s*(.*?)```", re.S)

MAX_DOC_CALLS = 3
MAX_PROBE_CALLS = 2
DOC_BUDGET_CHARS = 1200
PROBE_BUDGET_CHARS = 600
PROBE_TIMEOUT_S = 20


def resolve_doc(name):
    """Signature + docstring head for a dotted nki name, from the installed package.

    Accepts 'nisa.nc_matmul', 'nl.ndarray', 'nki.jit', or a bare leaf we try to locate in the
    usual modules. Returns a short string, never raises.
    """
    import importlib
    import inspect

    candidates = []
    if "." in name:
        mod_name, _, attr = name.rpartition(".")
        alias = {"nl": "nki.language", "nisa": "nki.isa", "nki": "nki"}.get(mod_name, mod_name)
        candidates.append((alias, attr))
    else:
        for alias in ("nki.language", "nki.isa", "nki"):
            candidates.append((alias, name))

    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        obj = getattr(mod, attr, None)
        if obj is None:
            continue
        short = mod_name.split(".")[-1]
        try:
            sig = f"{short}.{attr}{inspect.signature(obj)}"
        except (TypeError, ValueError):
            sig = f"{short}.{attr}"
        doc = inspect.getdoc(obj) or ""
        head = " ".join(doc.split("\n\n")[0].split()) if doc else "(no docstring)"
        return f"{sig}\n    {head[:400]}"
    # Not found: offer near names from the most likely module.
    import difflib
    leaf = name.rpartition(".")[2]
    for mod_name in ("nki.isa", "nki.language", "nki"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        names = [n for n in dir(mod) if not n.startswith("_")]
        close = difflib.get_close_matches(leaf, names, n=6, cutoff=0.4)
        if close:
            return f"no `{name}`. Closest real names in {mod_name}: {', '.join(close)}"
    return f"no `{name}` found in nki, nki.language or nki.isa."


def answer_docs(text):
    """Find DOC: lines in the model's reply and resolve them. Returns ('' | tool-output block)."""
    names = DOC_RE.findall(text or "")[:MAX_DOC_CALLS]
    if not names:
        return ""
    out = ["DOC results (the real NKI 0.6 API):"]
    for n in names:
        out.append(f"- {resolve_doc(n)}")
    return "\n".join(out)[:DOC_BUDGET_CHARS]


_PROBE_HARNESS = """\
import sys
import numpy as np
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except Exception as e:
    print("PROBE could not import the Neuron SDK:", e); sys.exit(0)

def _run():
{body}

try:
    _run()
except Exception as e:
    print("PROBE raised:", type(e).__name__, str(e)[:300])
"""


def answer_probe(text):
    """Run a ```probe``` snippet under nki.simulate in a subprocess. Returns '' or an output block.

    The snippet body is indented into a _run() function; it may print() shapes and values. We run
    it isolated with a timeout so a bad loop cannot hang the agent. Needs the SDK to do anything.
    """
    blocks = PROBE_RE.findall(text or "")[:MAX_PROBE_CALLS]
    if not blocks:
        return ""
    outs = ["PROBE results:"]
    for body in blocks:
        src = _PROBE_HARNESS.format(body=textwrap.indent(body.strip() or "pass", "    "))
        try:
            r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                               timeout=PROBE_TIMEOUT_S)
            out = (r.stdout + r.stderr).strip() or "(no output)"
        except subprocess.TimeoutExpired:
            out = f"(timed out after {PROBE_TIMEOUT_S}s -- the snippet ran too long)"
        outs.append(out[:PROBE_BUDGET_CHARS])
    return "\n".join(outs)[:PROBE_BUDGET_CHARS * MAX_PROBE_CALLS + 64]


def gather_tool_output(text, enabled):
    """Combine whatever tools are enabled. `enabled` is a set like {"doc", "probe"}."""
    chunks = []
    if "doc" in enabled:
        d = answer_docs(text)
        if d:
            chunks.append(d)
    if "probe" in enabled:
        p = answer_probe(text)
        if p:
            chunks.append(p)
    return "\n\n".join(chunks)


if __name__ == "__main__":
    # Degrades gracefully with no SDK; useful on the pod.
    print(resolve_doc("nisa.nc_matmul"))
    print("---")
    print(answer_docs("I need DOC: nl.ndarray\nand DOC: nisa.tensor_copy\n"))
