#!/usr/bin/env python3
"""
retrieve.py -- decide which slice of the official NKI guide the model gets to see.

The guide (aws-neuron/neuron-agentic-development, skills/neuron-nki-*) is ~116 KB, about 29k
tokens. The model sees 8192 tokens in total, prompt and answer together. So it cannot read the
guide; something has to pick. This file is that something.

    chunk   every markdown section (## / ###) and every example .py file is one chunk.
            Example files lose their docstrings and their test harness: the test imports torch,
            and the rule checker rightly rejects a kernel that mentions torch.matmul.
    score   BM25 over identifier-aware tokens, so `nc_matmul` and `tensor_reduce` match as
            words. The query is the operation plus, on repair rounds, the checker's feedback --
            so a PSUM error pulls in the PSUM section, not the one about transposes.
    pack    best-first until a character budget is spent. A chunk that does not fit is skipped,
            not truncated: half a code example teaches a broken pattern.

    python retrieve.py DOCS_DIR "average pooling sum reduce"     # see what a query retrieves
"""

import ast
import math
import os
import re
import sys

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOP = set("""a an the and or of to in on for with is are be by as at it this that from into
not no if then else when use using used can will must should only any all one each per your you
we our its their there here than so do does""".split())


def words(text):
    out = []
    for t in TOKEN.findall(text.lower()):
        if t in STOP or len(t) < 2:
            continue
        out.append(t)
        # nc_matmul also counts as nc, matmul -- the model's error says "matmul", the doc says
        # "nc_matmul", and both should meet.
        if "_" in t:
            out.extend(p for p in t.split("_") if len(p) > 1 and p not in STOP)
    return out


def _strip_example(src):
    """Keep the kernel, drop the docstrings and everything from the test harness on."""
    lines = []
    for line in src.splitlines():
        if line.startswith(("if __name__", "def test", "def main", "import torch")):
            break
        lines.append(line)
    src = "\n".join(lines)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None),
                                                          ast.Constant) \
                    and isinstance(first.value.value, str):
                spans.append((first.lineno, first.end_lineno))
    keep = src.splitlines()
    for a, b in sorted(spans, reverse=True):
        del keep[a - 1:b]
    return "\n".join(l for l in keep if not l.startswith("# Copyright")
                     and not l.startswith("# SPDX")).strip()


def load_chunks(root):
    chunks = []
    for d, _, files in os.walk(root):
        if "/." in d:
            continue
        for f in sorted(files):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            if "nki" not in rel:          # only the NKI skills, not the framework-porting ones
                continue
            if "profil" in rel:           # profiling is about timing a device we don't time yet
                continue
            text = open(p, errors="replace").read()
            if f.endswith(".py"):
                if "/examples/" not in p:
                    continue
                body = _strip_example(text)
                chunks.append(dict(src=rel, title=f, text=body))
            elif f.endswith(".md"):
                # split on ## and ### headings; keep the heading with its body
                parts = re.split(r"(?m)^(?=#{2,3} )", text)
                for part in parts:
                    part = part.strip()
                    if len(part) < 80:
                        continue
                    title = part.splitlines()[0].lstrip("# ").strip()
                    chunks.append(dict(src=rel, title=title, text=part))
    for c in chunks:
        c["words"] = words(c["title"] + " " + c["title"] + " " + c["text"])
    return chunks


CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
API_REF = re.compile(r"\b(nl|nisa)\.([A-Za-z_]\w*)")


def installed_api():
    """Every public name in the NKI that is actually installed, or None if nki is absent."""
    try:
        import nki.isa as nisa
        import nki.language as nl
    except ImportError:
        return None
    return {"nl": set(dir(nl)), "nisa": set(dir(nisa))}


def stale_names(chunk, api):
    """nl.X / nisa.X used in this chunk's CODE that the installed NKI does not have.

    Measured: the guide's average-pooling tutorial uses nl.mgrid, removed in the NKI this pod runs.
    Retrieved verbatim, it taught the model nl.mgrid and then nl.arange, three rounds running.
    Prose is exempt -- the migration tables NAME the removed calls in order to forbid them."""
    code = chunk["text"] if chunk["src"].endswith(".py") else \
        "\n".join(CODE_FENCE.findall(chunk["text"]))
    return sorted({f"{m}.{n}" for m, n in API_REF.findall(code) if n not in api[m]})


class Index:
    def __init__(self, root, check_freshness=True):
        self.chunks = load_chunks(root)
        self.dropped_stale = []
        api = installed_api() if check_freshness else None
        if api:
            keep = []
            for c in self.chunks:
                bad = stale_names(c, api)
                (self.dropped_stale.append((c["src"], c["title"], bad)) if bad
                 else keep.append(c))
            self.chunks = keep
        n = len(self.chunks)
        df = {}
        for c in self.chunks:
            for w in set(c["words"]):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (n - k + 0.5) / (k + 0.5)) for w, k in df.items()}
        self.avg = sum(len(c["words"]) for c in self.chunks) / max(n, 1)

    def score(self, c, q, k1=1.2, b=0.75):
        tf = {}
        for w in c["words"]:
            tf[w] = tf.get(w, 0) + 1
        L = len(c["words"])
        s = 0.0
        for w in set(q):
            if w in tf:
                f = tf[w]
                s += self.idf.get(w, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * L / self.avg))
        return s

    def pick(self, query, budget_chars, max_chunk_chars=None):
        """Best chunks for `query` whose total size stays under budget_chars."""
        q = words(query)
        ranked = sorted(self.chunks, key=lambda c: -self.score(c, q))
        limit = max_chunk_chars or budget_chars
        out, used = [], 0
        for c in ranked:
            if self.score(c, q) <= 0:
                break
            n = len(c["text"])
            if n > limit or used + n > budget_chars:
                continue
            out.append(c)
            used += n
        return out

    def render(self, query, budget_chars, max_chunk_chars=None):
        picked = self.pick(query, budget_chars, max_chunk_chars)
        if not picked:
            return "", []
        body = "\n\n".join(f"[{c['src']} -- {c['title']}]\n{c['text']}" for c in picked)
        return ("Excerpts from the official NKI guide, chosen for this task. Adapt the pattern; "
                "keep the entry-point name and signature you were asked for.\n\n" + body,
                [f"{c['src']}#{c['title']}" for c in picked])


if __name__ == "__main__":
    idx = Index(sys.argv[1])
    print(f"dropped {len(idx.dropped_stale)} stale chunks")
    print(f"{len(idx.chunks)} chunks, {sum(len(c['text']) for c in idx.chunks):,} chars")
    text, used = idx.render(" ".join(sys.argv[2:]), 6000)
    print("\n".join(used))
    print(f"--- {len(text)} chars")
