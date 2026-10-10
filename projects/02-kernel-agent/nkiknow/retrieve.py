"""Deterministic retrieval over the AWS-shipped NKI docs (no dependencies).

Index = markdown sections (split on headings, fence-aware) from
neuron-nki-docs/ and neuron-nki-writing/references/*.md, chunked to ~250 tokens
(chars/4). `lookup(query)` first tries symbol resolution through
indices/symbol-lookup.md (doc link + anchor), else BM25 keyword search.

Docs are read at runtime from the installed path (not committed; AWS content).
NEVER indexed: any `downloads/` or `examples/` folder (answer kernels), and any
fenced code block defining an answer kernel (def nki_matmul_* / def tensor_avgpool*),
which also occurs inside some tutorial pages.

CLI: python -m nkiknow.retrieve "query" [--docs-root DIR] [--max-tokens N]
"""
import argparse
import math
import os
import re
from collections import Counter

DEFAULT_ROOT = "/opt/conda/lib/python3.13/site-packages/neuron_agentic_development/artifacts/skills"
EXCLUDED_DIRS = {"downloads", "examples", "__pycache__"}
# Tutorial pages that walk through ladder answers in prose and pseudo-code (seen served for a level-4
# error on 2026-10-10). Stripping their code blocks was not enough.
EXCLUDED_FILES = {"matrix_multiplication.md", "average_pool2d.md"}
SYMBOL_FILE = "neuron-nki-docs/references/indices/symbol-lookup.md"
# Used for symbol resolution only; tables would add noise to BM25.
NOT_SEARCHED = {SYMBOL_FILE, "neuron-nki-docs/references/indices/hierarchical-toc.md"}
BANNED = ("def nki_matmul_", "def tensor_avgpool")
CHUNK_CHARS = 1000  # ~250 tokens

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_EXPLICIT = re.compile(r"\s*\{#([^}]+)\}\s*$")
_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = set("a an the of to in on for and or is are be with by as at it this that from how what why when do does not no use using into can".split())


def default_root():
    return os.environ.get("NKI_DOCS_ROOT") or DEFAULT_ROOT


def _slug(s):
    s = re.sub(r"[^a-z0-9 _-]", "", s.lower()).strip()
    return re.sub(r"\s+", "-", s)


def tokenize(text):
    out = []
    for w in _TOKEN.findall(text.lower()):
        if w in _STOP:
            continue
        out.append(w)
        if "_" in w:  # dma_copy -> also dma, copy
            out.extend(p for p in w.split("_") if p and p not in _STOP)
    return out


def _strip_banned_blocks(lines):
    """Drop fenced code blocks that define an answer kernel."""
    out, block, in_f = [], [], False
    for ln in lines:
        if _FENCE.match(ln):
            if not in_f:
                in_f, block = True, [ln]
            else:
                block.append(ln)
                if not any(b in "\n".join(block) for b in BANNED):
                    out.extend(block)
                in_f, block = False, []
        elif in_f:
            block.append(ln)
        else:
            out.append(ln)
    if in_f and not any(b in "\n".join(block) for b in BANNED):
        out.extend(block)
    return out


def split_sections(text):
    """-> list of (heading, anchor, body). Fence-aware ('# x' in code is not a heading)."""
    lines = _strip_banned_blocks(text.splitlines())
    secs, cur, in_f = [], ["", "", []], False
    for ln in lines:
        if _FENCE.match(ln):
            in_f = not in_f
        m = None if in_f else _HEAD.match(ln)
        if m:
            if cur[2] or cur[0]:
                secs.append(tuple(cur))
            title = m.group(2)
            e = _EXPLICIT.search(title)
            anchor = e.group(1) if e else _slug(title)
            title = _EXPLICIT.sub("", title)
            cur = [title, ("!" if e else "") + anchor, []]
        else:
            cur[2].append(ln)
    if cur[2] or cur[0]:
        secs.append(tuple(cur))
    return [(h, a, "\n".join(b).strip()) for h, a, b in secs]


def _chunk(body):
    if len(body) <= CHUNK_CHARS * 1.3:
        return [body]
    chunks, cur, n = [], [], 0
    for para in re.split(r"\n\s*\n", body):
        if cur and n + len(para) > CHUNK_CHARS:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        while len(para) > CHUNK_CHARS * 1.5:  # giant paragraph/code block: cut on lines
            cut = para.rfind("\n", 0, CHUNK_CHARS)
            cut = cut if cut > 0 else CHUNK_CHARS
            if cur:
                chunks.append("\n\n".join(cur))
                cur, n = [], 0
            chunks.append(para[:cut])
            para = para[cut:].lstrip("\n")
        cur.append(para)
        n += len(para)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _walk(root):
    for sub in ("neuron-nki-docs", "neuron-nki-writing"):
        base = os.path.join(root, sub)
        for dp, dns, fns in os.walk(base):
            dns[:] = sorted(d for d in dns if d not in EXCLUDED_DIRS)
            for fn in sorted(fns):
                if fn.endswith(".md") and fn not in EXCLUDED_FILES:
                    yield os.path.join(dp, fn)


class Index:
    def __init__(self, root=None):
        self.root = root or default_root()
        self.chunks = []  # dicts: path, anchor, heading, text, tf, len
        self.symbols = {}  # name -> list of (module, path, anchor)
        for fp in _walk(self.root):
            rel = os.path.relpath(fp, self.root)
            parts = rel.split(os.sep)
            assert not (set(parts) & EXCLUDED_DIRS), rel
            with open(fp, encoding="utf-8", errors="replace") as f:
                txt = f.read()
            if rel == SYMBOL_FILE:
                self._load_symbols(txt)
            if rel in NOT_SEARCHED:
                continue
            for head, anchor, body in split_sections(txt):
                explicit = anchor.startswith("!")
                anchor = anchor.lstrip("!")
                for i, c in enumerate(_chunk(body)):
                    if not c.strip() and not head:
                        continue
                    toks = tokenize(head) * 3 + tokenize(c) + tokenize(os.path.basename(rel))
                    self.chunks.append(dict(path=rel, anchor=anchor, heading=head, text=c,
                                            part=i, explicit=explicit, tf=Counter(toks), len=len(toks)))
        n = len(self.chunks)
        self.avgdl = sum(c["len"] for c in self.chunks) / max(n, 1)
        df = Counter()
        for c in self.chunks:
            df.update(c["tf"].keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _load_symbols(self, txt):
        row = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([\w.]+)\s*\|.*?\]\(\.\./([^)#]+)(?:#([^)]+))?\)")
        base = "neuron-nki-docs/references/"
        for ln in txt.splitlines():
            m = row.match(ln)
            if m:
                name, mod, path, anchor = m.groups()
                self.symbols.setdefault(name, []).append((mod, base + path, anchor or ""))

    # ---- search ----
    def search(self, query, k=3):
        q = tokenize(query)
        scored = []
        for i, c in enumerate(self.chunks):
            s = 0.0
            for t in set(q):
                f = c["tf"].get(t)
                if f:
                    s += self.idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * c["len"] / self.avgdl))
            if s > 0:
                scored.append((-s, i))
        scored.sort()
        return [(self.chunks[i], -s) for s, i in scored[:k]]

    def resolve_symbol(self, query):
        """Return (path, anchor) for the first NKI symbol found in the query, else None."""
        cands = []
        for m in re.finditer(r"\b(nl|nisa|nki(?:\.\w+)*)\.(\w+)", query):
            pre, name = m.groups()
            mod = {"nl": "nki.language", "nisa": "nki.isa"}.get(pre, pre)
            cands.append((name, mod))
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):  # bare names: only distinctive ones
            if ("_" in w or any(ch.isdigit() for ch in w)) and w in self.symbols:
                cands.append((w, None))
        for name, mod in cands:
            ents = self.symbols.get(name, [])
            pick = [e for e in ents if mod is None or e[0] == mod] or ([] if mod else ents)
            if pick:
                m_, path, anchor = pick[0]
                if not anchor:  # index gives only a page: prefer the symbol's own section
                    full = f"{m_}.{name}"
                    own = next((c for c in self.chunks if c["explicit"] and c["heading"].strip() == full), None)
                    if own:
                        return own["path"], own["anchor"]
                return path, anchor
        return None

    def section(self, path, anchor):
        """Chunks of the anchored section plus following un-anchored sub-sections
        (some API pages put the body under a second heading after `{#anchor}`)."""
        fc = [c for c in self.chunks if c["path"] == path]
        start = next((i for i, c in enumerate(fc) if anchor and c["anchor"] == anchor), None)
        if start is None:
            return fc[:1] if not anchor else []
        out = [fc[start]]
        for c in fc[start + 1:]:
            if c["explicit"] and c["anchor"] != anchor:
                break
            out.append(c)
        return out


_CACHE = {}


def get_index(root=None):
    root = root or default_root()
    if root not in _CACHE:
        _CACHE[root] = Index(root)
    return _CACHE[root]


def _fmt(c):
    h = f"## {c['heading']}\n" if c["heading"] else ""
    return h + c["text"]


def lookup(query, max_tokens=300, docs_root=None):
    ix = get_index(docs_root)
    budget = max_tokens * 4
    hit = ix.resolve_symbol(query)
    chunks = ix.section(*hit) if hit else []
    if not chunks:
        chunks = [c for c, _ in ix.search(query, k=3)]
    if not chunks:
        return "no match"
    src = f"{chunks[0]['path']}#{chunks[0]['anchor']}"
    out, used = [], 0
    for c in chunks:
        t = _fmt(c)
        if used + len(t) > budget:
            t = t[: max(budget - used, 0)]
        out.append(t)
        used += len(t)
        if used >= budget:
            break
    return "\n\n".join(out).rstrip() + f"\n\n[source: {src}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--docs-root", default=None)
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()
    print(lookup(a.query, a.max_tokens, a.docs_root))


if __name__ == "__main__":
    main()
