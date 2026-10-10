"""Retrieval tests. Docs root: --docs-root via env NKI_DOCS_ROOT (else installed path); skipped if absent.
Run: NKI_DOCS_ROOT=<skills dir> python -m pytest -s nkiknow/tests/test_retrieval.py
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nkiknow import retrieve as R  # noqa: E402

ROOT = R.default_root()
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason=f"docs root missing: {ROOT}")
QUERIES = json.load(open(os.path.join(os.path.dirname(__file__), "retrieval_queries.json")))


def _hit(c, exp):
    return any(c["path"] == e["file"] and (("anchor" not in e) or c["anchor"] == e["anchor"]) for e in exp)


def _ranked(ix, q):
    """Ranked distinct (path, anchor) as lookup() would return: symbol hit first, then BM25."""
    out = []
    sym = ix.resolve_symbol(q)
    if sym:
        for c in ix.section(*sym)[:1]:
            out.append(c)
    for c, _ in ix.search(q, k=20):
        if not any(o["path"] == c["path"] and o["anchor"] == c["anchor"] for o in out):
            out.append(c)
    return out


def test_hit_rates():
    ix = R.get_index()
    t1 = t3 = 0
    misses = []
    for item in QUERIES:
        r = _ranked(ix, item["query"])[:3]
        h1 = bool(r) and _hit(r[0], item["expect"])
        h3 = any(_hit(c, item["expect"]) for c in r)
        t1 += h1
        t3 += h3
        if not h3:
            misses.append((item["query"], [f"{c['path']}#{c['anchor']}" for c in r[:3]]))
    n = len(QUERIES)
    print(f"\nTOP1 {t1}/{n} = {t1/n:.0%}   TOP3 {t3}/{n} = {t3/n:.0%}")
    for q, got in misses:
        print("MISS", q, "->", got)
    assert t3 / n >= 0.70, f"top-3 hit rate {t3/n:.0%} < 70%"


def test_deterministic():
    assert R.lookup("accumulate in PSUM") == R.lookup("accumulate in PSUM")


def test_symbol_resolution():
    out = R.lookup("how do I call nisa.nc_matmul")
    assert "api-nki-isa-tensor.md#nki-isa-nc_matmul" in out


def test_max_tokens_respected():
    assert len(R.lookup("nisa.activation", max_tokens=100)) <= 100 * 4 + 200


def test_no_answer_kernels_indexed():
    ix = R.get_index()
    assert len(ix.chunks) > 100
    for c in ix.chunks:
        parts = set(c["path"].split(os.sep))
        assert not (parts & {"downloads", "examples"}), c["path"]
        assert "def nki_matmul_" not in c["text"], c["path"]
        assert "def tensor_avgpool" not in c["text"], c["path"]
    for q in ["nki_matmul_tiled_", "tensor_avgpool_kernel", "average pool kernel", "matrix multiplication tiled kernel"]:
        out = R.lookup(q, max_tokens=2000)
        assert "def nki_matmul_" not in out and "def tensor_avgpool" not in out
        assert "/downloads/" not in out and "/examples/" not in out
