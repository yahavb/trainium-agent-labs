import sys, types
import numpy as np
import pytest
from nkiknow import traffic


class Buf(str):
    pass


class T:
    def __init__(self, data, buffer, tid=0):
        self.buffer = buffer
        self.shape = data.shape
        self.nbytes = data.nbytes
        self._storage = types.SimpleNamespace(data=data.copy(), tensor_id=tid)

    def view(self):  # a slice shares the same storage handle
        v = object.__new__(T)
        v.__dict__.update(self.__dict__)
        v.nbytes = self.nbytes // 2
        return v


def fake_env(monkeypatch, script, a, b, c):
    isa = types.ModuleType("nki.isa")
    isa.dma_copy = lambda dst=None, src=None, **kw: None
    nki = types.ModuleType("nki"); nki.isa = isa
    monkeypatch.setitem(sys.modules, "nki", nki)
    monkeypatch.setitem(sys.modules, "nki.isa", isa)
    import nkibench

    def fake_count(kernel, args):
        counter = dict(bytes=0, transfers=0)
        orig = isa.dma_copy
        def counting(dst=None, src=None, **kw):
            counter["bytes"] += src.nbytes
            return orig(dst=dst, src=src, **kw)
        isa.dma_copy = counting
        try:
            script(isa)
        finally:
            isa.dma_copy = orig
        return c, counter
    monkeypatch.setattr(nkibench, "simulate_and_count", fake_count)


def test_attribution_sums_to_count(monkeypatch):
    a = np.ones((4, 4), "float32"); b = np.arange(16, dtype="float32").reshape(4, 4); out = np.zeros((4, 4), "float32")
    A, B = T(a, "MemoryRegion.private_hbm", 0), T(b, "MemoryRegion.private_hbm", 1)
    O = T(out, "MemoryRegion.shared_hbm", 5)
    S = T(a, "MemoryRegion.sbuf", 9)

    def script(isa):
        for _ in range(4):
            isa.dma_copy(dst=S, src=A.view())   # lhsT: 4 half-reads = 2x size
        isa.dma_copy(dst=S, src=B)              # rhs once
        isa.dma_copy(dst=O, src=S)              # output write
    fake_env(monkeypatch, script, a, b, out)
    o, counted, at = traffic.simulate_and_attribute(None, [a, b], ["lhsT", "rhs"])
    assert at["_total"] == counted["bytes"]
    assert at["lhsT"]["reads_x"] == pytest.approx(2.0)
    assert at["rhs"]["reads_x"] == pytest.approx(1.0)
    assert at["output"]["write_bytes"] == out.nbytes
    assert "lhsT" in traffic.explain(at, [a, b]) and "2.0x" in traffic.explain(at, [a, b])


def test_explain_cases():
    ok = {"lhsT": dict(reads_x=1.0), "rhs": dict(reads_x=1.0), "_total": 1}
    assert traffic.explain(ok) == ""
    s = traffic.explain({"lhsT": dict(reads_x=1.0), "rhs": dict(reads_x=8.0)})
    assert s.startswith("`rhs` was read 8.0x its size") and "outside the loop" in s
    assert "written 3.0x" in traffic.explain({"output": dict(reads_x=3.0)})
