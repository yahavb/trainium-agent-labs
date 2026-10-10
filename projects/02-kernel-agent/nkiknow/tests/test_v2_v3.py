import json
import agent


class FakeResp:
    def __init__(self, msg):
        self.status_code = 200
        self._m = msg
    def json(self):
        return {"choices": [{"message": self._m, "finish_reason": "stop"}]}


class A:
    base = "http://x/v1"; model = "m"; max_tokens = 100; context = 4096; think = False


def setup(monkeypatch, v, replies):
    """replies: list of message dicts returned in order; returns the list of request bodies."""
    import httpx
    monkeypatch.setattr(agent, "KNOWLEDGE", v)
    agent._DOC_CACHE.clear()
    bodies = []
    def post(url, json=None, **k):
        bodies.append(json)
        return FakeResp(replies[min(len(bodies) - 1, len(replies) - 1)])
    monkeypatch.setattr(httpx, "post", post)
    from nkiknow import retrieve
    monkeypatch.setattr(retrieve, "lookup", lambda q, max_tokens=300, **k: f"DOC({q}) [source: a.md#x]")
    return bodies


def test_v0_prompt_parity(monkeypatch):
    monkeypatch.setattr(agent, "KNOWLEDGE", "v0")
    assert agent.with_knowledge(4, "BODY") == "BODY"
    assert agent.ask_one is not None


def test_v2_appends_docs_on_failure(monkeypatch, tmp_path):
    setup(monkeypatch, "v2", [])
    monkeypatch.setattr(agent, "grade", lambda s, l: (0.1, {}, "TypeError: bad arg to nl.matmul\nmore"))
    class B(A): rounds = 1; samples = 1; offline = True; give_up_after = 4; terse = 0
    log = tmp_path / "l.jsonl"
    with open(log, "w") as f:
        agent.solve(B, 1, f)
    rec = json.loads(open(log).readline())
    assert "Relevant NKI documentation:\nDOC(TypeError: bad arg to nl.matmul)" in rec["feedback"]
    assert rec["sections"]["docs"] > 0
    # v1 gets no docs
    monkeypatch.setattr(agent, "KNOWLEDGE", "v1")
    assert agent._kv() == 1


def test_v2_cache_and_empty(monkeypatch):
    setup(monkeypatch, "v2", [])
    from nkiknow import retrieve
    n = []
    monkeypatch.setattr(retrieve, "lookup", lambda q, max_tokens=300: n.append(q) or "")
    assert agent.docs_note("TypeError: x") == "" and agent.docs_note("TypeError: x") == ""
    assert len(n) == 1


def test_v3_prompt_paragraph(monkeypatch):
    monkeypatch.setattr(agent, "KNOWLEDGE", "v3")
    p = agent.with_knowledge(1, "BODY")
    assert "LOOKUP <api name or topic>" in p and p.endswith("BODY")


def test_v3_lookup_then_code(monkeypatch, tmp_path):
    bodies = setup(monkeypatch, "v3", [{"content": "LOOKUP nl.matmul\nLOOKUP nl.load"},
                                       {"content": "```python\nx=1\n```"}])
    log = tmp_path / "l.jsonl"
    with open(log, "w") as f:
        out = agent.ask_one(A, "P", 4, 0, f)
    assert "x=1" in out and len(bodies) == 2
    assert bodies[1]["messages"][0]["content"].startswith("P\n\nDocumentation you requested:\n")
    assert "DOC(nl.matmul)" in bodies[1]["messages"][0]["content"]
    rec = json.loads(open(log).readline())
    assert rec["type"] == "lookup" and rec["queries"] == ["nl.matmul", "nl.load"]


def test_v3_caps_exchanges(monkeypatch):
    bodies = setup(monkeypatch, "v3", [{"content": "LOOKUP a"}])
    agent.ask_one(A, "P", 4, 0, None)
    assert len(bodies) == 3          # 1 ask + 2 lookup exchanges, then proceed


def test_v3_ignores_lookup_with_code(monkeypatch):
    bodies = setup(monkeypatch, "v3", [{"content": "LOOKUP a\n```python\nx=1\n```"}])
    out = agent.ask_one(A, "P", 4, 0, None)
    assert len(bodies) == 1 and "x=1" in out


def test_v3_total_cap(monkeypatch):
    setup(monkeypatch, "v3", [])
    from nkiknow import retrieve
    monkeypatch.setattr(retrieve, "lookup", lambda q, max_tokens=300: "w" * (max_tokens * 4))
    out = agent.run_lookups(["a", "b", "c"])
    assert len(out) < 700 * 4 + 200


def test_downloads_assertion(monkeypatch):
    setup(monkeypatch, "v3", [])
    from nkiknow import retrieve
    import pytest
    monkeypatch.setattr(retrieve, "lookup", lambda q, max_tokens=300: "x [source: docs/downloads/k.py#a]")
    with pytest.raises(AssertionError):
        agent.doc_lookup("q")


def test_native_tools(monkeypatch):
    tc = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"query": "nl.matmul"})}}
    bodies = setup(monkeypatch, "v3", [{"content": "", "tool_calls": [tc]}, {"content": "```python\ny=2\n```"}])
    monkeypatch.setattr(agent, "NATIVE_TOOLS", True)
    out = agent.ask_one(A, "P", 4, 0, None)
    assert "y=2" in out and bodies[0]["tools"][0]["function"]["name"] == "lookup"
    msgs = bodies[1]["messages"]
    assert msgs[-1]["role"] == "tool" and "DOC(nl.matmul)" in msgs[-1]["content"]


def test_native_no_tool_support(monkeypatch):
    import httpx
    setup(monkeypatch, "v3", [])
    monkeypatch.setattr(agent, "NATIVE_TOOLS", True)
    calls = []
    class R:
        def __init__(s, c, m): s.status_code = c; s.text = "no tools"; s._m = m
        def json(s): return {"choices": [{"message": s._m, "finish_reason": "stop"}]}
    def post(url, json=None, **k):
        calls.append(json)
        return R(400, None) if "tools" in json else R(200, {"content": "```python\nz=3\n```"})
    monkeypatch.setattr(httpx, "post", post)
    assert "z=3" in agent.ask_one(A, "P", 4, 0, None)
