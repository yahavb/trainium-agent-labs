import json
import agent


def test_card_levels(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "KNOWLEDGE", "v1")
    assert agent.matmul_card(2) == ""
    assert agent.matmul_card(3).startswith("How matrix multiplication works in NKI")
    p = agent.with_knowledge(4, "BODY")
    assert p.startswith("How matrix multiplication works in NKI") and p.endswith("BODY") and agent.SECTIONS["card"] > 0
    monkeypatch.setattr(agent, "KNOWLEDGE", "v0")
    assert agent.with_knowledge(4, "BODY") == "BODY" and agent.SECTIONS["card"] == 0


def test_seed_prompt(tmp_path):
    s = agent.seed_prompt(4, "def k(): pass")
    assert "def k(): pass" in s and "This kernel is correct. Make it move fewer HBM bytes while staying correct." in s


def test_lookup_note_only_v1(monkeypatch):
    monkeypatch.setattr(agent, "KNOWLEDGE", "v0")
    assert agent._lookup_note("TypeError: x", "") == ""
    from nkiknow import api_lookup
    monkeypatch.setattr(api_lookup, "signatures_for", lambda e, c: "SIG")
    monkeypatch.setattr(agent, "KNOWLEDGE", "v1")
    agent._reset_sections()
    assert agent._lookup_note("e", "c") == " SIG" and agent.SECTIONS["lookup"] == 3


def test_log_record_has_sections(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "KNOWLEDGE", "v1")
    log = tmp_path / "a.jsonl"
    class A: rounds=1; samples=1; offline=True; give_up_after=4; terse=0
    with open(log, "w") as f:
        agent.solve(A, 1, f)
    rec = json.loads(open(log).readline())
    assert rec["knowledge"] == "v1" and set(rec["sections"]) == {"card", "lookup", "traffic", "code", "feedback"}
