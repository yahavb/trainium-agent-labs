import json
import httpx
import agent


class A:
    model = "m"; base = "http://x/v1"; max_tokens = 100; context = 4096; think = False
    sampling = "organizer"


class Resp:
    status_code = 200
    def json(self):
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def capture(monkeypatch):
    seen = []
    monkeypatch.setattr(httpx, "post", lambda url, json=None, **k: seen.append(json) or Resp())
    return seen


def reset(monkeypatch, **kw):
    for k, v in dict(KNOWLEDGE="v1", GUIDE=True, SYSTEM_GUIDE=False, TAGGED=False,
                     EXAMPLES3=False, DIVERSE_SAMPLES=False).items():
        monkeypatch.setattr(agent, k, kw.get(k, v))


def test_sampling(monkeypatch):
    seen = capture(monkeypatch); reset(monkeypatch)
    agent.ask(A, "p")
    b = seen[-1]
    assert (b["temperature"], b["top_p"]) == (0.6, 0.95) and "top_k" not in b
    class Q(A): sampling = "qwen"
    agent.ask(Q, "p")
    b = seen[-1]
    assert (b["temperature"], b["top_p"], b["top_k"], b["min_p"], b["presence_penalty"]) == (0.7, 0.8, 20, 0, 1.0)


def test_system_guide(monkeypatch):
    seen = capture(monkeypatch); reset(monkeypatch, SYSTEM_GUIDE=True)
    p = agent.with_knowledge(1, "BODY")
    assert p.startswith("BODY\n\n" + agent.PLAN_LAST) and "named exactly `tensor_avgpool_kernel`" in p   # P5b: plan + required name last
    agent.ask(A, p)
    m = seen[-1]["messages"]
    assert m[0]["role"] == "system" and "NKI" in m[0]["content"] and m[1]["role"] == "user" and m[1]["content"].startswith("BODY")
    reset(monkeypatch)   # off: guide prepended, single user message
    agent.ask(A, agent.with_knowledge(1, "BODY"))
    m = seen[-1]["messages"]
    assert len(m) == 1 and m[0]["role"] == "user" and m[0]["content"].endswith("decorated with `@nki.jit`.")
    reset(monkeypatch, SYSTEM_GUIDE=True, GUIDE=False)   # needs --guide
    assert agent.system_guide() == ""


def test_budget_counts_system_guide_and_message_history(monkeypatch):
    seen = capture(monkeypatch); reset(monkeypatch, SYSTEM_GUIDE=True)
    class Small(A): max_tokens = 3000
    agent.ask(Small, "short")
    expected = min(3000, max(256, Small.context - (len(agent.nki_guide()) + len("short")) // 4 - 64))
    assert seen[-1]["max_tokens"] == expected < 3000

    reset(monkeypatch, SYSTEM_GUIDE=False)
    messages = [{"role": "user", "content": "x" * 8000}]
    agent.ask(Small, "short", messages=messages)
    assert seen[-1]["max_tokens"] == min(3000, max(256, Small.context - 8000 // 4 - 64))


def test_tagged(monkeypatch):
    reset(monkeypatch, TAGGED=True)
    g = agent.with_knowledge(1, agent.tag_task(agent.first_prompt(1)))
    assert g.startswith("<nki_guide>") and "</nki_guide>" in g and "<task>" in g and "</task>" in g
    assert agent.PLAN_LAST in g and g.rstrip().endswith("decorated with `@nki.jit`.")   # P5b: required name is the last line
    r = agent.repair_prompt(1, "SRC", "FB", ledger="- a")
    assert "<previous_kernel>" in r and "<checker_feedback>\nFB\n" in r and "<already_tried>\n- a\n</already_tried>" in r
    assert r.endswith("ONE python code block.") and r.index("</already_tried>") < r.index("Change exactly")
    reset(monkeypatch)
    plain = agent.repair_prompt(1, "SRC", "FB", ledger="- a")
    assert "<" not in plain.replace("<", "", 0).split("FB")[0] and "These approaches have already failed" in plain
    assert plain.endswith("- a")


def test_examples3(monkeypatch):
    reset(monkeypatch, EXAMPLES3=True)
    g = agent.nki_guide()
    assert g.count("def row_sum_kernel") == 1 and g.count("def copy_t_kernel") == 1 and g.count("def scale_shift_kernel") == 1 and "different operations; adapt the patterns" in g
    reset(monkeypatch)
    assert "row_sum_kernel" in agent.nki_guide()   # P2: guide now carries the three examples


def test_system_logged(monkeypatch, tmp_path):
    reset(monkeypatch, SYSTEM_GUIDE=True)
    class B: rounds=1; samples=1; offline=True; give_up_after=4; terse=0
    with open(tmp_path / "a", "w") as f:
        agent.solve(B, 1, f)
    rec = json.loads(open(tmp_path / "a").readline())
    assert "What NKI is" in rec["system"] and "What NKI is" not in rec["prompt"] and len(rec["system"]) > 1000


def test_diverse_sample_prompts_are_distinct_and_keep_final_instruction():
    prompt = "Plan generally.\nReply with ONE python code block."
    prompts = agent.diverse_sample_prompts(prompt, 4, enabled=True)
    assert len(set(prompts)) == 4
    assert all(p.endswith("Reply with ONE python code block.") for p in prompts)
    assert all(p.index("Planning focus") < p.index("Reply with") for p in prompts)


def test_diverse_samples_log_exact_offline_prompt_per_reply(monkeypatch, tmp_path):
    reset(monkeypatch, DIVERSE_SAMPLES=True)
    monkeypatch.setattr(agent, "first_prompt", lambda level, terse: "Plan.\nReply with code.")
    monkeypatch.setattr(agent, "with_knowledge", lambda level, prompt: prompt)
    monkeypatch.setattr(agent, "offline_answers", lambda level, n, rnd: [f"reply-{i}" for i in range(n)])
    monkeypatch.setattr(agent, "grade", lambda src, level: (0.0, {}, "feedback"))

    class B: rounds=1; samples=4; offline=True; give_up_after=4; terse=0; diverse_samples=True
    path = tmp_path / "diverse.jsonl"
    with path.open("w") as f:
        agent.solve(B, 1, f)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    expected = agent.diverse_sample_prompts("Plan.\nReply with code.", 4, enabled=True)
    assert [r["prompt"] for r in records] == expected
    assert [r["reply"] for r in records] == [f"reply-{i}" for i in range(4)]


def test_diverse_samples_off_by_default_preserves_prompt_logging(monkeypatch, tmp_path):
    reset(monkeypatch)
    monkeypatch.setattr(agent, "first_prompt", lambda level, terse: "Plan.\nReply with code.")
    monkeypatch.setattr(agent, "with_knowledge", lambda level, prompt: prompt)
    monkeypatch.setattr(agent, "offline_answers", lambda level, n, rnd: [f"reply-{i}" for i in range(n)])
    monkeypatch.setattr(agent, "grade", lambda src, level: (0.0, {}, "feedback"))

    class B: rounds=1; samples=2; offline=True; give_up_after=4; terse=0
    path = tmp_path / "default.jsonl"
    with path.open("w") as f:
        agent.solve(B, 1, f)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["prompt"] for r in records] == ["Plan.\nReply with code."] * 2


def test_diverse_samples_rebuild_after_round_feedback(monkeypatch, tmp_path):
    reset(monkeypatch, DIVERSE_SAMPLES=True)
    monkeypatch.setattr(agent, "first_prompt", lambda level, terse: "Plan.\nReply with code.")
    monkeypatch.setattr(agent, "with_knowledge", lambda level, prompt: prompt)
    monkeypatch.setattr(agent, "offline_answers",
                        lambda level, n, rnd: [f"def sample_{rnd}_{i}():\n    pass" for i in range(n)])
    monkeypatch.setattr(agent, "grade", lambda src, level: (0.0, {}, "ROUND_ZERO_FEEDBACK"))

    class B: rounds=2; samples=4; offline=True; give_up_after=4; terse=0
    path = tmp_path / "feedback.jsonl"
    with path.open("w") as f:
        agent.solve(B, 1, f)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    round_one = [r["prompt"] for r in records if r["round"] == 1]
    assert len(round_one) == 4
    assert all("ROUND_ZERO_FEEDBACK" in prompt for prompt in round_one)
    assert len(set(round_one)) == 4
