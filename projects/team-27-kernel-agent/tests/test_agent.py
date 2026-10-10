"""The loop, driven by a scripted model: every path that a live run can take, without a server."""
from pathlib import Path

from kagent.agent import ScriptedModel, solve
from kagent.levels import LEVELS

K = Path(__file__).parent.parent / "kernels"
GOOD = (K / "hand/l1_relu_affine.py").read_text()
# what Qwen3-8B actually returned on the first live call (E6): right numbers, one violation
ASARRAY = GOOD.replace("    M, N = x.shape\n", "    x = np.asarray(x, dtype=np.float32)\n    M, N = x.shape\n")
MASK = (K / "broken/l1_mask.py").read_text()


def block(code):
    return f"Here you go:\n```python\n{code}```\n"


def run(replies, **kw):
    m = ScriptedModel([block(r) if r.startswith("import") else r for r in replies])
    res = solve(LEVELS[1], m, say=lambda *_: None, **kw)
    return res, m.prompts


def test_first_prompt_has_no_rules_list():
    _, prompts = run([GOOD])
    p = prompts[0]
    assert "kernel(x, a, b)" in p and "np.sum" not in p and "fancy" not in p


def test_violation_repaired_in_one_round():
    res, prompts = run([ASARRAY, GOOD])
    assert res.status == "verified" and res.attempts == 2 and res.confidence > 0.9
    assert "Remove `np.asarray`" in prompts[1] and "x = np.asarray" in prompts[1]
    assert "Already tried" not in prompts[1]  # no ledger until a failure class repeats


def test_no_code_never_reaches_checker_and_prompt_changes():
    res, prompts = run(["<TRUNCATED>```python\ndef kernel(x, a, b):\n    M, N", "", GOOD])
    assert res.status == "verified" and res.attempts == 3
    assert prompts[0] != prompts[1]  # a greedy model given the same prompt repeats itself
    assert res.keys[:2] == ["reply:truncated", "reply:empty"]


def test_repeat_brings_ledger_then_stops():
    res, prompts = run([MASK, MASK, MASK, GOOD], patience=3)
    assert res.status == "stuck" and res.attempts == 3 and res.final_key == "rule:fancy-index"
    assert "Already tried" in prompts[2] and "Do something different" in prompts[2]
    assert "repeated an earlier version unchanged" in prompts[2]


def test_passing_dev_but_failing_holdout_is_not_verified():
    # correct on every dev shape, wrong on a width the dev set never uses: overfitting to the eval set
    cheat = GOOD.replace("    return out\n", "    if N == 1543:\n        out[0, 0] = 1e9\n    return out\n")
    res, _ = run([cheat])
    assert res.status == "dev-only" and res.confidence < 0.5


def test_naive_mode_sends_the_report_verbatim_and_resends_on_no_code():
    m = ScriptedModel([block(ASARRAY), "", block(GOOD)])
    res = solve(LEVELS[1], m, say=lambda *_: None, mode="naive")
    assert res.status == "verified" and res.attempts == 3
    assert "A checker reports:" in m.prompts[1] and "np.asarray [escape]" in m.prompts[1]
    assert "Remove `np.asarray`" not in m.prompts[1]  # no translated instruction
    assert m.prompts[2] == m.prompts[1]               # no reply gate: the same prompt again


def test_runs_use_different_wordings_with_the_same_task():
    p0, p1 = (ScriptedModel([block(GOOD)]) for _ in range(2))
    solve(LEVELS[1], p0, say=lambda *_: None, wording=0)
    solve(LEVELS[1], p1, say=lambda *_: None, wording=1)
    assert p0.prompts[0] != p1.prompts[0]
    assert LEVELS[1].task in p0.prompts[0] and LEVELS[1].task in p1.prompts[0]


def test_v3_structural_directives_allow_a_loop_rewrite():
    # E17: a structural fix must not be told to "keep everything else identical"
    from kagent.agent import repair_prompt
    from kagent.translate import Directive
    lv = LEVELS[3]
    structural = render_text(repair_prompt(lv, GOOD, Directive("numeric:tile-overwrite", "x"), []))
    local = render_text(repair_prompt(lv, GOOD, Directive("rule:escape", "x"), []))
    assert "Rewrite the loop structure" in structural and "keep everything else identical" not in structural
    assert "keep everything else identical" in local


def test_v3_crash_directive_names_the_fix():
    from kagent.harness import verify
    from kagent.translate import translate
    src = GOOD.replace("j1 = min(j0 + 512, N)", "j1 = j0 + 512").replace(
        "out = np.empty((M, N), dtype=np.float32)", "out = np.empty((M, N), dtype=np.float32)\n    t = np.zeros(4)")
    bad = (K / "hand/l2_row_sum.py").read_text().replace("acc += x[i0:i1, j]", "acc += x[i0:i1, j0:j1][:, j]")
    d = translate(verify(LEVELS[2], bad))
    assert d.key == "crash:IndexError" and "partial tile" in d.instruction and "local positions" in d.instruction


def render_text(parts):
    return "\n\n".join(t for _, t in parts)


def test_missing_organizers_checker_is_not_verified(monkeypatch):
    # red team PR #9: the final gate used to pass when kernelbench.py was absent
    from kagent import judge
    monkeypatch.setattr(judge, "available", lambda: False)
    res, _ = run([GOOD])
    assert res.status == "verified-unjudged" and res.confidence < 0.95


def _escapes(n):
    lines = ["    x = np.asarray(x)\n", "    a = np.asarray(a)\n", "    b = np.asarray(b)\n"][:n]
    return GOOD.replace("    M, N = x.shape\n", "".join(lines) + "    M, N = x.shape\n")


def test_same_class_with_falling_violation_count_is_not_stuck():
    # live v3 L9 went 13 -> 12 -> 10 -> 6 -> 5 violations; a falling count is progress, not a loop
    res, _ = run([_escapes(3), _escapes(2), _escapes(1), GOOD], patience=3)
    assert res.keys == ["rule:escape"] * 3
    assert res.status == "verified" and res.attempts == 4
