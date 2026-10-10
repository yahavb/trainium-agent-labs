"""--knowledge v0 must leave prompts and grade feedback byte-identical to the organizer's agent.py.
golden_v0.json holds sha256 of the pristine outputs (levels 1-4, offline-style inputs). Cases that
need the simulator are checked only where nki imports."""
import json, os
import agent
from golden_cases import cases

GOLDEN = json.load(open(os.path.join(os.path.dirname(__file__), "golden_v0.json")))


def test_v0_is_default():
    assert agent.KNOWLEDGE == "v0" and agent.SEED_KERNEL is None


def test_v0_prompts_and_feedback_unchanged():
    got = cases(agent)
    assert len(got) >= 28
    bad = [k for k, v in got.items() if GOLDEN.get(k) != v]
    assert not bad, bad
