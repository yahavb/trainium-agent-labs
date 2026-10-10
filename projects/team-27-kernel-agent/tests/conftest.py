import pytest


@pytest.fixture(autouse=True)
def organizers_checker_passes(monkeypatch, request):
    """The organizers' kernelbench.py lives on the seat pod, not in this repo: agent tests assume it
    passes, except the test that checks what happens when it is missing."""
    if "missing_organizers_checker" in request.node.name:
        return
    from kagent import judge
    monkeypatch.setattr(judge, "check", lambda level, src: (True, "organizers' checker: test stub"))
