"""CODER: writes the kernel. Three modes, each a single fresh prompt:

    write    the problem statement, the plan, and the documentation the planner pulled
    apply    the debugger's one named change to the current kernel, with the docs its error pulled
    improve  the reviewer's one improvement to a correct kernel

Nothing is pushed into a first prompt beyond the problem statement. Of agent.py's static API card it
keeps only the organisers' complete example kernel (a copy: imports, @nki.jit, the output in HBM, a
tile in SBUF, dma_copy in and out, return), which computes nothing and so gives no algorithm away. The
card's list of functions is gone: its matmul paragraph sent 100% of level-1 attempts to nc_matmul.
Without the example, the pull design's last level-1 failures included nl.mean run straight on the HBM
input. The coder may LOOKUP more itself (agents2/lookup.py).

Repairs and improvements carry no index of names: the change names what to use, and the documentation
for those names comes with it. The coder can still look a name up.
"""

from agents2.llm import Section
from agents2.planner import hardware_note, task_text

REPLY = "Reply with ONE python code block containing the imports and the function. No prose."


def skeleton():
    """The organisers' example kernel, verbatim from the end of agent.API_CARD ("" if it moved)."""
    import agent
    card = getattr(agent, "API_CARD", "")
    i = card.find("A complete kernel looks like this:")
    return card[i:].strip() if i >= 0 else ""


def max_tokens(cfg, level):
    from agents2.config import CODER_MAX_TOKENS_BIG
    base = cfg.roles["coder"].max_tokens
    return max(base, CODER_MAX_TOKENS_BIG) if level >= 4 else base


class Coder:
    def __init__(self, llm, retriever, cfg, extract_code, events=None):
        self.llm, self.retriever, self.cfg = llm, retriever, cfg
        self.extract_code, self.events = extract_code, events

    def _ask(self, level, sections, mode, ledger, pulled=(), temperature=None, **tags):
        """One coder call. The coder may LOOKUP documentation first (cfg.lookup["coder"] rounds)."""
        from agents2 import lookup
        text, meta, _ = lookup.ask(self.llm, "coder", sections, self.retriever, level,
                                   self.cfg.lookup.get("coder", 0), ledger, self.events,
                                   pulled=list(pulled), tags=dict(level=level, mode=mode, **tags),
                                   temperature=temperature, max_tokens=max_tokens(self.cfg, level))
        return self.extract_code(text), meta

    def write(self, level, plan, ledger, **tags):
        """The problem statement, the organisers' example kernel, the plan, and the documentation the
        planner pulled (for the functions its plan names and whatever it looked up)."""
        from agents2 import lookup
        import nkibench
        spec = nkibench.LEVELS[level]
        sections = [
            Section("intro", "Write an AWS Neuron NKI kernel.", required=True),
            Section("task", task_text(spec, level), required=True),
            Section("hardware", hardware_note(level), required=True),
            Section("hint", f"Hint: {spec.get('notes')}" if self.cfg.hint and spec.get("notes") else "",
                    priority=5),
            Section("skeleton", skeleton() if self.cfg.skeleton else "", priority=5),
            Section("plan", "Follow this plan:\n" + plan.text(), required=True),
            Section("aws", self.retriever.aws_excerpt(plan.calls, level), priority=1),
            lookup.index_section(self.retriever, level),
            Section("reply", REPLY, required=True),
        ]
        pulled = list(dict.fromkeys(list(plan.calls) + list(plan.docs)))
        return self._ask(level, sections, "write", ledger, pulled=pulled, **tags)

    def apply(self, level, plan, code, check, change, ledger, echo=False, **tags):
        """Make the debugger's change. `echo` is set when the last answer returned the kernel unchanged.
        The documentation shown is what the error pulled: the functions the debugger named."""
        import nkibench
        spec = nkibench.LEVELS[level]
        line = ""
        if check.get("line"):
            src = code.splitlines()
            if 0 < check["line"] <= len(src):
                line = f"The failing line is: `{src[check['line'] - 1].strip()}`"
        sections = [
            Section("intro", f"This NKI kernel for {spec['op']} fails a check.", required=True),
            Section("code", f"```python\n{code}\n```", required=True),
            Section("error", f"The check says: {(check.get('error') or '')[:600]}", required=True),
            Section("line", line, priority=5),
            Section("change", f"Make this change: {change['change']}", required=True),
            Section("example", change.get("example") or "", priority=4),
            Section("plan", "The plan this kernel follows:\n" + plan.text(), priority=3),
            Section("echo", "Your last answer returned this kernel unchanged. This time change it."
                    if echo else "", required=echo),
            Section("reply", "Keep the rest of the kernel as it is. " + REPLY, required=True),
        ]
        return self._ask(level, sections, "apply", ledger, pulled=change.get("names") or [],
                         temperature=(self.cfg.roles["coder"].temperature + 0.3) if echo else None,
                         **tags)

    def improve(self, level, plan, code, profile, change, ledger, echo=False, **tags):
        import nkibench
        spec = nkibench.LEVELS[level]
        sections = [
            Section("intro", f"This NKI kernel for {spec['op']} computes the right values on every test "
                             f"shape.",
                    required=True),
            Section("code", f"```python\n{code}\n```", required=True),
            Section("profile", profile, priority=4),
            Section("change", f"Improve it with this one change: {change}", required=True),
            Section("echo", "Your last answer returned this kernel unchanged. This time change it."
                    if echo else "", required=echo),
            Section("reply", "It must stay correct. " + REPLY, required=True),
        ]
        return self._ask(level, sections, "improve", ledger,
                         temperature=(self.cfg.roles["coder"].temperature + 0.3) if echo else None,
                         **tags)
