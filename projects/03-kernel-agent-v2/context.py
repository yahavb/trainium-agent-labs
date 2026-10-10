#!/usr/bin/env python3
"""
context.py — the token budget, and what happens when the prompt outgrows it.

The design constraint that is NOT a constraint: the challenge's 8192-token wall is one
configuration point here, not the shape of the design. `--context auto` asks the model
server what its max_model_len is and budgets against that; if the server is restarted
with more room, the same loop simply compacts less often. Prompts stay compact anyway,
measured for a different reason: on BOTH workshop endpoints a long prompt makes the model
reason in its hidden channel instead of answering (project 02, twice).

Mechanics, borrowed from Claude Code's auto-compaction and made measurable:

  * every piece of the prompt is a COMPONENT with a priority and a compactor
  * build() renders the prompt under a token budget (chars/4 -- good enough to plan with;
    the authoritative number is the API's usage.prompt_tokens, logged per attempt)
  * over budget -> compact the lowest-priority component one step, log the event, repeat.
    Compaction never touches the four load-bearing pieces: tool instructions, reference,
    the current kernel, and the one repair instruction.
  * events are returned so the trace can show WHERE the tokens went and WHAT got dropped
    -- the challenge calls that graph the single most interesting artifact.

Token estimate note: chars/4 is deliberately crude. Decisions are made against it; the
trace records the API's real count next to it, so the estimate's error is itself visible
in the dashboard.
"""

CHARS_PER_TOKEN = 4


def est_tokens(text):
    return len(text or "") // CHARS_PER_TOKEN


# Compaction ladders, per component. Each entry is a (name, fn) step; step 0 is the full
# text. A component compacts one step at a time, lowest priority first -- so the ledger
# collapses before the docs do, and the docs before the reference, which never collapses.

def _compact_ledger(lines):
    if len(lines) > 6:
        return lines[:2] + ["  ..."] + lines[-3:]
    return lines


def _compact_scratch(lines):
    if len(lines) > 6:
        return lines[-6:]
    return lines


COMPACTORS = {
    "ledger": _compact_ledger,     # one line per failed attempt
    "scratch": _compact_scratch,   # REPL transcript
}


class Context:
    """The prompt under a budget. Components are (name -> text); priorities decide the
    eviction order; NEVER_LOSE is protected absolutely."""

    # eviction order: first compacted first. Load-bearing context is last.
    ORDER = ["scratch", "ledger", "memory", "docs", "reference", "kernel", "feedback"]

    def __init__(self, budget_tokens):
        self.budget = budget_tokens
        self.texts = {}
        self.floors = {}      # component -> minimum worth keeping (compaction index)
        self.events = []

    def set(self, name, text, floor=0):
        """floor: how far this component may compact (index into its ladder; 0 = not
        compactible beyond full text, -1 = never drop)."""
        self.texts[name] = text or ""
        self.floors[name] = floor

    def used(self):
        return sum(est_tokens(t) for t in self.texts.values())

    def build(self):
        """Return (prompt_parts, events). Compacts until under budget; every action is
        logged as an event for the trace."""
        self.events = []
        for name in self.ORDER:
            while self.used() > self.budget and self.floors.get(name, 0) != -1 \
                    and name in self.texts:
                before = est_tokens(self.texts[name])
                new = self._compact_one(name)
                if new is None:                       # ladder exhausted
                    break
                self.texts[name] = new
                after = est_tokens(new)
                self.events.append(dict(component=name, action="compact",
                                        before_tokens=before, after_tokens=after))
                if after >= before:                   # compactor made no progress
                    break
        # last resort: drop the optional components entirely, in order
        for name in self.ORDER:
            if self.used() <= self.budget:
                break
            if self.floors.get(name, 0) == -1 or name not in self.texts:
                continue
            before = est_tokens(self.texts[name])
            del self.texts[name]
            self.events.append(dict(component=name, action="dropped",
                                    before_tokens=before, after_tokens=0))
        return self.texts, self.events

    def _compact_one(self, name):
        text = self.texts[name]
        lines = text.splitlines()
        fn = COMPACTORS.get(name)
        if fn and len(lines) > 2:
            compacted = fn(lines)
            if len(compacted) < len(lines):
                return "\n".join(compacted)
        # generic shrink: keep the head and the tail
        if len(lines) > 8:
            return "\n".join(lines[:4] + ["  [...]"] + lines[-4:])
        return None   # cannot compact further

    def render(self, header=""):
        """Ordered prompt text with section markers, under budget."""
        parts, _ = self.build()
        blocks = []
        if header:
            blocks.append(header)
        for name in ("memory", "reference", "docs", "kernel", "feedback", "ledger", "scratch"):
            if name in parts and parts[name].strip():
                blocks.append(parts[name])
        return "\n\n".join(blocks)


def token_report(parts, usage=None):
    """Per-part token sizes for the trace: {'total': n, 'parts': {...}} plus the API's own
    numbers when it reported them."""
    rep = dict(total=sum(est_tokens(t) for t in parts.values()),
               parts={k: est_tokens(v) for k, v in parts.items() if v},
               estimate=True)
    if usage:
        rep["api_prompt_tokens"] = usage.get("prompt_tokens")
        rep["api_completion_tokens"] = usage.get("completion_tokens")
        rep["estimate"] = False if usage.get("prompt_tokens") else True
    return rep
