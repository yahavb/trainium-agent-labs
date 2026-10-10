"""LOOKUP: how a role pulls documentation itself, instead of the harness pushing it into the prompt.

A prompt carries the problem statement and an index of names. Any role may answer with one line,

    LOOKUP: nl.sum, t.permute, rules

and gets that documentation from the retriever, then is asked again. The number of rounds is capped
per role. Pushing an API card into every prompt is what sent level 1 to nc_matmul (100% of attempts),
and the more a first prompt explains, the closer it comes to handing over the answer; this way the
system decides what it needs, and the log shows what it asked for.
"""

import re

from agents2.llm import Section

LOOKUP_RE = re.compile(r"(?im)^\s*\**LOOKUP\**\s*:\s*(.+)$")
MAX_NAMES = 6

# The reply format is the last thing in every prompt, and it offers the lookup as one of two ways to
# reply. A separate lookup paragraph placed before "Reply with ONE python code block" read as two
# competing formats, and an 8B model follows the last one: run 1 of the pull design made 4 lookups.
CHOICE = ("Reply in one of two ways.\n"
          "- To read documentation before you answer, reply with only this one line:\n"
          "  LOOKUP: <up to 6 nisa / nl names or tile methods (t.xxx), or the topic 'rules'>\n"
          "- Otherwise, give your answer. {answer}")
ANSWER_NOW = "You have the documentation you asked for. No more lookups: answer now. {answer}"


def parse_lookup(text):
    """The names asked for, or [] when the reply is an answer rather than a lookup. A reply that
    already holds code or a plan is an answer, even if it also says LOOKUP."""
    if not text or "```" in text or re.search(r"(?im)^\s*\**(APPROACH|CHANGE|CAUSE)\**\s*:", text):
        return []
    m = LOOKUP_RE.search(text)
    if not m:
        return []
    names = [n.strip(" `*.,;()") for n in re.split(r"[,;]| and ", m.group(1))]
    return [n for n in names if n][:MAX_NAMES]


def index_section(retriever, level):
    """The names, without an invitation: whether a lookup is allowed is said in the reply format,
    so a prompt that allows none (--no-lookup, the last round) doesn't contradict itself."""
    return Section("index", "NKI names:\n" + retriever.api_map(level), priority=3)


def _sections(sections, retriever, level, pulled, final, looked):
    """The caller's sections, with the documentation pulled so far before the last one (the reply
    format), and the format rewritten: two ways to reply, or answer now."""
    secs = list(sections)
    docs = retriever.lookup(pulled, level) if pulled else ""
    if docs:
        secs.insert(len(secs) - 1, Section("docs", "Documentation you looked up:\n" + docs, priority=6))
    last = secs[-1]
    if not final:
        text = CHOICE.format(answer=last.text)
    elif looked:
        text = ANSWER_NOW.format(answer=last.text)
    else:
        text = last.text
    secs[-1] = Section(last.name, text, last.priority, last.required)
    return secs


def ask(llm, role, sections, retriever, level, rounds, ledger, events, pulled=None, tags=None, **kw):
    """Ask, answering LOOKUP requests for up to `rounds` rounds. Returns (text, meta, names pulled).

    `pulled` seeds the documentation already fetched for this work (e.g. by the planner), shown as
    looked-up documentation from the start. A lookup asked for after "answer now" gets its
    documentation and one more call; a reply that is still a lookup goes back to the caller, which
    treats it as no answer. With rounds=0 nothing is looked up at all."""
    pulled = list(pulled or [])
    r, final, forced = 0, rounds <= 0, False
    while True:
        secs = _sections(sections, retriever, level, pulled, final, looked=r > 0)
        text, meta = llm.chat(role, secs, tags=dict(tags or {}, lookup_round=r), **kw)
        ledger.note_call(meta)
        names = [] if meta.get("error") else parse_lookup(text)
        if not names or forced or rounds <= 0:
            break
        new = [n for n in names if n not in pulled]
        if events:
            events.write("lookup", role=role, level=level, round=r, names=names, new=new, after_final=final,
                         **{k: v for k, v in (tags or {}).items() if k in ("thread", "approach", "mode")})
        forced = final                           # it was told to answer: one last call, then stop
        pulled += new
        r += 1
        final = final or not new or r >= rounds  # nothing new asked for: it has what it needs
    return text, meta, pulled
