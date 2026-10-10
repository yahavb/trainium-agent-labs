"""DEBUGGER: after a failed check, one named change for the coder, or "the approach is wrong".

Rules first. For the error kinds a rule understands (an invented name, a wrong keyword, a copy whose
sizes disagree, more than 128 partitions...), the change is built from the real signature, the closest
real names and a checked example, with no model call. The model is asked only for wrong values,
unclassified errors, and a failure that came back after a rule's change.

The old loop's feedback said what was wrong and never what to do, and it never questioned the
algorithm: level 1 stayed at 0.30 for 168 attempts. Here the model may answer APPROACH WRONG, which ends
the thread and sends the manager back to the planner.
"""

import re

from agents2 import errors
from agents2.llm import Section, head, numbered

FORMAT = ("Reply with exactly three lines:\n"
          "CAUSE: <what is wrong>\n"
          "LINE: <the line number>\n"
          "CHANGE: <the one change to make, concretely>\n"
          "If the plan itself cannot produce the right result, reply with one line instead:\n"
          "APPROACH WRONG: <why>")


def parse_change(text):
    clean = re.sub(r"[*`#]", "", text or "")
    m = re.search(r"(?im)^\s*APPROACH WRONG\s*:\s*(.+)$", clean)
    if m:
        return dict(approach_wrong=m.group(1).strip())
    get = lambda k: (re.search(rf"(?ims)^\s*{k}\s*:\s*(.*?)(?=^\s*(?:CAUSE|LINE|CHANGE)\s*:|\Z)", clean)
                     or [None, ""])[1].strip()
    change = get("CHANGE")
    if not change:
        change = re.sub(r"\s+", " ", clean).strip()[:500]
    line = re.search(r"\d+", get("LINE") or "")
    return dict(cause=get("CAUSE"), change=change, line=int(line.group()) if line else None)


class Debugger:
    def __init__(self, llm, retriever, cfg, enrich, fragment_note, events=None):
        self.llm, self.retriever, self.cfg = llm, retriever, cfg
        self.enrich, self.fragment_note, self.events = enrich, fragment_note, events

    def debug(self, level, plan, code, check, history, ledger, attempts, **tags):
        """Returns dict(path, kind, cause, change, names, example) or dict(approach_wrong=...).

        history: this thread's earlier (signature, change) pairs, newest last."""
        sig = (check["kind"], check.get("line"), head(check.get("error")))
        repeat = bool(history) and history[-1][0] == sig
        if not repeat:
            rule = errors.rule_change(check, self.retriever, level, self.enrich, self.fragment_note)
            if rule:
                return dict(path="rule", kind=check["kind"], **rule)
        out = self._ask_model(level, plan, code, check, history[-1][1] if repeat else None, ledger,
                              **tags)
        if out is None:                       # the request failed, or no answer: fall back to the report
            return dict(path="fallback", kind=check["kind"], cause=check.get("error"), names=[],
                        example="", change="Fix exactly what the check reports.")
        if out.get("approach_wrong") and attempts < 2:
            # Too early to give up on a plan: one failed check says little about the algorithm.
            out = dict(cause=check.get("error"), change="Fix exactly what the check reports.", line=None)
        if out.get("approach_wrong"):
            return dict(path="model", kind=check["kind"], approach_wrong=out["approach_wrong"])
        names = [n for n in re.findall(r"\b(?:nisa|nl)\.\w+|\.\w+(?=\()", out["change"])
                 if self.retriever.exists(n) and self.retriever.shown(n, level)]
        return dict(path="model", kind=check["kind"], cause=out.get("cause"), change=out["change"],
                    names=names[:2], example=self.fragment_note(check.get("error") or "", level))

    def _ask_model(self, level, plan, code, check, previous, ledger, **tags):
        err = check.get("error") or ""
        corner = check.get("corner")
        detail = ""
        if corner:
            detail = (f"Top-left corner of the output on {check.get('shape')}:\n"
                      f"got:      {corner['got']}\nexpected: {corner['want']}")
        names = [n for n in re.findall(r"\b(?:nisa|nl)\.\w+|\.\w+(?=\()", code)
                 if self.retriever.exists(n) and self.retriever.shown(n, level)]
        import nkibench
        from agents2 import lookup
        from agents2.planner import task_text
        sections = [
            Section("intro", "You are debugging an AWS Neuron NKI kernel. Find the cause and name ONE "
                             "change. Do not rewrite the kernel.", required=True),
            # Wrong values can only be judged against what the right values are.
            Section("task", task_text(nkibench.LEVELS[level], level),
                    priority=7 if check["kind"] in ("VALUES", "PARTIAL", "OTHER") else 1),
            Section("plan", "The plan it follows:\n" + plan.text(), priority=3),
            Section("code", "The kernel, with line numbers:\n" + numbered(code), required=True),
            Section("error", f"The check reports{' on ' + check['shape'] if check.get('shape') else ''}:"
                             f"\n{err[:900]}", required=True),
            Section("detail", detail, priority=5),
            Section("previous", f"This change was already tried, and the kernel still fails the same "
                                f"way (or came back unchanged). Name a DIFFERENT change, on a specific "
                                f"line:\n{previous.get('change', '')[:400]}" if previous else "", priority=6),
            Section("example", self.fragment_note(err, level), priority=3),
            lookup.index_section(self.retriever, level),
            Section("format", FORMAT, required=True),
        ]
        # The functions the kernel already calls are documented from the start (the kernel pulled
        # them); anything else the debugger wants, it looks up.
        text, meta, _ = lookup.ask(self.llm, "debugger", sections, self.retriever, level,
                                   self.cfg.lookup.get("debugger", 0), ledger, self.events,
                                   pulled=list(dict.fromkeys(names))[:6], tags=dict(level=level, **tags))
        if meta.get("error") or lookup.parse_lookup(text):
            return None          # failed, or still asking for docs after "answer now": no change
        return parse_change(text)
